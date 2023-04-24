import argparse
import logging
import yaml
import os
import time

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoattack import AutoAttack
from stadv_eot.attacks import StAdvAttack
from torchvision import transforms
from torchvision import datasets
import utils
from utils import str2bool, get_accuracy, get_image_classifier, load_data
from torchvision.utils import save_image

from runners.diffpure_ddpm import Diffusion
from runners.diffpure_guided import GuidedDiffusion
from runners.diffpure_sde import RevGuidedDiffusion
from runners.diffpure_ode import OdeGuidedDiffusion
from runners.diffpure_ldsde import LDGuidedDiffusion

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')

    # LDSDE
    parser.add_argument('--sigma2', type=float, default=1e-3, help='LDSDE sigma2')
    parser.add_argument('--lambda_ld', type=float, default=1e-2, help='lambda_ld')
    parser.add_argument('--eta', type=float, default=5., help='LDSDE eta')
    parser.add_argument('--step_size', type=float, default=1e-3, help='step size for ODE Euler method')

    # adv
    parser.add_argument('--domain', type=str, default='celebahq', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='Eyeglasses', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=64)
    parser.add_argument('--attack_type', type=str, default='square')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--attack_version', type=str, default='rand')

    parser.add_argument('--num_sub', type=int, default=1000, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.07)

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config

class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # image classifier
        self.classifier = get_image_classifier(args.classifier_name).to(config.device)

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'sde':
            self.runner = RevGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ode':
            self.runner = OdeGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ldsde':
            self.runner = LDGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
        # and you may want to change the freq)
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        start_time = time.time()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        minutes, seconds = divmod(time.time() - start_time, 60)

        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)

        if counter % 5 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before classifier): {x_re.shape}')
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        out = self.classifier((x_re + 1) * 0.5)
        x_re = (x_re + 1) * 0.5
        save_decoded_CIFAR10(x_re.cpu().data, name='./adversarial_samples/cifar10_diff.png')

        self.counter += 1

        return out
    
#     def Attack(self, x):
        
        
def save_decoded_CIFAR10(img, name):
    img = img.view(img.size(0), 3, 32, 32)
    save_image(img, name)
    
def eval_autoattack(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    model_ = model

    attack_version = args.attack_version  # ['standard', 'rand', 'custom']
    if attack_version == 'standard':
        attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    elif attack_version == 'rand':
        attack_list = ['apgd-ce', 'apgd-dlr']
    elif attack_version == 'custom':
        attack_list = args.attack_type.split(',')
    else:
        raise NotImplementedError(f'Unknown attack version: {attack_version}!')
    print(f'attack_version: {attack_version}, attack_list: {attack_list}')  # ['apgd-ce', 'apgd-t', 'fab-t', 'square']

    # ---------------- apply the attack to classifier ----------------
    print(f'apply the attack to classifier [{args.lp_norm}]...')
    classifier = get_image_classifier(args.classifier_name).to(config.device)
#     adversary_resnet = AutoAttack(classifier, norm=args.lp_norm, eps=args.adv_eps,
#                                   version=attack_version, attacks_to_run=[],
#                                   log_path=f'{log_dir}/log_resnet.txt', device=config.device)
#     if attack_version == 'custom':
#         adversary_resnet.apgd.n_restarts = 1
#         adversary_resnet.fab.n_restarts = 1
#         adversary_resnet.apgd_targeted.n_restarts = 1
#         adversary_resnet.fab.n_target_classes = 9
#         adversary_resnet.apgd_targeted.n_target_classes = 9
#         adversary_resnet.square.n_queries = 5000
#     if attack_version == 'rand':
#         adversary_resnet.apgd.eot_iter = args.eot_iter
#         print(f'[classifier] rand version with eot_iter: {adversary_resnet.apgd.eot_iter}')
#     print(f'{args.lp_norm}, epsilon: {args.adv_eps}')

#     x_adv_resnet = adversary_resnet.run_standard_evaluation(x_val, y_val, bs=adv_batch_size)
#     print(f'x_adv_renet shape: {x_adv_resnet.shape}')
#     torch.save([x_adv_resnet, y_val], f'{log_dir}/x_adv_resnet_sd{args.seed}.pt')
    
    [x_adv_resnet, y_val] = torch.load(f'{log_dir}/x_adv_resnet_sd{args.seed}.pt')
    save_decoded_CIFAR10(x_adv_resnet.cpu().data, name='./adversarial_samples/cifar10_adv.png')

    y_pred0 = torch.argmax(model(x_val.cuda()),dim = 1)
    print('accuracy with clean examples to smooth classifier:')
    correct_num = np.where(y_val.cpu() == y_pred0.cpu())[0].shape[0]
    print(correct_num/len(y_val))
    
    y_pred1 = torch.argmax(classifier(x_adv_resnet.cuda()),dim = 1)
    print('accuracy without diffusion:')
    correct_num = np.where(y_val.cpu() == y_pred1.cpu())[0].shape[0]
    print(correct_num/len(y_val))
    y_pred2 = torch.argmax(model(x_adv_resnet.cuda()),dim = 1)
    
    print('accuracy after diffusion:')
    correct_num = np.where(y_val.cpu() == y_pred2.cpu())[0].shape[0]
    print(correct_num/len(y_val))
    
    
    # ---------------- apply the attack to sde_adv ----------------
    print(f'apply the attack to sde_adv [{args.lp_norm}]...')
    model_.reset_counter()
#     adversary_sde = AutoAttack(model, norm=args.lp_norm, eps=args.adv_eps,
#                                version=attack_version, attacks_to_run=[],
#                                log_path=f'{log_dir}/log_sde_adv.txt', device=config.device)
#     if attack_version == 'custom':
#         adversary_sde.apgd.n_restarts = 1
#         adversary_sde.fab.n_restarts = 1
#         adversary_sde.apgd_targeted.n_restarts = 1
#         adversary_sde.fab.n_target_classes = 9
#         adversary_sde.apgd_targeted.n_target_classes = 9
#         adversary_sde.square.n_queries = 5000
#     if attack_version == 'rand':
#         adversary_sde.apgd.eot_iter = args.eot_iter
#     print(f'[adv_sde] rand version with eot_iter: {adversary_sde.apgd.eot_iter}')
#     print(f'{args.lp_norm}, epsilon: {args.adv_eps}')

#     x_adv_sde = adversary_sde.run_standard_evaluation(x_val, y_val, bs=adv_batch_size)
#     print(f'x_adv_sde shape: {x_adv_sde.shape}')
#     torch.save([x_adv_sde, y_val], f'{log_dir}/x_adv_sde_sd{args.seed}.pt')
#     save_decoded_CIFAR10(x_adv_sde.cpu().data, name='./adversarial_samples/cifar10_diffusion_adv.png')

    [x_adv_sde, y_val] = torch.load(f'{log_dir}/x_adv_sde_sd{args.seed}.pt')
    
    y_pred3 = torch.argmax(classifier(x_adv_sde.cuda()),dim = 1)
    
    print('accuracy after attack diffusion to original classifier:')
    correct_num = np.where(y_val.cpu() == y_pred3.cpu())[0].shape[0]
    print(correct_num/len(y_val))
    
    
    y_pred4 = torch.argmax(model(x_adv_sde.cuda()),dim = 1)
    
    print('accuracy after attack diffusion:')
    correct_num = np.where(y_val.cpu() == y_pred4.cpu())[0].shape[0]
    print(correct_num/len(y_val))
          
def robustness_eval(args, config):
    middle_name = '_'.join([args.diffusion_type, args.attack_version]) if args.attack_version in ['stadv', 'standard', 'rand'] \
        else '_'.join([args.diffusion_type, args.attack_version, args.attack_type])
    log_dir = os.path.join(args.image_folder, args.classifier_name, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    print(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = utils.Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    adv_batch_size = args.adv_batch_size
    # load model
    print('starting the model and loader...')
    model = SDE_Adv_Model(args, config)


    # load data
    x_val, y_val = load_data(args, adv_batch_size)

    if args.attack_version in ['standard', 'rand', 'custom']:
        eval_autoattack(args, config, model, x_val, y_val, adv_batch_size, log_dir)

    logger.close()
    
    
if __name__ == '__main__':
    args, config = parse_args_and_config()
    robustness_eval(args, config)

