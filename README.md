# Attack Diffusion Models for Adversarial Purification
This is a repository for storing code for course project. The code was adapted from the official implementation of the paper [Diffusion Models for Adversarial Purification](https://github.com/NVlabs/DiffPure).

## Requirements

- 64-bit Python 3.8.
- CUDA=11.0 must be installed first.

## Data and pre-trained models
ImageNet data should be put in `./dataset/imagenet_lmdb/val/` and categorized into different classes in different subfolders.
[convert the ImageNet dataset to LMDB](https://github.com/Lyken17/Efficient-PyTorch/tree/master/tools). 

There is no need to download CIFAR-10 separately.

For the pre-trained diffusion models, you need to first download them from the following links:

- [Score SDE](https://github.com/yang-song/score_sde_pytorch) for
  CIFAR-10: (`vp/cifar10_ddpmpp_deep_continuous`: [download link](https://drive.google.com/file/d/16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif/view?usp=sharing))
- [Guided Diffusion](https://github.com/openai/guided-diffusion) for
  ImageNet: (`256x256 diffusion unconditional`: [download link](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt))


For the pre-trained classifiers, most of them do not need to be downloaded separately, except for

- `resnet-50` on CIFAR-10: [download link](https://drive.google.com/drive/folders/1SEGilIEAnx9OC1JVhmOynreCF3oog7Fi?usp=sharing)

Note that you have to put all the pretrained models in the `pretrained` directory.

## Run experiments on CIFAR-10

### AutoAttack L2

bash cifar10_L2_rand.sh
bash imagenet_L2_rand.sh

### AutoAttack Linf

bash imagenet_Linf_rand.sh

## License

Please check the [LICENSE](LICENSE) file. This work may be used non-commercially, meaning for research or evaluation
purposes only. For business inquiries, please contact
[researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com).

