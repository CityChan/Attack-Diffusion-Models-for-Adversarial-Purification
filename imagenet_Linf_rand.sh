CUDA_VISIBLE_DEVICES=0 python imagenet_L2_rand.py --exp ./exp_results --config imagenet.yml \
  -i imagenet-robust_adv-$t-eps$adv_eps-4x4-bm0-t0-end1e-5-cont-eot20 \
  --t 10 \
  --adv_eps 0.0157 \
  --adv_batch_size 4 \
  --num_sub 32 \
  --domain imagenet \
  --classifier_name imagenet-resnet50 \
  --seed 0 \
  --data_seed 42 \
  --diffusion_type sde \
  --attack_version rand \
  --eot_iter 5
  --lp_norm Linf \