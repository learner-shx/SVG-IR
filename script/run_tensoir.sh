#!/bin/bash

root_dir="dataset/TensoIR/"
list="hotdog armadillo ficus lego"

for i in $list
do
    python train.py --eval \
        -s ${root_dir}${i} \
        -m output/TensoIR/${i}/gss \
        --lambda_normal_render_depth 0.0 \
        --lambda_normal_smooth 0.02 \
        --lambda_mask_entropy 0.1 \
        --save_training_vis \
        --densify_grad_normal_threshold 1e-8 \
        --lambda_depth_var 1e-2

    python eval_nvs.py --eval \
        -m output/TensoIR/${i}/gss \
        -c output/TensoIR/${i}/gss/chkpnt30000.pth

    python train.py --eval \
        -s ${root_dir}${i} \
        -m output/TensoIR/${i}/render_relight \
        -c output/TensoIR/${i}/gss/chkpnt30000.pth \
        --save_training_vis \
        --position_lr_init 0.00000 \
        --position_lr_final 0.0 \
        --normal_lr 0.001 \
        --sh_lr 0.00025 \
        --opacity_lr 0.005 \
        --scaling_lr 0.0000 \
        --rotation_lr 0.000 \
        --iterations 50000 \
        --lambda_base_color_smooth 0.1 \
        --lambda_roughness_smooth 0.05 \
        --lambda_light_smooth 0.0 \
        --lambda_light 0.00 \
        -t render_relight --sample_num 64 \
        --save_training_vis_iteration 200 \
        --lambda_env_smooth 0.02 \
        --env_resolution 32 

    python eval_nvs.py --eval \
        -m "output/TensoIR/${i}/render_relight" \
        -c "output/TensoIR/${i}/render_relight/chkpnt50000.pth" \
        -t render_relight \
        --skip_train
    
    python eval_relighting_tensoIR.py \
        -m "output/TensoIR/${i}/render_relight" \
        -c "output/TensoIR/${i}/render_relight/chkpnt50000.pth" \
        --sample_num 384 \

done