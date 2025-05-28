#!/bin/bash

root_dir="datasets/Synthetic4Relight/"
# list="hotdog air_baloons jugs chair"
# list="jugs chair"
# list="air_baloons jugs hotdog chair"
list="jugs hotdog chair"
# list="air_baloons"
# list="hotdog"
# list="chair"
# list="jugs"

for i in $list
do
    # python train.py --eval \
    #     -s ${root_dir}${i} \
    #     -m output/Syn4Relight/${i}/gss \
    #     --lambda_normal_render_depth 0.001 \
    #     --lambda_normal_smooth 0.02 \
    #     --lambda_mask_entropy 0.1 \
    #     --save_training_vis \
    #     --densify_grad_normal_threshold 1e-8 \
    #     --lambda_depth_var 1e-2

    # python eval_nvs.py --eval \
    #     -m output/Syn4Relight/${i}/3dgs \
    #     -c output/Syn4Relight/${i}/3dgs/chkpnt30000.pth

    python train.py --eval \
        -s ${root_dir}${i} \
        -m output/Syn4Relight/${i}/render_relight \
        -c output/Syn4Relight/${i}/gss/chkpnt30000.pth \
        --save_training_vis \
        --position_lr_init 0.0000 \
        --position_lr_final 0.000000 \
        --normal_lr 0.001 \
        --sh_lr 0.0 \
        --opacity_lr 0.005 \
        --scaling_lr 0.0005 \
        --rotation_lr 0.0001 \
        --iterations 50000 \
        --lambda_base_color_smooth 1.0 \
        --lambda_roughness_smooth 0.5 \
        --lambda_light_smooth 1 \
        --lambda_light 0.02 \
        -t render_relight --sample_num 64 \
        --save_training_vis_iteration 200 \
        --lambda_env_smooth 0.02 \
        # --env_resolution 32  \
        # --env_lr 0.01

    
    # python eval_nvs.py --eval \
    #     -m "output/Syn4Relight/${i}/render_relight" \
    #     -c "output/Syn4Relight/${i}/render_relight/chkpnt50000.pth" \
    #     -t render_relight
    
    python eval_relighting_syn4.py \
        -m "output/Syn4Relight/${i}/render_relight" \
        -c "output/Syn4Relight/${i}/render_relight/chkpnt50000.pth" \
        --sample_num 256 
done