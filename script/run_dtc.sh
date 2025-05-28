#!/bin/bash

root_dir="D:\data\3DGS\dtc\rendered_data/"
# list="ficus armadillo lego hotdog"
list="birdhouse bathroom Gargoyle Mallard airplane block"
list="airplane"
for i in $list
do
    # python train.py --eval \
    #     -s ${root_dir}${i} \
    #     -m output/dtc/${i}/gss \
    #     --lambda_normal_render_depth 0.0 \
    #     --lambda_normal_smooth 0.02 \
    #     --lambda_mask_entropy 0.1 \
    #     --save_training_vis \
    #     --densify_grad_normal_threshold 1e-8 \
    #     --lambda_depth_var 1e-2

    # python eval_nvs.py --eval \
    #     -m output/dtc/${i}/gss \
    #     -c output/dtc/${i}/gss/chkpnt30000.pth

    python train.py --eval \
        -s ${root_dir}${i} \
        -m output/dtc/${i}/render_relight \
        -c output/dtc/${i}/gss/chkpnt30000.pth \
        --save_training_vis \
        --position_lr_init 0.00000 \
        --position_lr_final 0.0 \
        --normal_lr 0.001 \
        --sh_lr 0.00025 \
        --opacity_lr 0.005 \
        --scaling_lr 0.0000 \
        --rotation_lr 0.000 \
        --iterations 50000 \
        --lambda_base_color_smooth 0.005 \
        --lambda_roughness_smooth 0.005 \
        --lambda_light_smooth 0.0 \
        --lambda_light 0.00 \
        -t render_relight --sample_num 32 \
        --save_training_vis_iteration 200 \
        --lambda_env_smooth 0.02 \
        --env_resolution 32 
    
    python eval_nvs.py --eval \
        -m "output/dtc/${i}/render_relight" \
        -c "output/dtc/${i}/render_relight/chkpnt50000.pth" \
        -t render_relight

    # python eval_nvs.py --eval \
    #     -m "D:\Results\InvGS\TensoIR - v2.1/${i}/render_relight" \
    #     -c "D:\Results\InvGS\TensoIR - v2.1/${i}/render_relight/chkpnt50000.pth" \
    #     -t render_relight

    python eval_relighting_tensoIR.py \
        -m "output/dtc/${i}/render_relight" \
        -c "output/dtc/${i}/render_relight/chkpnt50000.pth" \
        --sample_num 200 \

    # python eval_relighting_tensoIR.py \
    #     -m "D:\Results\InvGS\TensoIR - no local lights/${i}/render_relight" \
    #     -c "D:\Results\InvGS\TensoIR - no local lights/${i}/render_relight/chkpnt40000.pth" \
    #     -o "test_rli_40000" \
    #     --sample_num 256 \

    # python eval_relighting_tensoIR.py \
    #     -m "D:\Results\InvGS\TensoIR - v2.1/${i}/render_relight" \
    #     -c "D:\Results\InvGS\TensoIR - v2.1/${i}/render_relight/chkpnt50000.pth" \
    #     --sample_num 256
done