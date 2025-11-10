#!/bin/bash

# ===== 参数配置 =====
SAVE_DIR="exp_11.08/main/varying_K"   # 可以根据需要改名字
mkdir -p "$SAVE_DIR"                   # 如果文件夹不存在则创建

# ===== 实验循环 =====
for K in $(seq 2 10); do
    echo "🚀 Running experiment with K=${K} ..."

    OUTFILE="${SAVE_DIR}/result_K${K}.pkl"

    # 运行实验
    python main.py \
        --DR_generation_method mlp \
        --kmeans_coef 0.3 \
        --alpha_range -10 10 \
        --beta_range -10 10 \
        --tau_range -50 50 \
        --x_mean_range -30 30 \
        --N_segment_size 100 \
        --implementation_scale 2 \
        --X_noise_std_scale 0.2 \
        --disturb_covariate_noise 3 \
        --Y_noise_std_scale 0.25 \
        --K "$K" \
        --d 8 \
        --partial_x 1 \
        --N_sims 200 \
        --algorithms dast mst kmeans-standard gmm-standard clr-standard \
        --disallowed_ball_radius 0.4 \
        --save_file "$OUTFILE"

    echo "✅ Finished K=${K}. Log saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"
