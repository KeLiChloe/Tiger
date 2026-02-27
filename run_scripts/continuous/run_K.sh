#!/bin/bash

# ===== 参数配置 =====
SAVE_DIR="exp_feb_2026/continuous/varying_K_set3_bigM"
mkdir -p "$SAVE_DIR"

# ===== 实验循环 =====
# varying K: 1 -> 10
for K in $(seq 2 1 15); do
    K_TAG=$(printf "%02d" "$K")

    echo "🚀 Running experiment with K=${K} ..."

    OUTFILE="${SAVE_DIR}/exp_K_${K_TAG}.pkl"

    python main.py \
        --disturb_covariate_noise 3 \
        --DR_generation_method mlp \
        --kmeans_coef 0.3 \
        --alpha_range -5 5 \
        --beta_range -5 5 \
        --delta_range -0.8 0.8 \
        --tau_range -30 30 \
        --x_mean_range -50 50 \
        --N_segment_size 100 \
        --implementation_scale 5 \
        --X_noise_std_scale 0.2 \
        --Y_noise_std_scale 0.15 \
        --K "$K" \
        --d 1 \
        --partial_x 1 \
        --N_sims 100 \
        --algorithms dast causal_forest dr_learner kmeans-standard gmm-standard clr-standard mst t_learner s_learner x_learner \
        --disallowed_ball_radius 0.4 \
        --save_file "$OUTFILE" \
        --sequence_seed 1024

    echo "✅ Finished K=${K}. Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"
