#!/bin/bash

# ===== 参数配置 =====
SAVE_DIR="exp_feb_2026/discrete/varying_K"
mkdir -p "$SAVE_DIR"

# ===== 实验循环 =====
# varying K: 2 -> 15
for K in $(seq 2 1 15); do
    K_TAG=$(printf "%02d" "$K")

    echo "🚀 Running experiment with K=${K} ..."

    OUTFILE="${SAVE_DIR}/exp_K_${K_TAG}.pkl"

    python main.py \
        --outcome_type discrete \
        --p_range 0.1 0.6 \
        --disturb_covariate_noise 3 \
        --DR_generation_method lightgbm \
        --kmeans_coef 0.3 \
        --x_mean_range -50 50 \
        --N_segment_size 100 \
        --implementation_scale 5 \
        --X_noise_std_scale 0.2 \
        --K "$K" \
        --d 1 \
        --partial_x 1 \
        --action_num 2 \
        --N_sims 100 \
        --algorithms dast causal_forest dr_learner kmeans-standard gmm-standard clr-standard mst t_learner s_learner x_learner \
        --disallowed_ball_radius 0.4 \
        --save_file "$OUTFILE" \
        --sequence_seed 1024

    echo "✅ Finished K=${K}. Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"
