#!/bin/bash

# ===== 参数配置 =====
SAVE_DIR="exp_feb_2026/discrete/varying_d"
mkdir -p "$SAVE_DIR"

# ===== 实验循环 =====
# varying d: same range as continuous counterpart
for D in $(seq 1 1 15); do
    D_TAG=$(printf "%02d" "$D")

    echo "🚀 Running experiment with d=${D} ..."

    OUTFILE="${SAVE_DIR}/exp_d_${D_TAG}.pkl"

    python main.py \
        --outcome_type discrete \
        --p_range 0.1 0.6 \
        --disturb_covariate_noise 3 \
        --DR_generation_method lightgbm \
        --kmeans_coef 0.3 \
        --x_mean_range -30 30 \
        --N_segment_size 100 \
        --implementation_scale 5 \
        --X_noise_std_scale 0.2 \
        --K 5 \
        --d "$D" \
        --partial_x 1 \
        --action_num 2 \
        --N_sims 100 \
        --algorithms dast kmeans-standard gmm-standard clr-standard mst t_learner s_learner x_learner \
        --disallowed_ball_radius 0.3 \
        --save_file "$OUTFILE" \
        --sequence_seed 92

    echo "✅ Finished d=${D}. Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"
