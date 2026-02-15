#!/bin/bash

# ===== 参数配置 =====
SAVE_DIR="exp_feb_2026/varying_X_noise"
mkdir -p "$SAVE_DIR"

# ===== 实验循环 =====
# X_noise_std_scale: 0.05 -> 0.50 step 0.05
for X_NOISE in $(seq 0.05 0.1 1.05); do
    X_TAG=$(printf "%03d" "$(echo "$X_NOISE * 100" | bc | cut -d'.' -f1)")

    echo "🚀 Running experiment with X_noise_std_scale=${X_NOISE} ..."

    OUTFILE="${SAVE_DIR}/exp_Xnoise_${X_TAG}.pkl"

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
        --X_noise_std_scale "$X_NOISE" \
        --Y_noise_std_scale 0.1 \
        --K 4 \
        --d 6 \
        --partial_x 1 \
        --N_sims 100 \
        --algorithms dast causal_forest dr_learner kmeans-standard gmm-standard clr-standard mst t_learner s_learner x_learner \
        --disallowed_ball_radius 0.4 \
        --save_file "$OUTFILE" \
        --sequence_seed 886

    echo "✅ Finished X_noise_std_scale=${X_NOISE}. Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"
