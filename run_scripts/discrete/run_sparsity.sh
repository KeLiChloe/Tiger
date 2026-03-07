#!/bin/bash

# ===== 参数配置 =====
# Varying outcome sparsity via target_p_range.
# target_p controls P(Y=1 | x=x_mean, D=0) per segment.
# The window has fixed half-width 0.05; its center shifts from 0.05 (rare) to 0.40 (common).
#
# Examples:
#   center = 0.05  =>  target_p ~ Uniform(0.01, 0.10)  ~8%  positive rate
#   center = 0.20  =>  target_p ~ Uniform(0.15, 0.25)  ~20% positive rate
#   center = 0.40  =>  target_p ~ Uniform(0.35, 0.45)  ~40% positive rate
SAVE_DIR="exp_feb_2026/discrete/varying_sparsity"
mkdir -p "$SAVE_DIR"

HALF_W=0.05

# ===== 实验循环 =====
for CENTER in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40; do
    CENTER_TAG=${CENTER/./p}

    P_LO=$(python -c "print(round(max(0.01, $CENTER - $HALF_W), 3))")
    P_HI=$(python -c "print(round(min(0.99, $CENTER + $HALF_W), 3))")

    echo "Running experiment with center_p=${CENTER}, target_p_range=(${P_LO}, ${P_HI}) ..."

    OUTFILE="${SAVE_DIR}/result_p${CENTER_TAG}.pkl"

    python main.py \
        --outcome_type discrete \
        --target_p_range "$P_LO" "$P_HI" \
        --beta_range -0.02 0.02 \
        --tau_range -1 1 \
        --disturb_covariate_noise 3 \
        --DR_generation_method lightgbm \
        --kmeans_coef 0.3 \
        --x_mean_range -30 30 \
        --N_segment_size 100 \
        --implementation_scale 5 \
        --X_noise_std_scale 0.2 \
        --K 5 \
        --d 6 \
        --partial_x 1 \
        --action_num 2 \
        --N_sims 100 \
        --algorithms dast kmeans-standard gmm-standard clr-standard mst t_learner s_learner x_learner \
        --disallowed_ball_radius 0.3 \
        --save_file "$OUTFILE" \
        --sequence_seed 92

    echo "Finished center_p=${CENTER}. Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "All experiments completed! Results saved in $SAVE_DIR/"
