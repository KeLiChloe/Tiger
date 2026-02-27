#!/bin/bash

# ===== 参数配置 =====
# Varying the sparsity of y:
#   Center p shifts from 0.02 (very sparse: y=1 ~2%) to 0.20 (denser: y=1 ~20%)
#   Fixed half-width = 0.08 around the center (enough gap to distinguish segments)
#   Boundary handling: when center - 0.08 < 0.01, shift the window up so p_low = 0.01
#   => p_range = [max(0.01, center-0.08), max(0.01,center-0.08)+0.16]
SAVE_DIR="exp_feb_2026/discrete/varying_sparsity"
mkdir -p "$SAVE_DIR"

HALF_W=0.08  # fixed half-width: treatment effect gap between segments stays ~0.16

# ===== 实验循环 =====
# center p: 0.02 -> 0.20, step 0.02
for CENTER in 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20; do
    CENTER_TAG=${CENTER/./p}

    # If center - HALF_W < 0.01, pin p_low at 0.01 and shift p_high accordingly
    P_LOW=$(python -c "print(round(max(0.01, $CENTER - $HALF_W), 2))")
    P_HIGH=$(python -c "print(round(max(0.01, $CENTER - $HALF_W) + 2*$HALF_W, 2))")

    echo "🚀 Running experiment with center_p=${CENTER}, p_range=(${P_LOW}, ${P_HIGH}) ..."

    OUTFILE="${SAVE_DIR}/result_p${CENTER_TAG}.pkl"

    python main.py \
        --outcome_type discrete \
        --p_range "$P_LOW" "$P_HIGH" \
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

    echo "✅ Finished center_p=${CENTER}, p_range=(${P_LOW}, ${P_HIGH}). Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"
