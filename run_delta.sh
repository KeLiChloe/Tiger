#!/bin/bash

# ===== 参数配置 =====
SAVE_DIR="exp_feb_2026/varying_delta_range"
mkdir -p "$SAVE_DIR"

# ===== 实验循环 =====
# delta_range: (0,0) -> (-1,1), step=0.1
for STEP in $(seq 1 0.2 2); do
    STEP_FMT=$(printf "%.1f" "$STEP")
    STEP_TAG=${STEP_FMT/./p}

    DELTA_LOW=$(python -c "print(-1 * float('$STEP_FMT'))")
    DELTA_HIGH=$(python -c "print(float('$STEP_FMT'))")

    echo "🚀 Running experiment with delta_range=(${DELTA_LOW}, ${DELTA_HIGH}) ..."

    OUTFILE="${SAVE_DIR}/result_delta_${STEP_TAG}.pkl"

    python main.py \
        --disturb_covariate_noise 3 \
        --DR_generation_method mlp \
        --kmeans_coef 0.3 \
        --alpha_range -5 5 \
        --beta_range -5 5 \
        --delta_range "$DELTA_LOW" "$DELTA_HIGH" \
        --tau_range -50 50 \
        --x_mean_range -30 30 \
        --N_segment_size 100 \
        --implementation_scale 5 \
        --X_noise_std_scale 0.2 \
        --Y_noise_std_scale 0.15 \
        --K 5 \
        --d 6 \
        --partial_x 1 \
        --N_sims 100 \
        --algorithms dast kmeans-standard gmm-standard clr-standard mst t_learner s_learner x_learner \
        --disallowed_ball_radius 0.3 \
        --save_file "$OUTFILE" \
        --sequence_seed 92

    echo "✅ Finished delta_range=(${DELTA_LOW}, ${DELTA_HIGH}). Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"
