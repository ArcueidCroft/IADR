#!/usr/bin/env bash
set -e

OUT_DIR="./bootstrap_test_datasets/spiral_disk"
GRID=256
STEPS=250
N_PER_EPS=100

declare -a EPS_SEED=(
    "0.0078 42"
    "0.0093 43"
    "0.0069 44"
)

for entry in "${EPS_SEED[@]}"; do
    EPS=$(echo "${entry}" | awk '{print $1}')
    SEED=$(echo "${entry}" | awk '{print $2}')
    PREFIX="sd_$(echo ${EPS} | sed 's/^0\.//')"   # 0.0078 -> sd_0078

    echo "============================================================"
    echo "[$(date)] eps=${EPS} seed=${SEED} prefix=${PREFIX}"
    echo "============================================================"

    python get_datasets_sd.py \
        --epsilon-list "${EPS}" \
        --num-samples ${N_PER_EPS} \
        --steps ${STEPS} \
        --grid ${GRID} \
        --seed ${SEED} \
        --random-rotate \
        --base-radius-range 0.28 0.32 \
        --ripple-amplitude-range 0.08 0.10 \
        --ripple-r-freq-range 10 12 \
        --ripple-theta-freq-range 5 5 \
        --center-x 0.5 \
        --center-y 0.5 \
        --save-dir "${OUT_DIR}" \
        --prefix "${PREFIX}"
done

echo ""
echo "[$(date)] All done. Output: ${OUT_DIR}"
