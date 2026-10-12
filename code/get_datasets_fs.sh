#!/usr/bin/env bash
set -e

OUT_DIR="./bootstrap_test_datasets/five_stars"
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
    PREFIX="fs_$(echo ${EPS} | sed 's/^0\.//')"   # 0.0078 -> fs_0078

    echo "============================================================"
    echo "[$(date)] eps=${EPS} seed=${SEED} prefix=${PREFIX}"
    echo "============================================================"

    python get_datasets_fs.py \
        --epsilon-list "${EPS}" \
        --num-samples ${N_PER_EPS} \
        --steps ${STEPS} \
        --grid ${GRID} \
        --seed ${SEED} \
        --random-rotate \
        --n-arms 5 \
        --arm-polygon-sides 3 \
        --arm-outer-radius-range 0.20 0.22 \
        --arm-shift-x-range 0.38 0.43 \
        --arm-shift-y 0.0 \
        --center-disk-radius-range 0.14 0.15 \
        --global-shift-x 0.5 \
        --global-shift-y 0.5 \
        --save-dir "${OUT_DIR}" \
        --prefix "${PREFIX}"
done

echo ""
echo "[$(date)] All done. Output: ${OUT_DIR}"
