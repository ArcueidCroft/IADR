#!/usr/bin/env bash
          N=100 with steps=699 (T=700) -> ~18 GB per eps.
set -e

OUT_DIR="./bootstrap_test_datasets/circle"
GRID=256
STEPS=699        # T=700 frames; set to 199 (T=200) to match the fs/sd budget and save storage.
N_PER_EPS=100

# eps_id = drop the leading "0.": 0.0069 -> 0069
declare -a EPS_SEED=(
    "0.0065 0"
    "0.0069 0"
    "0.0072 0"
    "0.0078 0"
    "0.0081 0"
    "0.008425 0"
    "0.0090 0"
    "0.0093 0"
)

mkdir -p "${OUT_DIR}"

for entry in "${EPS_SEED[@]}"; do
    EPS=$(echo  "${entry}" | awk '{print $1}')
    SEED=$(echo "${entry}" | awk '{print $2}')
    PREFIX="circle_$(echo ${EPS} | sed 's/^0\.//')"   # 0.0069 -> circle_0069

    echo "============================================================"
    echo "[$(date)] eps=${EPS}  seed=${SEED}  prefix=${PREFIX}"
    echo "============================================================"

    python get_datasets_circle_random.py \
        --epsilon-list "${EPS}" \
        --num-samples  ${N_PER_EPS} \
        --steps        ${STEPS} \
        --grid         ${GRID} \
        --seed         ${SEED} \
        --radius-min   0.25 \
        --radius-max   0.35 \
        --center-x-range 0.5 0.5 \
        --center-y-range 0.5 0.5 \
        --save-dir     "${OUT_DIR}" \
        --prefix       "${PREFIX}"
done

echo ""
echo "[$(date)] All done. Output: ${OUT_DIR}"
echo "Files:"
ls -lh "${OUT_DIR}"/*.npy 2>/dev/null || true
