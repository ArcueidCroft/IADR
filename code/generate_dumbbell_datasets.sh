#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/bootstrap_test_datasets_n40/dumbbell}"
DEVICE="${DEVICE:-cpu}"
OVERWRITE="${OVERWRITE:-0}"

GRID=256
STEPS=250
NUM_SAMPLES=40
SEED=123
SUBSTEPS=10

# One unseen in-range width and two training widths.
EPS_LIST=(0.0069 0.0078 0.0090)

mkdir -p "${SAVE_DIR}"

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
    overwrite_args+=(--overwrite)
fi

for eps in "${EPS_LIST[@]}"; do
    # 0.0069 -> 0069
    tag="${eps#0.}"
    prefix="bd_${tag}"

    echo "Generating epsilon=${eps}: ${prefix}.npy"

    "${PYTHON_BIN}" "${SCRIPT_DIR}/get_datasets_dumbbell.py" \
        --grid "${GRID}" \
        --steps "${STEPS}" \
        --num-samples "${NUM_SAMPLES}" \
        --substeps "${SUBSTEPS}" \
        --epsilon-list "${eps}" \
        --save-dir "${SAVE_DIR}" \
        --prefix "${prefix}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --ball-radius 0.12 \
        --ball-radius-jitter 0.02 \
        --half-center-distance 0.25 \
        --half-center-distance-jitter 0.04 \
        --neck-width 0.06 \
        --neck-width-jitter 0.02 \
        --center-x 0.5 \
        --center-y 0.5 \
        --position-jitter 0.04 \
        --min-boundary-margin 0.05 \
        --random-rotate \
        --vis-dir "${SAVE_DIR}/visualisations/${prefix}" \
        --vis-num-frames 5 \
        --vis-cmap RdBu_r \
        --vis-save-individual \
        "${overwrite_args[@]}"
done

echo "All dumbbell datasets are in ${SAVE_DIR}"
find "${SAVE_DIR}" -maxdepth 1 -type f -name 'bd_*.npy' -print | sort
