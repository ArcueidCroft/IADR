#!/usr/bin/env bash
# Training driver template. Two reference sweeps are kept here as commented
# blocks: one for the eight per-eps DR specialists and one for the IADR
# multi-seed sweep. Un-comment the block you want to run, edit
# CONFIG_YML / EPS_LIST / SEED_LIST as needed, and launch.
#
# The active line at the bottom trains a single model directly.

# ---- Sweep template 1: per-eps DR specialists ---------------------
# Trains DR1 once per (eps, seed) pair sequentially on a single GPU. Each
# checkpoint lands in its own version subdirectory under code/logs/.
# The script does not rely on CLI overrides because nnpf's stage-1 argparse
# does not accept model-specific flags such as --version. Instead it copies
# the template yml into a temp workdir, replaces the seed via sed, and
# appends the version + four data paths to produce a complete yml that
# nnpf train then reads.
#
# Prerequisites:
#   - cwd is the project root (./), train_datasets/single_eps/*.npy ready
#   - models_per_epsilon.py is on PYTHONPATH (or sits next to the launcher)
#   - CONFIG_YML is the DR1 template; it must contain at least
#       model: models_per_epsilon.py:DR1
#       bounds / N / check_val_every_n_epoch / kernel_size / lr / seed / max_epochs
#     (version / train_u / train_eps / valid_u / valid_eps are appended
#      by this script per (eps, seed) iteration)
#
# Uncomment the block below to use:
#
# set -euo pipefail
# export CUDA_VISIBLE_DEVICES=0
#
# CONFIG_YML="config/DR1_per_eps_template.yml"
# DATA_DIR="train_datasets/single_eps"
#
# EPS_LIST=(0_0069 0_0081 0_0093)
# SEED_LIST=(0 1 2)
#
# WORK_DIR="$(mktemp -d -t dr1_sweep_XXXXXX)"
# trap 'rm -rf "$WORK_DIR"' EXIT
#
# for CODE in "${EPS_LIST[@]}"; do
#   for SEED in "${SEED_LIST[@]}"; do
#     VERSION="DR1_${CODE}_s${SEED}"
#     YML="${WORK_DIR}/${VERSION}.yml"
#
#     # 1) Copy the template, replace the seed
#     sed -E "s|^seed:.*|seed: ${SEED}|" "${CONFIG_YML}" > "${YML}"
#
#     # 2) Append version + four data paths
#     cat >> "${YML}" <<EOF
# version: ${VERSION}
# train_u: ${DATA_DIR}/train_dataset_${CODE}.npy
# train_eps: ${DATA_DIR}/train_dataset_eps_${CODE}.npy
# valid_u: ${DATA_DIR}/valid_dataset_${CODE}.npy
# valid_eps: ${DATA_DIR}/valid_dataset_eps_${CODE}.npy
# EOF
#
#     echo "===== [$(date '+%F %T')] training ${VERSION} ====="
#     sed -n 's/^/  | /p' "${YML}"
#     nnpf train --config "${YML}" --force_gpu
#     echo "===== [$(date '+%F %T')] finished ${VERSION} ====="
#   done
# done
# echo "ALL DONE"

# ---- Sweep template 2: IADR multi-seed -----------------
# Trains the IADR class once per seed on a single GPU. Data paths and other
# fields are inherited unchanged from CONFIG_YML; only seed and version
# are overridden per iteration.
#
# Uncomment the block below to use:
#
# set -euo pipefail
# export CUDA_VISIBLE_DEVICES=0
#
# CONFIG_YML="config/<your_IADR_config>.yml"
# SEED_LIST=(6 7 8)
# VERSION_PREFIX="IADR"
#
# WORK_DIR="$(mktemp -d -t iadr_sweep_XXXXXX)"
# trap 'rm -rf "$WORK_DIR"' EXIT
#
# for SEED in "${SEED_LIST[@]}"; do
#   VERSION="${VERSION_PREFIX}_s${SEED}"
#   YML="${WORK_DIR}/${VERSION}.yml"
#
#   sed -E \
#     -e "s|^seed:.*|seed: ${SEED}|" \
#     -e "s|^version:.*|version: ${VERSION}|" \
#     "${CONFIG_YML}" > "${YML}"
#
#   echo "===== [$(date '+%F %T')] training ${VERSION} ====="
#   sed -n 's/^/  | /p' "${YML}"
#   nnpf train --config "${YML}" --force_gpu
#   echo "===== [$(date '+%F %T')] finished ${VERSION} ====="
# done
# echo "ALL DONE"

# ---- Active line: train a single model directly --------------------
export CUDA_VISIBLE_DEVICES=0
nnpf train --config config/IADR_ReactOnly.yml --force_gpu
