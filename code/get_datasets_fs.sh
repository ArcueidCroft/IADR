#!/usr/bin/env bash
# 用 get_datasets_fs_random.py 在指定 ε 上生成"温和扰动"的 5-arm 星形数据集。
# 区间围绕原 baseline (n_arms=5, polygon=triangle, arm_outer_radius=0.2,
# arm_shift_x=0.4, center_disk_radius=0.12, center=(0.5,0.5)) 窄幅抖动；
# n_arms / polygon_sides / 中心位置 全部固定，朝向 0~2π 随机。
# 每个 ε 一份 npy，独立 seed 保证不撞同一组初始构型。
set -e

OUT_DIR="./bootstrap_test_datasets_wide/five_stars"
GRID=256
STEPS=250
N_PER_EPS=35

declare -a EPS_SEED=(
    "0.0078 42"
    "0.0120 43"
    "0.0047 44"
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
