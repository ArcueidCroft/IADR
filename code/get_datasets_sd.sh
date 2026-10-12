#!/usr/bin/env bash
# 用 get_datasets_sd_random.py 在指定 ε 上生成"温和扰动"螺旋圆盘数据集。
# 区间围绕原 baseline (radius=0.3, amp=0.08, r_freq=10, theta_freq=5, center=(0.5,0.5))
# 窄幅抖动；中心固定在域正中央 (0.5, 0.5)；theta_freq 固定为 5 瓣；
# 朝向 0~2π 随机。每个 ε 一份 npy，独立 seed 保证不撞同一组初始构型。
set -e

OUT_DIR="./bootstrap_test_datasets_wide/spiral_disk"
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
