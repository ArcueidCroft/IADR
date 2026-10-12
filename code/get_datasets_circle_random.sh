#!/usr/bin/env bash
# 用 get_datasets_circle_random.py 在指定 ε 上生成"温和扰动"的圆形数据集。
# 每个样本随机圆心 (±0.03 around 0.5) + 随机半径 [0.25, 0.35]。
# 每个 ε 独立保存为一份 npy，命名如 circle_0065.npy；
# 结束后再合成一份 circle_5eps.npy，shape=(5,T,N,1,H,W)。
# 存储估算：N=100, steps=199 (T=200) -> ~5 GB/eps; steps=699 (T=700) -> ~18 GB/eps。
set -e

OUT_DIR="./bootstrap_test_datasets_wide/circle"
PREFIX_BASE="circle"
COMBINED_PREFIX="circle_5eps"
GRID=256
STEPS=699        # T=200 帧，与 fs/sd 一致节省存储; 改成 699 则 T=700 匹配原 circle 格式
N_PER_EPS=35

# eps_id = 去掉 "0." 前缀: 0.0065 -> 0065
# dt 按 paper 口径取 eps^2；Python 内部会再拆成 10 个 dt/10 子步。
declare -a EPS_SEED=(
    "0.0078 42 6.103515625e-5"
    "0.0120 43 6.103515625e-5"
    "0.0047 44 6.103515625e-5"
)

mkdir -p "${OUT_DIR}"

for entry in "${EPS_SEED[@]}"; do
    EPS=$(echo  "${entry}" | awk '{print $1}')
    SEED=$(echo "${entry}" | awk '{print $2}')
    DT=$(echo   "${entry}" | awk '{print $3}')
    PREFIX="${PREFIX_BASE}_$(echo ${EPS} | sed 's/^0\.//')"   # 0.0065 -> circle_0065

    echo "============================================================"
    echo "[$(date)] eps=${EPS}  dt=${DT}  seed=${SEED}  prefix=${PREFIX}"
    echo "============================================================"

    python get_datasets_circle_random.py \
        --epsilon-list "${EPS}" \
        --num-samples  ${N_PER_EPS} \
        --steps        ${STEPS} \
        --dt           ${DT} \
        --grid         ${GRID} \
        --seed         ${SEED} \
        --radius-min   0.25 \
        --radius-max   0.35 \
        --center-x-range 0.47 0.53 \
        --center-y-range 0.47 0.53 \
        --save-dir     "${OUT_DIR}" \
        --prefix       "${PREFIX}"
done

echo ""
echo "[$(date)] Combining per-eps files into ${OUT_DIR}/${COMBINED_PREFIX}.npy"
python - "${OUT_DIR}" "${PREFIX_BASE}" "${COMBINED_PREFIX}" "${EPS_SEED[@]}" <<'PY'
import os
import sys
import numpy as np

out_dir, prefix_base, combined_prefix = sys.argv[1:4]
entries = sys.argv[4:]
data_parts = []
eps_values = []
dt_values = []

for entry in entries:
    eps, _seed, dt = entry.split()
    tag = eps.replace("0.", "", 1)
    prefix = f"{prefix_base}_{tag}"
    data_path = os.path.join(out_dir, f"{prefix}.npy")
    eps_path = os.path.join(out_dir, f"{prefix}_eps.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    arr = np.load(data_path)
    if arr.ndim != 6 or arr.shape[0] != 1:
        raise ValueError(f"{data_path} expected shape (1,T,N,1,H,W), got {arr.shape}")
    data_parts.append(arr)
    eps_values.append(float(np.load(eps_path)[0]) if os.path.exists(eps_path) else float(eps))
    dt_values.append(float(dt))

combined = np.concatenate(data_parts, axis=0)
np.save(os.path.join(out_dir, f"{combined_prefix}.npy"), combined)
np.save(os.path.join(out_dir, f"{combined_prefix}_eps.npy"), np.asarray(eps_values, dtype=np.float32))
np.save(os.path.join(out_dir, f"{combined_prefix}_dt.npy"), np.asarray(dt_values, dtype=np.float32))
print(f"combined data shape: {combined.shape}")
print(f"combined eps: {eps_values}")
print(f"combined dt: {dt_values}")
PY

echo ""
echo "[$(date)] All done. Output: ${OUT_DIR}"
echo "Files:"
ls -lh "${OUT_DIR}"/*.npy 2>/dev/null || true
