#!/usr/bin/env bash
# Multi-GPU evaluation driver: dispatches every (model, dataset block, eps id)
# rollout-bootstrap task across a pool of GPUs in parallel.
#
# Each task runs test_dr1_bootstrap.py once on one trained checkpoint and one
# (shape, eps) test cell, writes a per-task log into ${LOG_DIR}/per_task/, and
# the script finally concatenates the per-task logs into one master log and
# parses it into an xlsx via log_to_xlsx_bootstrap.py.

set -uo pipefail

# DISCO_ROOT must point at the upstream DISCO repository if you intend to
# evaluate the FullDiscoA_Run baseline (commented out by default in the
# MODELS list below). Override at the shell or before sourcing.
export DISCO_ROOT="${DISCO_ROOT:-./disco_repo}"
export PYTHONPATH="${DISCO_ROOT}:${PYTHONPATH:-}"

# =========================
# GPU pool: edit to match the host
# =========================
GPUS=(0 1 2)
N_GPUS=${#GPUS[@]}

# =========================
# Bootstrap std/CI evaluation for IADR / per-eps DR / ablations / baselines
# =========================
MODEL_ROOT="logs/"
VIS_ROOT="results/dr1_vis/"
LOG_DIR="results_logs/dr1_bootstrap_total_best"
XLSX_NAME="results_dr1_bootstrap_total_best"
PER_TASK_LOG_DIR="${LOG_DIR}/per_task"
BATCH_SIZE=4
DEVICE=cuda

BOOT_B=2000
BOOT_SEED=42
BOOT_METRICS="L2_Tinf,V_Tinf,AIW_t"

VIS_DPI=300
VIS_STRIDE=1

mkdir -p "${LOG_DIR}" "${VIS_ROOT}" "${PER_TASK_LOG_DIR}"

# =========================
# Dataset blocks: dir|prefix|ids
# =========================
declare -a DATASET_BLOCKS=(
  "bootstrap_test_datasets_n40/circle|circle|0065 0069 0072 0078 0081 008425 0090 0093"
  "bootstrap_test_datasets_n40/five_stars|fs|0069 0078 0093"
  "bootstrap_test_datasets_n40/spiral_disk|sd|0069 0078 0093"
)

# =========================
# Models to evaluate.
# Each entry is "<checkpoint_subdir>|<test_script>|<allowed_shape_prefixes>".
# Comment out the lines you do not need; un-comment FullDiscoA_Run only after
# DISCO_ROOT is set and the upstream DISCO repo is on PYTHONPATH.
# =========================
declare -a MODELS=(
    "IADR_DiffOnly/IADR_DiffOnly|test_dr1_bootstrap.py|ALL"
    "IADR_ReactOnly/IADR_ReactOnly|test_dr1_bootstrap.py|ALL"
    "DR1/BlindDR_5eps_s0|test_dr1_bootstrap.py|ALL"
    "FNO_Epsilon/FNO_concat|test_dr1_bootstrap.py|ALL"
    "DR1/DR1_0_0065_s2|test_dr1_bootstrap.py|ALL"
    "DR1/DR1_0_0069_s0|test_dr1_bootstrap.py|ALL"
    "DR1/DR1_0_0072_s0|test_dr1_bootstrap.py|ALL"
    "DR1/DR1_0_0078_s2|test_dr1_bootstrap.py|ALL"
    "DR1/DR1_0_0081_s0|test_dr1_bootstrap.py|ALL"
    "DR1/DR1_0_008425_s2|test_dr1_bootstrap.py|ALL"
    "DR1/DR1_0_009_s2|test_dr1_bootstrap.py|ALL"
    "DR1/DR1_0_0093_s2|test_dr1_bootstrap.py|ALL"
    "IADR/IADR_s1|test_dr1_bootstrap.py|ALL"
    "UNet_DISCO/DISCO_UNet|test_dr1_bootstrap.py|ALL"
    # "FullDiscoA_Run/FullDisco_A_s0|test_dr1_bootstrap.py|ALL"
)

MASTER_LOG="${LOG_DIR}/${XLSX_NAME}.log"
: > "${MASTER_LOG}"

echo "========================================" | tee -a "${MASTER_LOG}"
echo "Master log : ${MASTER_LOG}"               | tee -a "${MASTER_LOG}"
echo "Per-task   : ${PER_TASK_LOG_DIR}"          | tee -a "${MASTER_LOG}"
echo "Time start : $(date)"                      | tee -a "${MASTER_LOG}"
echo "MODEL_ROOT : ${MODEL_ROOT}"                | tee -a "${MASTER_LOG}"
echo "VIS_ROOT   : ${VIS_ROOT}"                  | tee -a "${MASTER_LOG}"
echo "GPU pool   : ${GPUS[*]}  (N=${N_GPUS})"    | tee -a "${MASTER_LOG}"
echo "BOOT       : B=${BOOT_B} seed=${BOOT_SEED} metrics=${BOOT_METRICS}" | tee -a "${MASTER_LOG}"
echo "VIS        : dpi=${VIS_DPI} stride=${VIS_STRIDE}" | tee -a "${MASTER_LOG}"
echo "========================================" | tee -a "${MASTER_LOG}"

prefix_allowed() {
  local allow="$1"; local p="$2"
  if [[ "$allow" == "ALL" ]]; then return 0; fi
  for x in $allow; do [[ "$x" == "$p" ]] && return 0; done
  return 1
}

# =========================
# Expand models x dataset_blocks x eps_ids into a flat task list
# =========================
TASKS=()
for m in "${MODELS[@]}"; do
  IFS="|" read -r MODEL_NAME TEST_SCRIPT MODEL_PREFIXES <<< "${m}"
  MODEL_SHORT="${MODEL_NAME##*/}"
  MODEL_DIR="${MODEL_ROOT}/${MODEL_NAME}"

  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "[WARN] Missing model dir, skip: ${MODEL_DIR}" | tee -a "${MASTER_LOG}"
    continue
  fi
  if [[ ! -f "${TEST_SCRIPT}" ]]; then
    echo "[WARN] Missing test script, skip: ${TEST_SCRIPT}" | tee -a "${MASTER_LOG}"
    continue
  fi

  for blk in "${DATASET_BLOCKS[@]}"; do
    IFS="|" read -r DATA_DIR PREFIX IDS_STR <<< "${blk}"
    if ! prefix_allowed "${MODEL_PREFIXES}" "${PREFIX}"; then continue; fi
    # shellcheck disable=SC2206
    ids=( ${IDS_STR} )
    for did in "${ids[@]}"; do
      TASKS+=("${MODEL_NAME}|${TEST_SCRIPT}|${MODEL_DIR}|${MODEL_SHORT}|${DATA_DIR}|${PREFIX}|${did}")
    done
  done
done

N_TASKS=${#TASKS[@]}
echo "" | tee -a "${MASTER_LOG}"
echo "[$(date)] Dispatching ${N_TASKS} tasks across ${N_GPUS} GPUs" | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"

# =========================
# Per-task runner
# =========================
run_task() {
  local task_spec="$1"
  local gpu="$2"
  IFS="|" read -r MODEL_NAME TEST_SCRIPT MODEL_DIR MODEL_SHORT DATA_DIR PREFIX DID <<< "${task_spec}"
  local TEST_DATA="${DATA_DIR}/${PREFIX}_${DID}.npy"
  local EPS_DATA="${DATA_DIR}/${PREFIX}_${DID}_eps.npy"
  local VIS_OUT="${VIS_ROOT}/${MODEL_SHORT}/${PREFIX}_${DID}"
  local TASK_LABEL="${MODEL_SHORT}__${PREFIX}_${DID}"
  local TASK_LOG="${PER_TASK_LOG_DIR}/${TASK_LABEL}.log"

  {
    echo "========================================"
    echo "[$(date)] START  GPU=${gpu}  ${TASK_LABEL}"
    echo "ModelDir : ${MODEL_DIR}"
    echo "TestData : ${TEST_DATA}"
    echo "EpsData  : ${EPS_DATA}"
    echo "VisOut   : ${VIS_OUT}"
    echo "Processing: ${MODEL_SHORT} on ${PREFIX}_${DID}"
    echo "========================================"
  } > "${TASK_LOG}"

  if [[ ! -f "${TEST_DATA}" ]]; then
    echo "[WARN] Missing test file, skip." >> "${TASK_LOG}"
    return 0
  fi
  if [[ ! -f "${EPS_DATA}" ]]; then
    echo "[ERROR] Missing eps file: ${EPS_DATA}" >> "${TASK_LOG}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" python "${TEST_SCRIPT}" \
    --checkpoint "${MODEL_DIR}" \
    --ckpt_prefer last \
    --test "${TEST_DATA}" \
    --eps_values "${EPS_DATA}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --bootstrap_B "${BOOT_B}" \
    --bootstrap_seed "${BOOT_SEED}" \
    --bootstrap_metrics "${BOOT_METRICS}" \
    --vis_outdir "${VIS_OUT}" \
    --vis_dpi "${VIS_DPI}" \
    --vis_stride "${VIS_STRIDE}" >> "${TASK_LOG}" 2>&1
  local rc=$?
  echo "[$(date)] FINISH GPU=${gpu}  ${TASK_LABEL}  rc=${rc}" >> "${TASK_LOG}"
  return $rc
}

# =========================
# GPU pool scheduler: barrier every N_GPUS launches
# =========================
declare -a TASK_LOGS=()
declare -a TASK_PIDS=()

i=0
for spec in "${TASKS[@]}"; do
  gpu_idx=$((i % N_GPUS))
  gpu=${GPUS[$gpu_idx]}
  IFS="|" read -r _MN _TS _MD MODEL_SHORT _DD PREFIX DID <<< "${spec}"
  TASK_LABEL="${MODEL_SHORT}__${PREFIX}_${DID}"
  TASK_LOG="${PER_TASK_LOG_DIR}/${TASK_LABEL}.log"

  echo "[$(date)] dispatch [${i}/${N_TASKS}] gpu=${gpu}  ${TASK_LABEL}" | tee -a "${MASTER_LOG}"
  run_task "${spec}" "${gpu}" &
  pid=$!
  TASK_PIDS+=("${pid}")
  TASK_LOGS+=("${TASK_LOG}")

  i=$((i + 1))
  if (( i % N_GPUS == 0 )); then
    for pid in "${TASK_PIDS[@]:$((i - N_GPUS)):${N_GPUS}}"; do
      wait "${pid}" || echo "[WARN] task pid=${pid} returned non-zero" | tee -a "${MASTER_LOG}"
    done
  fi
done

# Drain any straggler PIDs
for pid in "${TASK_PIDS[@]}"; do
  wait "${pid}" 2>/dev/null || true
done

echo "" | tee -a "${MASTER_LOG}"
echo "[$(date)] All ${N_TASKS} tasks done. Concatenating per-task logs into master log..." | tee -a "${MASTER_LOG}"
for tlog in "${TASK_LOGS[@]}"; do
  echo "" >> "${MASTER_LOG}"
  echo "###### $(basename "${tlog}") ######" >> "${MASTER_LOG}"
  cat "${tlog}" >> "${MASTER_LOG}"
done

echo "" | tee -a "${MASTER_LOG}"
echo "All done. Generating xlsx report from BOOT lines..." | tee -a "${MASTER_LOG}"
if [[ -f "log_to_xlsx_bootstrap.py" ]]; then
    python log_to_xlsx_bootstrap.py \
      --log "${MASTER_LOG}" \
      --out "${LOG_DIR}/${XLSX_NAME}.xlsx" 2>&1 | tee -a "${MASTER_LOG}"
fi
echo "Logs at: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "Per-task logs at: ${PER_TASK_LOG_DIR}" | tee -a "${MASTER_LOG}"
echo "Sample-0 PNGs at: ${VIS_ROOT}" | tee -a "${MASTER_LOG}"
echo "xlsx at: ${LOG_DIR}/${XLSX_NAME}.xlsx" | tee -a "${MASTER_LOG}"
