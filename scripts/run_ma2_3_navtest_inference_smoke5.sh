#!/usr/bin/env bash
# ============================================================================
# run_ma2_3_navtest_inference_smoke5.sh — single-GPU 5-token AutoVLA navtest
#                                          inference smoke (MA2.3 step 1)
# ----------------------------------------------------------------------------
# Purpose:
#   Before writing the dual-GPU dispatcher, prove the full inference pipeline
#   end-to-end on 5 tokens with 1 GPU:
#     hydra startup → ckpt load → agent.initialize → token loop →
#     load metric_cache[token] → load json → AutoVLA.predict → pdm_score
#   Measures: per-token latency, peak VRAM, trajectory-decode success rate.
#
# Knobs (env-overridable):
#   GPU        — single device index (default 0)
#   TIMEOUT    — seconds (default 1800; expect ~60-300s)
#   LOG_FILE   — log path (default logs/ma2_3_infer_smoke5.log)
#
# Outputs:
#   $LOG_FILE                       — full stdout/stderr
#   ${NAVSIM_EXP_ROOT}/<exp_name>/  — per-token csv + submission.pkl
# ============================================================================
set -uo pipefail

PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
AUTOVLA_ROOT="${PROJECT_ROOT}/code/third_party/AutoVLA"
NAVSIM_ROOT="${AUTOVLA_ROOT}/navsim"

# ----- conda env -----
# Use autovla env (has Qwen2.5-VL + peft + lzma + lightning + navsim deps).
# navsim env from prior-work is for evaluator/CV pipeline only; it likely
# lacks Qwen2.5-VL / peft for AutoVLA agent forward.
PY="${PY:-/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "[smoke5] FATAL: python not found at ${PY}" >&2
  exit 2
fi

# ----- navsim env vars (writes NAVSIM_DEVKIT_ROOT / OPENSCENE_DATA_ROOT / ...) -----
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/setup_navsim_env_vars.sh"

# ----- knobs -----
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

TIMEOUT="${TIMEOUT:-1800}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/ma2_3_infer_smoke5_${TS}.log}"
EXP_NAME="ma2_3_infer_smoke5_${TS}"

# ----- assets -----
CKPT="${PROJECT_ROOT}/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
QWEN_BASE="${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"
JSON_DIR="${PROJECT_ROOT}/data/navtest_nocot_smoke_seed"
METRIC_CACHE="${PROJECT_ROOT}/data/navtest_metric_cache"
TRAIN_YAML="${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR_DATA="${PROJECT_ROOT}/data/navsim_v2_local"   # not actually used; agent reads abs paths from json

for f in "${CKPT}" "${QWEN_BASE}" "${JSON_DIR}" "${METRIC_CACHE}" "${TRAIN_YAML}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[smoke5] FATAL: asset not found: ${f}" >&2
    exit 3
  fi
done

mkdir -p "$(dirname "${LOG_FILE}")"

echo "[smoke5] PROJECT_ROOT = ${PROJECT_ROOT}"
echo "[smoke5] EXP_NAME     = ${EXP_NAME}"
echo "[smoke5] GPU          = ${GPU}"
echo "[smoke5] TIMEOUT      = ${TIMEOUT}s"
echo "[smoke5] LOG          = ${LOG_FILE}"

cd "${NAVSIM_ROOT}"
export PYTHONPATH="${NAVSIM_ROOT}:${AUTOVLA_ROOT}:${PYTHONPATH:-}"

# ----- launch -----
echo "[smoke5] launching..."
timeout --signal=TERM --kill-after=30s "${TIMEOUT}s" \
    "${PY}" "${NAVSIM_ROOT}/navsim/planning/script/run_pdm_score_cot.py" \
    experiment_name="${EXP_NAME}" \
    train_test_split=navtest_smoke5 \
    metric_cache_path="${METRIC_CACHE}" \
    +json_data_path="${JSON_DIR}" \
    +agent.config_path="${TRAIN_YAML}" \
    +agent.checkpoint_path="${CKPT}" \
    +agent.sensor_data_path="${SENSOR_DATA}" \
    +agent.codebook_cache_path="${AUTOVLA_ROOT}/codebook_cache/agent_vocab.pkl" \
    +agent.lora_conf.use_lora=false \
    worker=single_machine_thread_pool \
    worker.max_workers=1 \
    > "${LOG_FILE}" 2>&1 || EXIT=$?

EXIT="${EXIT:-0}"
echo "[smoke5] exit=${EXIT}"
echo "[smoke5] log tail:"
tail -40 "${LOG_FILE}"
