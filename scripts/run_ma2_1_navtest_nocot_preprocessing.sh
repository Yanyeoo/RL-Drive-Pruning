#!/bin/bash
# MA2.1 — navtest no-CoT preprocessing launcher
#
# Purpose:
#   Run AutoVLA's nocot_sample_generation.py over the NAVSIM navtest split
#   to produce per-token JSON files at data/navtest_nocot/{token}.json.
#   These JSONs feed the downstream evaluator (run_pdm_score_cot.py) by
#   providing camera_paths + ego state + gt/his trajectory + (empty) cot_output.
#
# Config:
#   code/third_party/AutoVLA/config/dataset/qwen2.5-vl-3B-navtest.yaml
#   - pretrained_model_path: /apdcephfs/.../models/Qwen2.5-VL-3B-Instruct  (processor only)
#   - dataset_name:          nuplan
#   - navsim_log_path:       .../navsim_v2/navsim_logs/test
#   - sensor_blobs_path:     .../navsim_v2/sensor_blobs/test
#   - scene_filter:          navsim/.../scene_filter/navtest.yaml  (relative inside AutoVLA repo)
#
# Notes:
#   * Must cd into AutoVLA repo root because the script does f"./config/{args.config}.yaml".
#   * No model inference (nocot); CPU + many workers is fine.
#   * Idempotent restart: pass --pre_generated_dir to skip already-processed tokens.
#
# References:
#   docs/plan/MA2_navtest_baseline_integration_map.md  (§ 4.2)
#   docs/plan/implementation_plan.md  (§ M0.3 / MA2.1)

set -euo pipefail

# --- paths (absolute) ---
WS_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
AUTOVLA_ROOT="${WS_ROOT}/code/third_party/AutoVLA"
OUTPUT_DIR="${WS_ROOT}/data/navtest_nocot"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"

CONFIG="dataset/qwen2.5-vl-3B-navtest"
NUM_WORKERS="${NUM_WORKERS:-16}"
# Optional resume: PRE_GENERATED_DIR="$OUTPUT_DIR" to skip done tokens
PRE_GENERATED_DIR="${PRE_GENERATED_DIR:-}"

# --- env ---
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

# NAVSIM v2 env (the scene_filter / SceneLoader chain typically reads these)
export NUPLAN_MAPS_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/maps/nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2"
export NAVSIM_EXP_ROOT="${WS_ROOT}/exp"
export NUPLAN_EXP_ROOT="${WS_ROOT}/exp"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${NAVSIM_EXP_ROOT}"

echo "[MA2.1] AUTOVLA_ROOT = ${AUTOVLA_ROOT}"
echo "[MA2.1] OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "[MA2.1] CONFIG       = ${CONFIG}"
echo "[MA2.1] NUM_WORKERS  = ${NUM_WORKERS}"
echo "[MA2.1] PRE_GEN_DIR  = ${PRE_GENERATED_DIR:-<none>}"

cd "${AUTOVLA_ROOT}"

# AutoVLA's tools/preprocessing/*.py does `from dataset_utils.preprocessing....`
# which is resolved from repo root. Make sure repo root is on sys.path.
export PYTHONPATH="${AUTOVLA_ROOT}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ -n "${PRE_GENERATED_DIR}" ]]; then
  EXTRA_ARGS+=(--pre_generated_dir "${PRE_GENERATED_DIR}")
fi

"${PY}" tools/preprocessing/nocot_sample_generation.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers "${NUM_WORKERS}" \
    "${EXTRA_ARGS[@]}"

echo "[MA2.1] Done. Token count:"
ls "${OUTPUT_DIR}" | wc -l
