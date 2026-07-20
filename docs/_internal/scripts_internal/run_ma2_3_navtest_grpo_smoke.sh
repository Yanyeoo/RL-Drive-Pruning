#!/usr/bin/env bash
# =============================================================================
# MA2.3 — GRPO RL smoke run on navtest (no-CoT)
#
# Validates: double-card FSDP load + SFT ckpt remap + 1-2 step rollout+reward+backward.
#
# Prerequisites (must run beforehand, full or sufficiently large dry-run):
#   - MA2.1 output : ${PROJECT_ROOT}/data/navtest_nocot/*.json
#   - MA2.2 output : ${PROJECT_ROOT}/data/navtest_metric_cache/<log>/unknown/<token>/
#   - SFT ckpt     : ${PROJECT_ROOT}/models/AutoVLA/AutoVLA_PDMS_89.ckpt
#   - Qwen base    : ${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct/
#
# Knobs (env-overridable):
#   SMOKE_TIMEOUT  — seconds to let it run before timeout-killing (default 360s)
#   CONFIG_NAME    — relative to ./config/  (default training/qwen2.5-vl-3B-navtest-grpo-nocot)
#   GPUS           — devices to pass through CUDA_VISIBLE_DEVICES (default "0,1")
#   LOG_FILE       — log path (default logs/ma2_3_smoke.log)
#
# Strategy:
#   We do NOT modify upstream run_rft.py. We use a small dataset (MA2.1 dry-run
#   residue ~225 json) and rely on Trainer's batch loop + timeout to limit work.
#   Trainer config sets limit_val_batches=0 so val never runs; train will iterate
#   until killed.
#
# Smoke pass criteria (manual inspection of log):
#   1. "Loading and remapping checkpoint from: ..." prints without exception
#   2. LoRA wrap line prints trainable param count
#   3. FSDP initializes on 2 ranks (look for "rank=0" / "rank=1")
#   4. First training_step completes (look for any loss / reward log entry)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
AUTOVLA_ROOT="${PROJECT_ROOT}/code/third_party/AutoVLA"

# ----- env -----
PY="${PY:-/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "[MA2.3] FATAL: python not found at ${PY}" >&2
  exit 2
fi

GPUS="${GPUS:-0,1}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

CONFIG_NAME="${CONFIG_NAME:-training/qwen2.5-vl-3B-navtest-grpo-nocot}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-360}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/ma2_3_smoke.log}"

# ----- preflight checks -----
echo "[MA2.3] preflight checks..."

CFG_FILE="${AUTOVLA_ROOT}/config/${CONFIG_NAME}.yaml"
if [[ ! -f "${CFG_FILE}" ]]; then
  echo "[MA2.3] FATAL: config not found: ${CFG_FILE}" >&2
  exit 3
fi

SFT_CKPT="${PROJECT_ROOT}/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
if [[ ! -f "${SFT_CKPT}" ]]; then
  echo "[MA2.3] FATAL: SFT ckpt not found: ${SFT_CKPT}" >&2
  exit 4
fi

QWEN_BASE="${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"
if [[ ! -d "${QWEN_BASE}" ]]; then
  echo "[MA2.3] FATAL: Qwen base dir not found: ${QWEN_BASE}" >&2
  exit 5
fi

MA21_JSON_DIR="${PROJECT_ROOT}/data/navtest_nocot_smoke_seed"
MA22_CACHE_DIR="${PROJECT_ROOT}/data/navtest_metric_cache"
JSON_COUNT=$(find "${MA21_JSON_DIR}" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
CACHE_COUNT=$(find "${MA22_CACHE_DIR}" -type f 2>/dev/null | wc -l)

echo "[MA2.3] MA2.1 json count   : ${JSON_COUNT}  (${MA21_JSON_DIR})"
echo "[MA2.3] MA2.2 cache count  : ${CACHE_COUNT} (${MA22_CACHE_DIR})"

if [[ "${JSON_COUNT}" -lt 4 ]]; then
  echo "[MA2.3] FATAL: too few MA2.1 jsons (<4); cannot smoke." >&2
  exit 6
fi
if [[ "${CACHE_COUNT}" -lt 100 ]]; then
  echo "[MA2.3] WARN: MA2.2 cache count low; reward lookup may miss tokens." >&2
fi

# token-intersection check: every json's token must have a metric_cache entry,
# otherwise reward_function -> PDM_Reward.rl_pdm_score(token) -> KeyError (not
# caught by the upstream try/except in score.py:55).
#
# We use the existing python env to call MetricCacheLoader (the same loader the
# trainer uses), which builds {token -> cache_path}. Then assert json tokens
# subset of that dict.
echo "[MA2.3] running token intersection check..."
INTERSECT_RC=0
"${PY}" - <<PYEOF || INTERSECT_RC=$?
import sys, json
from pathlib import Path
sys.path.insert(0, "${AUTOVLA_ROOT}")
sys.path.insert(0, "${AUTOVLA_ROOT}/navsim")
from navsim.common.dataloader import MetricCacheLoader

json_dir = Path("${MA21_JSON_DIR}")
cache_dir = Path("${MA22_CACHE_DIR}")

loader = MetricCacheLoader(cache_dir)
cache_tokens = set(loader.metric_cache_paths.keys())
print(f"[preflight] cache tokens loaded: {len(cache_tokens)}", flush=True)

json_files = sorted(json_dir.glob("*.json"))
json_tokens = {p.stem for p in json_files}
print(f"[preflight] json tokens:         {len(json_tokens)}", flush=True)

missing = json_tokens - cache_tokens
covered = json_tokens & cache_tokens
print(f"[preflight] covered: {len(covered)}, missing: {len(missing)}", flush=True)

if len(covered) < 4:
    print(f"[preflight] FATAL: covered tokens < 4; smoke cannot proceed.", flush=True)
    sys.exit(7)

if missing:
    sample_missing = list(missing)[:5]
    print(f"[preflight] WARN: {len(missing)} json tokens not in cache. examples: {sample_missing}", flush=True)
    print(f"[preflight] These will cause KeyError if visited. Recommend pruning json_dataset_path.", flush=True)

sys.exit(0)
PYEOF

if [[ "${INTERSECT_RC}" -ne 0 ]]; then
  echo "[MA2.3] FATAL: token intersection preflight failed (rc=${INTERSECT_RC})." >&2
  exit "${INTERSECT_RC}"
fi

# ----- run -----
mkdir -p "$(dirname "${LOG_FILE}")"
cd "${AUTOVLA_ROOT}"
export PYTHONPATH="${AUTOVLA_ROOT}:${AUTOVLA_ROOT}/navsim:${PYTHONPATH:-}"

echo "[MA2.3] launching smoke: timeout=${SMOKE_TIMEOUT}s gpus=${GPUS} cfg=${CONFIG_NAME}"
echo "[MA2.3] log -> ${LOG_FILE}"

# timeout: SIGTERM after SMOKE_TIMEOUT, then SIGKILL 30s later if it ignores
timeout --signal=TERM --kill-after=30s "${SMOKE_TIMEOUT}s" \
  "${PY}" tools/run_rft.py --config "${CONFIG_NAME}" "$@" \
  > "${LOG_FILE}" 2>&1 || EXIT=$?

EXIT="${EXIT:-0}"
echo "[MA2.3] exit code = ${EXIT}  (124 = timed out as planned)"
echo "[MA2.3] log tail:"
tail -40 "${LOG_FILE}"
