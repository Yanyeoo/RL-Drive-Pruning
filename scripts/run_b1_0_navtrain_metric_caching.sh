#!/usr/bin/env bash
# =============================================================================
# B1.0 — navtrain metric_cache precompute (19,225 token subset)
#
# Mirror of run_ma2_2_navtest_metric_caching.sh but for the navtrain side.
# Pre-requisite for B1.1+ (navtrain free-lunch sweep, V0/V1/V2/V3 PDMS eval).
#
# Inputs (existing assets, no new heavy data prep):
#   - scene_filter:    common/train_test_split/scene_filter/navtrain_avail19k.yaml
#                      (auto-generated 2026-06-24 by tools/scan_navtrain_full_window.py,
#                       103,288 -> 19,225 trigger tokens with full image+log coverage)
#   - train_test_split wrapper: common/train_test_split/navtrain_avail19k.yaml
#                      (1-line wrap, data_split=trainval)
#
# Output:
#   ${CACHE_PATH}/<log_name>/<token>/metric_cache.pkl.xz
#   (lane curve + ego_state + observation + drivable_area_map per token)
#
# Knobs (env-overridable):
#   NUM_WORKERS         CPU workers (default 16; process pool)
#   USE_PROCESS_POOL    "1" => processes (default), "0" => threads
#   CACHE_PATH          dump dir (default data/navtrain_metric_cache)
#   LIMIT_TOKENS        smoke override (set to a positive int via hydra
#                       to truncate; not wired by default)
#
# Estimated wall (16 workers, navtest 6962 token took ~30 min => ~3 min/1k):
#   19,225 / 6962 * 30 min ≈ 83 min   (≈ 1.5h, matches B1.0 budget)
# =============================================================================
set -euo pipefail

# ----- project roots -----
PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
AUTOVLA_ROOT="${PROJECT_ROOT}/code/third_party/AutoVLA"
NAVSIM_ROOT="${AUTOVLA_ROOT}/navsim"

# ----- data roots (shared with MA2.1 / MA2.2) -----
export OPENSCENE_DATA_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2"
export NUPLAN_MAPS_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/maps/nuplan-maps-v1.0"

# ----- cache dir -----
CACHE_PATH="${CACHE_PATH:-${PROJECT_ROOT}/data/navtrain_metric_cache}"
export NAVSIM_EXP_ROOT="${CACHE_PATH}"
mkdir -p "${CACHE_PATH}"

# ----- worker config -----
NUM_WORKERS="${NUM_WORKERS:-16}"
USE_PROCESS_POOL="${USE_PROCESS_POOL:-1}"
if [[ "${USE_PROCESS_POOL}" == "1" ]]; then
  WORKER_PROC_FLAG="worker.use_process_pool=true"
else
  WORKER_PROC_FLAG="worker.use_process_pool=false"
fi

# ----- python interpreter -----
PY="${PY:-/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "[B1.0] FATAL: python not found at ${PY}" >&2
  exit 2
fi

# ----- python path -----
cd "${NAVSIM_ROOT}"
export PYTHONPATH="${NAVSIM_ROOT}:${PYTHONPATH:-}"

echo "[B1.0] cache_path  = ${CACHE_PATH}"
echo "[B1.0] num_workers = ${NUM_WORKERS}  use_process_pool=${USE_PROCESS_POOL}"
echo "[B1.0] OPENSCENE_DATA_ROOT = ${OPENSCENE_DATA_ROOT}"
echo "[B1.0] NUPLAN_MAPS_ROOT    = ${NUPLAN_MAPS_ROOT}"

"${PY}" navsim/planning/script/run_metric_caching.py \
  train_test_split=navtrain_avail19k \
  cache.cache_path="${CACHE_PATH}" \
  worker=single_machine_thread_pool \
  worker.max_workers="${NUM_WORKERS}" \
  ${WORKER_PROC_FLAG} \
  "$@"
