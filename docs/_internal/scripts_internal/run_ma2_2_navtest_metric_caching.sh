#!/usr/bin/env bash
# =============================================================================
# MA2.2 — navtest metric_cache precompute
#
# Runs navsim's run_metric_caching.py to precompute per-token
#   MetricCache(trajectory, ego_state, observation, centerline,
#               route_lane_ids, drivable_area_map)
# and dump them as lzma+pickle blobs into <CACHE_PATH>/.
#
# Why we need it:
#   MA2.3 RL rollout will read these caches to compute reward without
#   re-running the full PDM observation/drivable-area pipeline per step.
#
# Knobs (env-overridable):
#   NUM_WORKERS  — CPU workers (default 16)
#   USE_PROCESS_POOL — if "1" use processes (default), else threads
#   CACHE_PATH   — where to dump cache (default tokenrl_autoVLA/data/navtest_metric_cache)
#
# Lessons inherited from MA2.1 (journal 2026-06-15 18:50):
#   - PYTHONPATH must include AutoVLA's navsim/ subpkg
#   - NUPLAN_MAPS_ROOT points to the directory CONTAINING nuplan-maps-v1.0.json
#   - We use navtest_local_filtered scene_filter to drop 4 missing-image logs
#     (metric_cache itself does not read images, but we keep the token set
#      aligned with MA2.1/MA2.3 to avoid silent divergence)
# =============================================================================
set -euo pipefail

# ----- project roots -----
PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
AUTOVLA_ROOT="${PROJECT_ROOT}/code/third_party/AutoVLA"
NAVSIM_ROOT="${AUTOVLA_ROOT}/navsim"

# ----- data roots (shared with MA2.1) -----
export OPENSCENE_DATA_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2"
export NUPLAN_MAPS_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/maps/nuplan-maps-v1.0"

# ----- cache dir -----
CACHE_PATH="${CACHE_PATH:-${PROJECT_ROOT}/data/navtest_metric_cache}"
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

# ----- python interpreter (absolute path; same env as MA2.1) -----
PY="${PY:-/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "[MA2.2] FATAL: python not found at ${PY}" >&2
  exit 2
fi

# ----- python path -----
cd "${NAVSIM_ROOT}"
export PYTHONPATH="${NAVSIM_ROOT}:${PYTHONPATH:-}"

echo "[MA2.2] cache_path  = ${CACHE_PATH}"
echo "[MA2.2] num_workers = ${NUM_WORKERS}  use_process_pool=${USE_PROCESS_POOL}"
echo "[MA2.2] OPENSCENE_DATA_ROOT = ${OPENSCENE_DATA_ROOT}"
echo "[MA2.2] NUPLAN_MAPS_ROOT    = ${NUPLAN_MAPS_ROOT}"

"${PY}" navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  train_test_split/scene_filter=navtest_local_filtered \
  cache.cache_path="${CACHE_PATH}" \
  worker=single_machine_thread_pool \
  worker.max_workers="${NUM_WORKERS}" \
  ${WORKER_PROC_FLAG} \
  "$@"
