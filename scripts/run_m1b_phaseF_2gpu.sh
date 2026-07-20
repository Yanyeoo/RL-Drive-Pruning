#!/usr/bin/env bash
# ============================================================================
# run_m1b_phaseF_2gpu.sh — Phase F full navtest dispatcher (2 GPU, job-pool)
# ----------------------------------------------------------------------------
# Purpose: drive the full Phase F sweep over 4 variants × 4 shards using TWO
# GPUs, with simple per-GPU job queues. Each (variant, shard) is one job.
#
# Skips work that already has an aggregate.json under results/raw/. Uses the
# existing scripts/run_m1b_freelunch_sweep.sh as the inner per-job runner via
# env-var overrides (VARIANTS, SCENE_FILTER, GPU, TAG_PREFIX, TIMEOUT).
#
# Knobs (env):
#   SHARDS              space-separated shard ids to run. Default "0 1 2 3".
#   VARIANTS            space-separated variants. Default "V0 V1 V2 V3".
#   GPUS                space-separated GPU ids. Default "0 1".
#   TIMEOUT             per-job seconds (default 8100 = 2.25 h, ~25% over
#                       observed 108min). Hard kill at +30s.
#   TAG_PREFIX          experiment_name prefix (default m1b_phaseF_full).
#   SKIP_DONE           if 1 (default), skip any (variant,shard) where an
#                       aggregate.json with non-null pdms already exists.
#   DRY_RUN             if 1, only prints job plan.
#
# Output: each inner job creates results/raw/<TAG_PREFIX>_<V>_s<S>_<TS>/ with
#   merged.csv, aggregate.json, manifest.json, shard0.log
#
# Logs: logs/m1b_phaseF_2gpu_<TS>.log (this dispatcher's own log)
# ============================================================================
set -uo pipefail

PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
cd "${PROJECT_ROOT}"

SHARDS="${SHARDS:-0 1 2 3}"
VARIANTS="${VARIANTS:-V0 V1 V2 V3}"
GPUS="${GPUS:-0 1}"
TIMEOUT="${TIMEOUT:-8100}"
TAG_PREFIX="${TAG_PREFIX:-m1b_phaseF_full}"
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"

TS_LAUNCH=$(date +%Y%m%d_%H%M%S)
DISPATCH_LOG="logs/m1b_phaseF_2gpu_${TS_LAUNCH}.log"
mkdir -p logs results/raw

exec > >(tee -a "${DISPATCH_LOG}") 2>&1

echo "[phaseF_2gpu] ============================================================"
echo "[phaseF_2gpu] Phase F full navtest dispatcher  ts=${TS_LAUNCH}"
echo "[phaseF_2gpu] SHARDS    = ${SHARDS}"
echo "[phaseF_2gpu] VARIANTS  = ${VARIANTS}"
echo "[phaseF_2gpu] GPUS      = ${GPUS}"
echo "[phaseF_2gpu] TIMEOUT   = ${TIMEOUT}s per job"
echo "[phaseF_2gpu] SKIP_DONE = ${SKIP_DONE}"
echo "[phaseF_2gpu] log       = ${DISPATCH_LOG}"
echo "[phaseF_2gpu] git HEAD  = $(git rev-parse --short HEAD)"
echo "[phaseF_2gpu] ============================================================"

# ---- helper: detect already-done (variant, shard) ----
# A "done" job needs: aggregate.json with non-null pdms AND manifest.json with
# variant==V AND scene_filter matching the right shard yaml.
#
# IMPORTANT: scripts/run_m1b_freelunch_sweep.sh hard-codes the output dir name
# to "M1b_freelunch_<V>_<TS>" and IGNORES TAG_PREFIX (TAG_PREFIX only flows
# into the hydra experiment_name). So we cannot identify shards by dir name —
# every (V, *) lands under M1b_freelunch_<V>_* and we must read manifest.json
# to disambiguate by scene_filter.
is_done() {
  local V="$1" S="$2"
  local want_sf="navtest_local_filtered_shard${S}_20260616_154858"
  shopt -s nullglob
  for d in results/raw/M1b_freelunch_${V}_* \
           results/raw/M1b_phaseF_full_s${S}_${V}_* \
           results/raw/M1b_phaseF_s${S}_${V}_* \
           results/raw/M1b_phaseF_s0_${V}_*; do
    [[ -f "${d}/aggregate.json" && -f "${d}/manifest.json" ]] || continue
    if python3 -c "
import json, sys
m = json.load(open('${d}/manifest.json'))
a = json.load(open('${d}/aggregate.json'))
ok = (
    m.get('variant') == '${V}'
    and m.get('scene_filter') == '${want_sf}'
    and m.get('rc') == 0
    and a.get('pdms') is not None
    and a.get('n_valid', 0) > 100
)
sys.exit(0 if ok else 1)
" 2>/dev/null; then
      echo "${d}/aggregate.json"
      return 0
    fi
  done
  return 1
}

# ---- shard yaml resolver ----
shard_yaml() {
  echo "navtest_local_filtered_shard$1_20260616_154858"
}

# ---- build job list ----
JOBS=()
for V in ${VARIANTS}; do
  for S in ${SHARDS}; do
    if [[ "${SKIP_DONE}" == "1" ]]; then
      if EX=$(is_done "${V}" "${S}"); then
        echo "[phaseF_2gpu] SKIP done: V=${V} S=${S}  (existing: ${EX})"
        continue
      fi
    fi
    JOBS+=("${V}:${S}")
  done
done

N_JOBS=${#JOBS[@]}
echo "[phaseF_2gpu] total queued jobs: ${N_JOBS}"
if [[ ${N_JOBS} -eq 0 ]]; then
  echo "[phaseF_2gpu] nothing to do."
  exit 0
fi

# ---- distribute jobs round-robin across GPUs ----
GPU_ARR=(${GPUS})
N_GPU=${#GPU_ARR[@]}
declare -A QUEUE
for g in "${GPU_ARR[@]}"; do QUEUE[$g]=""; done

i=0
for J in "${JOBS[@]}"; do
  g=${GPU_ARR[$(( i % N_GPU ))]}
  QUEUE[$g]+=" ${J}"
  i=$(( i + 1 ))
done

echo "[phaseF_2gpu] per-GPU queues:"
for g in "${GPU_ARR[@]}"; do
  echo "  GPU${g}: ${QUEUE[$g]}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[phaseF_2gpu] DRY_RUN=1 — not launching."
  exit 0
fi

# ---- launch one worker per GPU ----
PIDS=()
for g in "${GPU_ARR[@]}"; do
  WORKER_LOG="logs/m1b_phaseF_2gpu_gpu${g}_${TS_LAUNCH}.log"
  (
    echo "[worker gpu${g}] start  jobs:${QUEUE[$g]}"
    for J in ${QUEUE[$g]}; do
      V="${J%:*}"; S="${J#*:}"
      TS=$(date +%Y%m%d_%H%M%S)
      EXP_TAG="${TAG_PREFIX}_${V}_s${S}"
      SF=$(shard_yaml "${S}")
      echo "[worker gpu${g}] ----------------------------------------"
      echo "[worker gpu${g}] start V=${V} S=${S}  scene_filter=${SF}  ts=${TS}"
      VARIANTS="${V}" \
        SCENE_FILTER="${SF}" \
        GPU="${g}" \
        TIMEOUT="${TIMEOUT}" \
        TAG_PREFIX="M1b_phaseF_full_s${S}" \
        bash scripts/run_m1b_freelunch_sweep.sh
      echo "[worker gpu${g}] done  V=${V} S=${S}  at $(date -Iseconds)"
    done
    echo "[worker gpu${g}] all jobs done"
  ) > "${WORKER_LOG}" 2>&1 &
  WPID=$!
  PIDS+=(${WPID})
  echo "[phaseF_2gpu] GPU${g} worker pid=${WPID}  log=${WORKER_LOG}"
done

echo "[phaseF_2gpu] all workers launched, waiting..."
echo "[phaseF_2gpu] worker pids: ${PIDS[*]}"

# wait for all workers
RC_AGG=0
for p in "${PIDS[@]}"; do
  if wait "${p}"; then
    echo "[phaseF_2gpu] worker pid=${p} OK"
  else
    rc=$?
    echo "[phaseF_2gpu] worker pid=${p} FAILED rc=${rc}"
    RC_AGG=1
  fi
done

echo "[phaseF_2gpu] ============================================================"
echo "[phaseF_2gpu] all workers done at $(date -Iseconds)  rc_agg=${RC_AGG}"
echo "[phaseF_2gpu] ============================================================"
exit ${RC_AGG}
