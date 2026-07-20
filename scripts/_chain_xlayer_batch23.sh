#!/usr/bin/env bash
# ============================================================================
# _chain_xlayer_batch23.sh  (2026-06-30, transient chain driver)
# ----------------------------------------------------------------------------
# Waits for the in-flight batch-1 dispatcher (single-layer L8/L16/L20 K4) to
# finish, then launches batch-2/3 in VALUE-PRIORITY order on the freed 4 GPUs
# via run_m1b_phaseF_2gpu.sh. Round-robin job assignment means each GPU runs
# the first variant's shard first, so the highest-value variant (Lcomb4K4,
# the 16-head cumulative free-lunch headline) completes first and is secured
# even if the 4-GPU window closes at 24:00.
#
# Order (value-first):
#   1. Lcomb4K4  (L8+L12+L16+L20 bot-4 = 16 heads; HEADLINE cumulative free-lunch)
#   2. Lcomb3K4  (3 new layers, 12 heads; sibling without L12)
#   3. L8K6 L16K6 L20K6  (over-prune K=6, locate per-layer cliff wall)
#
# SKIP_DONE=1 so anything already complete is skipped; safe to re-run.
# This is a throwaway helper — deleted after the run completes.
# ============================================================================
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
cd "${ROOT}"

WAIT_PID="${1:?usage: $0 <batch1_dispatch_pid>}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/_chain_xlayer_batch23_${TS}.log"
exec > >(tee -a "${LOG}") 2>&1

echo "[chain] waiting for batch-1 dispatch pid=${WAIT_PID} to finish ..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
echo "[chain] batch-1 dispatch pid=${WAIT_PID} done at $(date -Iseconds). Launching batch-2/3."

VARIANTS="Lcomb4K4 Lcomb3K4 L8K6 L16K6 L20K6" \
  SHARDS="0 1 2 3" \
  GPUS="0 1 2 3" \
  TAG_PREFIX="M1b_xlayer" \
  TIMEOUT=8100 \
  SKIP_DONE=1 \
  bash scripts/run_m1b_phaseF_2gpu.sh

echo "[chain] batch-2/3 dispatcher returned at $(date -Iseconds)."
