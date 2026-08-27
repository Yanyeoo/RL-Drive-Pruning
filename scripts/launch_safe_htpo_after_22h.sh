#!/usr/bin/env bash
# Start this once before leaving. It waits until 22:00 local time and then
# waits for eight sufficiently idle GPUs before exec'ing the unattended pipeline.
set -Eeuo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
cd "$ROOT"
START_AT="${SAFE_START_AT:-2026-07-27 22:00:00}"
MAX_WAIT_SEC="${SAFE_GPU_WAIT_SEC:-10800}"
POLL_SEC="${SAFE_GPU_POLL_SEC:-60}"
LOG_DIR="$ROOT/logs/safe_htpo_launcher"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[safe-htpo-launch $(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }
now_epoch() { date +%s; }
start_epoch=$(date -d "$START_AT" +%s)

while [[ "$(now_epoch)" -lt "$start_epoch" ]]; do
  remain=$((start_epoch - $(now_epoch)))
  log "Waiting for configured start time ${START_AT} (${remain}s remaining)"
  sleep "$(( remain < POLL_SEC ? remain : POLL_SEC ))"
done

begin=$(now_epoch)
while true; do
  if command -v nvidia-smi >/dev/null 2>&1; then
    # A device with <1 GiB used memory and <=10% utilization is considered idle.
    free_count=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '$1+0 < 1024 && $2+0 <= 10 {n++} END{print n+0}')
    if [[ "$free_count" -ge 8 ]]; then
      log "Detected ${free_count} idle GPUs; starting unattended Safe-HTPO pipeline"
      exec bash "$ROOT/scripts/run_safe_htpo_aaai_unattended.sh"
    fi
    log "Only ${free_count}/8 GPUs idle; retrying in ${POLL_SEC}s"
  else
    log "WARN: nvidia-smi unavailable; attempting pipeline launch and relying on torchrun CUDA preflight"
    exec bash "$ROOT/scripts/run_safe_htpo_aaai_unattended.sh"
  fi

  if (( $(now_epoch) - begin > MAX_WAIT_SEC )); then
    log "FATAL: timed out waiting ${MAX_WAIT_SEC}s for eight idle GPUs"
    exit 1
  fi
  sleep "$POLL_SEC"
done
