#!/usr/bin/env bash
# watch_sota_v4.sh — 监控 v4 训练进度，每 10 分钟记录关键指标
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
WATCHLOG="$ROOT/logs/sota_v4_watchdog.log"

log(){ echo "[watch $(date +%H:%M:%S)] $*" | tee -a "$WATCHLOG"; }

while :; do
  alive=0
  for pid in $(cat "$ROOT/logs/sota_v4_R1.pids" 2>/dev/null); do
    if kill -0 "$pid" 2>/dev/null; then alive=$((alive+1)); fi
  done
  
  if [[ "$alive" -eq 0 ]]; then
    log "All processes finished! Checking final state..."
    break
  fi
  
  # Get latest metrics from each shard
  for sh in 0 1 2 3; do
    logfile="$ROOT/logs/sota_v4_R1_sh${sh}.log"
    if [[ -f "$logfile" ]]; then
      last_step=$(grep "step" "$logfile" | tail -1)
      if [[ -n "$last_step" ]]; then
        # Extract step number, R, kr
        step=$(echo "$last_step" | grep -oP 'step\s+\d+' | grep -oP '\d+')
        r=$(echo "$last_step" | grep -oP 'R=\d+\.\d+')
        kr=$(echo "$last_step" | grep -oP 'kr=\d+\.\d+')
        safe=$(echo "$last_step" | grep -oP 'safe=\d+\.\d+')
        log "sh${sh}: step=$step $r $kr $safe"
      fi
    fi
  done
  
  # GPU status
  gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | paste -sd '|')
  log "GPU: $gpu_info"
  
  sleep 600  # 10 min
done

log "=== Training complete ==="
for sh in 0 1 2 3; do
  logfile="$ROOT/logs/sota_v4_R1_sh${sh}.log"
  last=$(grep "step" "$logfile" | tail -1)
  log "sh${sh} final: $last"
  # Check for DONE message
  grep "DONE" "$logfile" | tail -1
done
