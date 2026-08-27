#!/usr/bin/env bash
# auto_restart_batch2.sh
# 12:00 GPU回收后自动重跑 batch2
# 用法: 在12:00回收前启动，等到12:00后GPU空闲时自动启动batch2
# nohup bash scripts/auto_restart_batch2.sh > logs/auto_restart_batch2.log 2>&1 &
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"

echo "[restart $(date +%H:%M:%S)] Waiting until after 12:00 for GPU recycle..."

# 等到12:00之后
while true; do
  current_hour=$(date +%H)
  current_min=$(date +%M)
  if [[ "$current_hour" -ge 12 ]]; then
    echo "[restart $(date +%H:%M:%S)] Past 12:00, now checking GPU availability..."
    break
  fi
  echo "[restart $(date +%H:%M:%S)] Still before 12:00, waiting..."
  sleep 60
done

echo "[restart $(date +%H:%M:%S)] Waiting for GPU0/1/2 to become available..."

# 等待所有GPU空闲（回收后GPU会短暂不可用，然后重新可用）
while true; do
  all_free=true
  for i in 0 1 2; do
    mem=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$i" 2>/dev/null | awk -F',' '{print $2}' | tr -d ' MiB')
    if [[ -z "$mem" ]]; then
      echo "[restart $(date +%H:%M:%S)] GPU$i not available yet, retrying..."
      all_free=false
      break
    fi
    if [[ "$mem" -gt 1000 ]]; then
      all_free=false
    fi
  done

  if $all_free; then
    echo "[restart $(date +%H:%M:%S)] All GPU0/1/2 are free! Checking for existing SUPP CSVs..."
    break
  fi

  sleep 30
done

# 检查batch2是否已有产出（防止重复跑）
EXISTING=$(ls "$ROOT/results/raw/tokenprune_S3_full/SUPP_sft_r05_fallback_sh0.csv" 2>/dev/null)
if [[ -n "$EXISTING" ]]; then
  echo "[restart $(date +%H:%M:%S)] batch2 CSVs already exist, skipping."
  echo "[restart $(date +%H:%M:%S)] Existing CSVs:"
  ls "$ROOT/results/raw/tokenprune_S3_full/SUPP_sft_r05_fallback"* "$ROOT/results/raw/tokenprune_S3_full/SUPP_attnL12"* "$ROOT/results/raw/tokenprune_S3_full/SUPP_sft_r025"* "$ROOT/results/raw/tokenprune_S3_full/SUPP_sft_r075"* 2>/dev/null
  exit 0
fi

echo "[restart $(date +%H:%M:%S)] No existing batch2 CSVs. Launching batch2..."

BATCH2_LOG="$ROOT/logs/supplement_batch2_$(date +%Y%m%d_%H%M%S).log"
nohup bash "$ROOT/scripts/run_supplement_batch2.sh" > "$BATCH2_LOG" 2>&1 &
BATCH2_PID=$!
echo "[restart $(date +%H:%M:%S)] batch2 launched (PID=$BATCH2_PID, log=$BATCH2_LOG)"

sleep 30
if ps -p "$BATCH2_PID" > /dev/null 2>&1; then
  echo "[restart $(date +%H:%M:%S)] batch2 confirmed running."
else
  echo "[restart $(date +%H:%M:%S)] WARNING: batch2 may have failed. Check $BATCH2_LOG"
fi

echo "[restart $(date +%H:%M:%S)] Monitor: tail -f $BATCH2_LOG"
