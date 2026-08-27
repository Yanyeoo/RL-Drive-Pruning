#!/usr/bin/env bash
# monitor_evals.sh — 监控eval进度，完成后自动聚合
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
LOGDIR="$ROOT/logs/defensive_20260810"

check_progress() {
  local log="$1" total="$2" name="$3"
  local cnt; cnt=$(grep -c "Processing scenario" "$log" 2>/dev/null || echo 0)
  local pct=$(( cnt * 100 / total ))
  echo "  $name: $cnt/$total ($pct%)"
}

while true; do
  echo "=== $(date +%H:%M:%S) ==="
  
  sft_cnt=$(grep -c "Processing scenario" "$LOGDIR/_DEF_SFT_r0355_sh0_retry.log" 2>/dev/null || echo 0)
  pm_cnt=$(grep -c "Processing scenario" "$LOGDIR/_DEF_prumerge_r05_sh3_retry.log" 2>/dev/null || echo 0)
  
  check_progress "$LOGDIR/_DEF_SFT_r0355_sh0_retry.log" 2949 "SFT r=0.355"
  check_progress "$LOGDIR/_DEF_prumerge_r05_sh3_retry.log" 2868 "PruMerge sh3"
  
  # Check if SFT finished
  if grep -q "Finished running evaluation\|Saved results\|saved.*csv\|aggregate" "$LOGDIR/_DEF_SFT_r0355_sh0_retry.log" 2>/dev/null; then
    echo ">>> SFT r=0.355 DONE!"
    sft_done=1
  fi
  
  # Check if PruMerge finished
  if grep -q "Finished running evaluation\|Saved results\|saved.*csv\|aggregate" "$LOGDIR/_DEF_prumerge_r05_sh3_retry.log" 2>/dev/null; then
    echo ">>> PruMerge sh3 DONE!"
    pm_done=1
  fi
  
  if [[ "${sft_done:-0}" == "1" ]] && [[ "${pm_done:-0}" == "1" ]]; then
    echo "=== BOTH DONE! ==="
    break
  fi
  
  sleep 600
done
