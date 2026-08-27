#!/usr/bin/env bash
# v5_graceful_shutdown.sh — 17:55 记录当前状态，kill 训练进程
# v5 ckpt 每 50 step 自动保存，kill 时最近 ckpt 已落盘
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"

DEADLINE="17:55"
while [[ $(date +%H:%M) < "$DEADLINE" ]]; do
  sleep 60
done

echo "[shutdown $(date)] Recording v5 final state..."

# Log final steps
for sh in 0 1 2 3; do
  logf="$ROOT/logs/sota_v5_R1_sh${sh}.log"
  last=$(grep "step" "$logf" 2>/dev/null | tail -1)
  echo "sh${sh}: $last" >> "$ROOT/logs/sota_v5_R1_shutdown.log"
done

# Kill training processes
if [[ -f "$ROOT/logs/sota_v5_R1.pids" ]]; then
  for pid in $(cat "$ROOT/logs/sota_v5_R1.pids"); do
    kill "$pid" 2>/dev/null && echo "killed PID $pid"
  done
fi

echo "[shutdown $(date)] Done. Checkpoints saved to ckpt/s3_token_scorer_budget_rl_20260805_154009_sh*/"
