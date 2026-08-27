#!/usr/bin/env bash
# auto_chain_sota.sh — 等待 Block A 结束后自动启动 SOTA RL 训练
# 本脚本在后台运行，轮询 Block A 的 8 个 CSV，全就绪后启动 run_budget_rl_sota.sh
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
CHAINLOG="$ROOT/logs/auto_chain_sota_$(date +%Y%m%d_%H%M%S).log"
BLOCK_A_OUT="$ROOT/results/raw/blockA_multiseed"
HARD_STOP_EPOCH=$(date -d "today 20:00" +%s)

log(){ echo "[chain $(date +%H:%M:%S)] $*" | tee -a "$CHAINLOG"; }

gpus_idle(){
  local busy; busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)
  [[ "$busy" -eq 0 ]]
}

wait_gpus_free(){
  local tries=0
  while ! gpus_idle; do
    tries=$((tries+1))
    (( tries % 30 == 1 )) && log "waiting for GPUs (${tries}0s)..."
    sleep 10
    [[ $tries -gt 180 ]] && { log "ERROR: timeout waiting for GPU"; return 1; }
  done
  return 0
}

log "=== auto chain started (waiting for Block A) ==="

# Wait for Block A
while :; do
  n=$(ls "$BLOCK_A_OUT"/MSEED_budgetrl_s*_nt0.csv 2>/dev/null | wc -l)
  alive=$(ps -eo cmd | grep -c "[r]un_blockA_multiseed.sh" 2>/dev/null || echo 0)
  log "Block A: $n/8 CSVs, alive=$alive"
  if [[ "$n" -ge 8 ]]; then log "Block A complete ($n/8)"; break; fi
  if [[ "$alive" -eq 0 && "$n" -gt 0 ]]; then log "Block A process gone with $n/8 — proceeding"; break; fi
  sleep 120
done

# Check hard stop
if [[ $(date +%s) -ge $HARD_STOP_EPOCH ]]; then
  log "SKIP SOTA RL (past hard stop 20:00, GPU reclaim at 24:00)"
  exit 0
fi

# Wait GPUs free
wait_gpus_free || exit 1

# Run Block A aggregation
log "=== Aggregating Block A ==="
"$ROOT/miniconda3/envs/autovla/bin/python" - <<'EOF' 2>&1 | tee -a "$CHAINLOG"
import glob, statistics, pandas as pd
rows=[]
for f in sorted(glob.glob("/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/blockA_multiseed/MSEED_budgetrl_s*_nt0.csv")):
    d=pd.read_csv(f); d=d[d['token']!='average']
    rows.append(d['score'].mean())
print(f"Block A: n={len(rows)} mean={statistics.mean(rows):.5f} std={statistics.stdev(rows):.5f}" if len(rows)>1 else f"Block A: n={len(rows)}")
EOF

# Launch SOTA RL v3 (adaptive efficiency)
log ">>> Launching SOTA Budget RL v3 (adaptive efficiency)"
setsid nohup bash "$ROOT/scripts/run_budget_rl_sota_v3.sh" > "$ROOT/logs/budget_rl_sota_master.log" 2>&1 < /dev/null &
log "SOTA RL PID=$!"
echo "$!" > "$ROOT/logs/budget_rl_sota_master.pid"
log "=== chain done ==="
