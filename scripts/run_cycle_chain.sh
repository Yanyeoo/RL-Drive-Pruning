#!/usr/bin/env bash
# run_cycle_chain.sh — 2026-08-03 周期编排：Block A 完成后自动接力 B → C
# 目的：GPU 今晚 24:00 回收，不能让卡空转。
#
# 依赖：Block A 已由 run_blockA_multiseed.sh 启动（8 GPU）。
# 本脚本轮询 Block A 的 8 个 CSV，全部就绪后依次启动 Block B、Block C。
#
# 硬性保护：
#   1) 启动任何 Block 前确认 8 卡全部空闲（<1000 MiB），避免双开。
#   2) 到 HARD_STOP 时间后不再启动新 Block（避免跑一半被回收浪费）。
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
CHAINLOG="$ROOT/logs/cycle_chain_$(date +%Y%m%d_%H%M%S).log"
BLOCK_A_OUT="$ROOT/results/raw/blockA_multiseed"
# 不晚于该时刻启动新 Block（Block 需 ~3.7h，24:00 回收）
HARD_STOP_EPOCH=$(date -d "today 20:10" +%s)

log(){ echo "[chain $(date +%H:%M:%S)] $*" | tee -a "$CHAINLOG"; }

gpus_idle(){
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)
  [[ "$busy" -eq 0 ]]
}

wait_gpus_free(){
  local tries=0
  while ! gpus_idle; do
    tries=$((tries+1))
    (( tries % 10 == 1 )) && log "waiting for GPUs to free up ... (${tries}0s)"
    sleep 10
    [[ $tries -gt 180 ]] && { log "ERROR: GPUs still busy after 30min, abort chain"; return 1; }
  done
  return 0
}

log "=== cycle chain started ==="

# ---------- 1. 等 Block A ----------
log "waiting for Block A (8 CSVs in $BLOCK_A_OUT)"
while :; do
  n=$(ls "$BLOCK_A_OUT"/MSEED_budgetrl_s*_nt0.csv 2>/dev/null | wc -l)
  alive=$(ps -eo cmd | grep -c "[r]un_blockA_multiseed.sh")
  if [[ "$n" -ge 8 ]]; then log "Block A complete ($n/8 CSVs)"; break; fi
  if [[ "$alive" -eq 0 ]]; then log "Block A process gone with only $n/8 CSVs — proceeding anyway"; break; fi
  sleep 60
done

# ---------- 2. Block B ----------
if [[ $(date +%s) -lt $HARD_STOP_EPOCH ]]; then
  wait_gpus_free || exit 1
  log ">>> launching Block B (matched-compute triplet)"
  bash "$ROOT/scripts/run_blockBC.sh" B >> "$CHAINLOG" 2>&1
  log "<<< Block B finished"
else
  log "SKIP Block B (past hard stop $(date -d @$HARD_STOP_EPOCH +%H:%M))"
fi

# ---------- 3. Block C ----------
if [[ $(date +%s) -lt $HARD_STOP_EPOCH ]]; then
  wait_gpus_free || exit 1
  log ">>> launching Block C (same-compute baselines r=0.355)"
  bash "$ROOT/scripts/run_blockBC.sh" C >> "$CHAINLOG" 2>&1
  log "<<< Block C finished"
else
  log "SKIP Block C (past hard stop $(date -d @$HARD_STOP_EPOCH +%H:%M))"
fi

log "=== cycle chain done ==="
