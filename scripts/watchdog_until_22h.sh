#!/usr/bin/env bash
# Periodic status dump until 22:00 or chain completes.
# Writes append-only to logs/watchdog_22h.log so user can cat it on return.
set -u

STAGING="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain"
DL_LOG="/apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/navtrain_download_aria2.log"
CHAIN_LOG="/apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/post_dl_chain.log"
OUT="/apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/watchdog_22h.log"

# stop time: today 22:05 (give chain 5min margin)
STOP_TS=$(date -d "today 22:05" +%s)

dump() {
  {
    echo "================================================================"
    echo "[snapshot $(date '+%F %H:%M:%S')]"
    echo "--- procs ---"
    pgrep -af "aria2c|download_navtrain|post_dl_chain" || echo "NO PROCS"
    echo "--- staging ls ---"
    ls -la "$STAGING" 2>&1 | head -15
    echo "--- sentinels ---"
    ls -la "$STAGING/.chain_complete" "$STAGING/.chain_failed" "$STAGING/.download_complete" 2>&1
    echo "--- download tail (last 6 lines) ---"
    tail -6 "$DL_LOG" 2>/dev/null
    echo "--- chain tail (last 6 lines) ---"
    tail -6 "$CHAIN_LOG" 2>/dev/null
    echo "--- disk ---"
    df -h "$STAGING" | tail -1
    echo ""
  } >> "$OUT"
}

echo "=== watchdog started $(date '+%F %H:%M:%S'), will stop at 22:05 or on .chain_complete/.chain_failed ===" >> "$OUT"

while true; do
  dump
  # exit early if chain finished
  if [[ -f "$STAGING/.chain_complete" ]]; then
    echo "[watchdog] chain_complete detected, exiting" >> "$OUT"
    break
  fi
  if [[ -f "$STAGING/.chain_failed" ]]; then
    echo "[watchdog] chain_failed detected, exiting" >> "$OUT"
    break
  fi
  # exit at 22:05
  now=$(date +%s)
  if (( now >= STOP_TS )); then
    echo "[watchdog] 22:05 reached, exiting" >> "$OUT"
    break
  fi
  sleep 1800   # 30 min
done

echo "=== watchdog stopped $(date '+%F %H:%M:%S') ===" >> "$OUT"
