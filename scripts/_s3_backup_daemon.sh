#!/usr/bin/env bash
# _s3_backup_daemon.sh — refresh backups of small artifacts every 30 min until
# 23:58 tonight (disk persists across reclaim; this guards against overwrite +
# snapshots the growing full-navtest CSVs as arms complete).
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
DEST="$ROOT/backups/W1_progress"; mkdir -p "$DEST"
log(){ echo "[bkdaemon $(date +%H:%M:%S)] $*"; }
while :; do
  hm=$(date +%H%M)
  cp -a docs/results/key_results.md docs/journal/2026-07-06.md \
        docs/journal/HANDOFF_2026-07-06_W2.md docs/plan/s3_execution_plan.md \
        docs/plan/design_decisions.md "$DEST/" 2>/dev/null
  mkdir -p "$DEST/tokenprune_S3_full" "$DEST/code"
  cp -a results/raw/tokenprune_S3_full/*.csv "$DEST/tokenprune_S3_full/" 2>/dev/null
  cp -a results/raw/tokenprune_S3/*.csv "$DEST/tokenprune_S3_full/" 2>/dev/null
  cp -a scripts/run_s3_maintable_full_navtest.sh scripts/s3_aggregate_maintable.py \
        scripts/s3_budget_policy_phaseA.py code/rldrive/scoring/token_scorer.py \
        code/rldrive/scoring/budget_policy.py "$DEST/code/" 2>/dev/null
  n=$(ls results/raw/tokenprune_S3_full/*.csv 2>/dev/null | wc -l)
  log "backup refreshed -> $DEST (full-navtest CSVs so far: $n)"
  [[ "$hm" -ge 2358 ]] && { log "past 23:58, final backup done, exit"; break; }
  sleep 1800
done
