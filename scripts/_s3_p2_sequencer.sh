#!/usr/bin/env bash
# _s3_p2_sequencer.sh — midnight-reclaim adaptation (window ends 00:00).
# STOP_S3 was touched so the running p1 dispatcher drains after its current
# scorer-r0.5 jobs (CSVs preserved). This waits for that drain, then relaunches
# the main-table dispatcher with PRIORITY arm order so the cheap, essential
# full-navtest arms (r=1.0 ref, random r=0.5) complete before reclaim.
# SKIP_DONE means already-harvested scorer-r0.5 CSVs are not recomputed.
# Launch: nohup bash scripts/_s3_p2_sequencer.sh > logs/_s3_p2_seq.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
log(){ echo "[p2seq $(date +%H:%M:%S)] $*"; }

log "waiting for p1 dispatcher jobs to drain (STOP_S3 set)..."
# wait until no pdm scoring process remains (current scorer-r0.5 arms finished)
while pgrep -f run_pdm_score_cot >/dev/null; do sleep 60; done
log "p1 drained. harvested scorer-r0.5 CSVs:"
ls -1 "$ROOT/results/raw/tokenprune_S3_full/"*.csv 2>/dev/null | sed 's#.*/##' || true

rm -f "$ROOT/STOP_S3"
# priority order: r=1.0 ref + random r=0.5 (both 1-pass, cheap, essential) FIRST,
# then attn_L12 r=0.5, then scorer Pareto. scorer r=0.5 last (already done -> skipped).
export ARMS_SPEC="attn_L12 1.0;random 0.5;attn_L12 0.5;scorer 0.75;scorer 0.25;scorer 0.5"
log "relaunching dispatcher with priority ARMS_SPEC"
bash "$ROOT/scripts/run_s3_maintable_full_navtest.sh"
log "p2 dispatcher returned."
