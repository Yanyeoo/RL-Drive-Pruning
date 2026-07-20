#!/usr/bin/env bash
# run_tokenprune_S2_2gpu.sh — S2 headroom-gate dispatcher (2x H20).
# Runs 5 arms on the 200-scene subset split, 2 GPU queues balanced by cost
# (attn_L12 prune arms are 2-pass ~2x; r=1.0 and random are 1-pass ~1x):
#   GPU0: attn_L12 r=0.75 ; attn_L12 r=0.25          (2x + 2x)
#   GPU1: attn_L12 r=1.00 ; attn_L12 r=0.50 ; random r=0.50   (1x + 2x + 1x)
# Per-scene CSVs land in results/raw/tokenprune_S2/. Oracle post-proc reads them.
# Stop: touch STOP_S2   Launch: nohup bash scripts/run_tokenprune_S2_2gpu.sh > logs/_s2_dispatch.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
SPLIT="${SPLIT:-navtest_s2sub200_shard0}"
PREFIX="${PREFIX:-S2sub200}"
STOP="${ROOT}/STOP_S2"
RUN="bash ${ROOT}/scripts/run_tokenprune_sweep.sh"
MARK="${ROOT}/logs/_s2_dispatch_marker.txt"
log(){ echo "[s2disp $(date +%H:%M:%S)] $*"; }

if pgrep -f run_pdm_score_cot >/dev/null; then
  log "ABORT: run_pdm_score_cot already running (avoid double-run)."; exit 1
fi

# arm: "selector keep_ratio exp_name".  4 attn arms first (oracle needs them);
# random last (only for selector-gain, non-critical).
gpu0_worker(){
  for arm in "attn_L12 0.75 ${PREFIX}_attnL12_r075" "attn_L12 0.25 ${PREFIX}_attnL12_r025"; do
    [[ -f "$STOP" ]] && { log "STOP_S2 -> gpu0 abort"; return; }
    set -- $arm; log "GPU0 start $3"; $RUN "$1" "$2" "$SPLIT" 0 "$3"; log "GPU0 done $3 rc=$?"
  done
}
gpu1_worker(){
  for arm in "attn_L12 1.0 ${PREFIX}_attnL12_r100" "attn_L12 0.5 ${PREFIX}_attnL12_r050" "random 0.5 ${PREFIX}_random_r050"; do
    [[ -f "$STOP" ]] && { log "STOP_S2 -> gpu1 abort"; return; }
    set -- $arm; log "GPU1 start $3"; $RUN "$1" "$2" "$SPLIT" 1 "$3"; log "GPU1 done $3 rc=$?"
  done
}

log "S2 dispatch start. split=$SPLIT prefix=$PREFIX. arms=5, 2 GPUs."
gpu0_worker & P0=$!
gpu1_worker & P1=$!
wait $P0 $P1
log "S2 dispatch: both GPU queues finished."
{
  echo "=== S2 dispatch marker $(date -Iseconds) ==="
  echo "split=$SPLIT prefix=$PREFIX"
  for f in results/raw/tokenprune_S2/${PREFIX}_*.csv; do
    [[ -s "$f" ]] && echo "  csv: $f  ($(wc -l < "$f") lines)"
  done
} > "$MARK"
log "marker -> $MARK"
