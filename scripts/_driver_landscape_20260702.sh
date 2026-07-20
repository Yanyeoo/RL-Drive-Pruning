#!/usr/bin/env bash
# ============================================================================
# _driver_landscape_20260702.sh — unattended path-A landscape backfill driver
# (cycle 2026-07-02 -> 2026-07-03 18:00). Identical logic to the 20260701
# driver, only the DEADLINE changed. Reads landscape_queue.txt each iteration
# (live-extensible). Runs one variant at a time, 4-shard on 2x H20 (no double-
# open). Injects bot-K mask ONLY between dispatchers (never editing a running
# sweep). After each variant lands: harvest PDMS, refresh figures, cp -a backup.
#
# Stops on: STOP_DRIVER sentinel OR deadline 2026-07-03 18:00 OR queue exhausted.
# SKIP_DONE=1 => safe to restart (already-done variants auto-skip).
# Launch: nohup bash scripts/_driver_landscape_20260702.sh > logs/_driver_landscape_20260702.log 2>&1 &
# Stop:   touch STOP_DRIVER
# ============================================================================
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
QUEUE_FILE="${ROOT}/landscape_queue.txt"
DEADLINE=$(date -d '2026-07-03 18:00' +%s)
STOP="${ROOT}/STOP_DRIVER"
RESLOG="${ROOT}/logs/_driver_landscape_results.log"

log(){ echo "[driver $(date +%m-%d\ %H:%M:%S)] $*"; }

harvest(){
  "${PY}" scripts/plot_layer_prunability_landscape.py >/dev/null 2>&1 || true
  "${PY}" scripts/plot_magnitude_vs_prunability.py     >/dev/null 2>&1 || true
  { echo "### harvest $(date -Iseconds)"
    for d in $(ls -dt results/raw/M1b_freelunch_L*_* 2>/dev/null); do
      [[ -s "$d/aggregate.json" ]] && echo "  $(basename $d) pdms=$(grep -oE '\"pdms\": [0-9.]+' $d/aggregate.json|head -1|grep -oE '[0-9.]+') n=$(grep -oE '\"n_valid\": [0-9]+' $d/aggregate.json|head -1|grep -oE '[0-9]+')"
    done; } >> "${RESLOG}"
}
backup(){
  local bk="backups/driver_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$bk/aggregates"
  for d in results/raw/M1b_freelunch_L*_*; do
    [[ -s "$d/aggregate.json" ]] && cp -a "$d/aggregate.json" "$bk/aggregates/$(basename $d).aggregate.json"
  done
  cp -a docs/results/key_results.md docs/results/figures/landscape_data.json "$bk/" 2>/dev/null || true
}
variant_done(){  # 4 distinct shards (by manifest scene_filter) have aggregate?
  local V="$1"
  local c
  c=$("${PY}" - "$V" <<'PY'
import json, glob, os, sys
V = sys.argv[1]
sh = set()
for m in glob.glob(f"results/raw/M1b_freelunch_{V}_*/manifest.json"):
    d = os.path.dirname(m)
    ag = os.path.join(d, "aggregate.json")
    if not os.path.exists(ag):
        continue
    try:
        man = json.load(open(m)); a = json.load(open(ag))
    except Exception:
        continue
    sf = man.get("scene_filter", "")
    if "shard" in sf and a.get("pdms") is not None:
        sh.add(sf.split("shard")[1][0])
print(len(sh))
PY
)
  [[ "${c:-0}" -ge 4 ]]
}

log "driver start. queue_file=${QUEUE_FILE} deadline=$(date -d @${DEADLINE} '+%m-%d %H:%M')"

# 1) wait for any pre-existing dispatcher to finish (avoid double-open)
if pgrep -f 'run_m1b_phaseF_2gpu' >/dev/null; then
  log "waiting for in-flight dispatcher to finish ..."
  while pgrep -f 'run_m1b_phaseF_2gpu' >/dev/null; do
    [[ -f "$STOP" ]] && { log "STOP; abort."; exit 0; }
    sleep 60
  done
fi
log "GPUs free. harvest + backup."
harvest; backup

# 2) consume queue
while true; do
  [[ -f "$STOP" ]] && { log "STOP sentinel; stopping."; break; }
  (( $(date +%s) >= DEADLINE )) && { log "deadline reached; stopping."; break; }
  [[ -f "$QUEUE_FILE" ]] || { log "no queue file; stopping."; break; }

  SKIPF="${ROOT}/logs/_driver_skipped.txt"; touch "$SKIPF"
  # pick first variant that is neither done nor skipped (ignore comment lines)
  NEXT=""
  for V in $(grep -vE '^[[:space:]]*#' "$QUEUE_FILE" | tr '\n' ' '); do
    [[ -z "$V" ]] && continue
    grep -qx "$V" "$SKIPF" && continue
    if ! variant_done "$V"; then NEXT="$V"; break; fi
  done
  [[ -z "$NEXT" ]] && { log "queue exhausted (all done/skipped); stopping."; break; }

  # ensure mask known (safe: no sweep running here, between dispatchers)
  bash scripts/_ensure_variant_mask.sh "$NEXT" || true
  if ! grep -qE "^[[:space:]]*${NEXT}\)" scripts/run_m1b_freelunch_sweep.sh; then
    log "mask unresolved for ${NEXT}; skipping."; echo "$NEXT" >> "$SKIPF"; continue
  fi

  log ">>> launching ${NEXT} (4-shard, GPUS 0 1) ..."
  VARIANTS="${NEXT}" SHARDS="0 1 2 3" GPUS="0 1" TAG_PREFIX="M1b_freelunch" \
    TIMEOUT=8100 SKIP_DONE=1 bash scripts/run_m1b_phaseF_2gpu.sh \
    > "logs/_driver_${NEXT}_$(date +%Y%m%d_%H%M%S).log" 2>&1
  log ">>> ${NEXT} dispatcher returned. harvest + backup."
  harvest; backup
done

harvest; backup
log "driver DONE at $(date -Iseconds)."
