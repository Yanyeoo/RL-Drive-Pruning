#!/usr/bin/env bash
# ============================================================================
# _watchdog_landscape_20260701.sh — long-lived watchdog (automation bridge is
# unavailable, so we self-host the recurring loop).
# Every LOOP_SLEEP seconds until deadline 2026-07-02 18:00 (or STOP_DRIVER):
#   - refresh the landscape figure
#   - if the driver is dead, queue is unfinished, GPUs are idle, and we are
#     before the deadline -> relaunch the driver (SKIP_DONE makes it safe)
# Never double-launches (checks pgrep first). Never rm/mv anything.
#
# Launch: nohup bash scripts/_watchdog_landscape_20260701.sh > logs/_watchdog_landscape_20260701.log 2>&1 &
# Stop:   touch STOP_DRIVER
# ============================================================================
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
cd "${ROOT}"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
DEADLINE=$(date -d '2026-07-02 18:00' +%s)
STOP="${ROOT}/STOP_DRIVER"
LOOP_SLEEP=1800   # 30 min
QUEUE_FILE="${ROOT}/landscape_queue.txt"
SKIPF="${ROOT}/logs/_driver_skipped.txt"

log(){ echo "[watchdog $(date +%m-%d\ %H:%M:%S)] $*"; }

queue_unfinished(){
  touch "$SKIPF"
  [[ -f "$QUEUE_FILE" ]] || return 1
  local V c
  for V in $(grep -vE '^[[:space:]]*#' "$QUEUE_FILE" | tr '\n' ' '); do
    [[ -z "$V" ]] && continue
    grep -qx "$V" "$SKIPF" && continue
    c=$("${PY}" - "$V" <<'PY'
import json, glob, os, sys
V = sys.argv[1]; sh = set()
for m in glob.glob(f"results/raw/M1b_freelunch_{V}_*/manifest.json"):
    d = os.path.dirname(m); ag = os.path.join(d, "aggregate.json")
    if not os.path.exists(ag): continue
    try:
        man = json.load(open(m)); a = json.load(open(ag))
    except Exception: continue
    sf = man.get("scene_filter", "")
    if "shard" in sf and a.get("pdms") is not None: sh.add(sf.split("shard")[1][0])
print(len(sh))
PY
)
    [[ "${c:-0}" -lt 4 ]] && return 0
  done
  return 1
}

log "watchdog start. deadline=$(date -d @${DEADLINE} '+%m-%d %H:%M') loop=${LOOP_SLEEP}s"
while true; do
  [[ -f "$STOP" ]] && { log "STOP sentinel; exit."; break; }
  now=$(date +%s)
  (( now >= DEADLINE )) && { log "deadline reached; final harvest + exit."; "${PY}" scripts/plot_layer_prunability_landscape.py >/dev/null 2>&1 || true; break; }

  # always refresh figures
  "${PY}" scripts/plot_layer_prunability_landscape.py >/dev/null 2>&1 || true
  "${PY}" scripts/plot_magnitude_vs_prunability.py >/dev/null 2>&1 || true

  driver_up=$(pgrep -f '_driver_landscape_20260701.sh' >/dev/null && echo 1 || echo 0)
  gpu_busy=$(pgrep -f 'run_pdm_score' >/dev/null && echo 1 || echo 0)

  if [[ "$driver_up" == "0" ]] && [[ "$gpu_busy" == "0" ]] && queue_unfinished; then
    log "driver dead + GPUs idle + queue unfinished -> relaunching driver."
    nohup bash scripts/_driver_landscape_20260701.sh \
      > "logs/_driver_landscape_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    log "relaunched driver pid=$!"
  else
    log "status: driver_up=${driver_up} gpu_busy=${gpu_busy} queue_unfinished=$(queue_unfinished && echo yes || echo no)"
  fi
  sleep "${LOOP_SLEEP}"
done
log "watchdog DONE at $(date -Iseconds)."
