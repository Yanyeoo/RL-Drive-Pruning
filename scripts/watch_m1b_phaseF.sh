#!/usr/bin/env bash
# watch_m1b_phaseF.sh — passive monitor for the running Phase F sweep.
#
# Read-only. Polls every 10 min, writes one snapshot per poll to
# logs/m1b_phaseF_watch.log. Stops when:
#   - sweep dispatcher process is gone AND all 4 manifests exist (DONE)
#   - sweep dispatcher process is gone but < 4 manifests exist (FAIL/PARTIAL)
#   - 20h elapsed (hard ceiling)
#
# Usage:
#   nohup setsid bash scripts/watch_m1b_phaseF.sh > /dev/null 2>&1 &
set -u

REPO=/apdcephfs/private_shayladeng/tokenrl_autoVLA
OUT=${REPO}/logs/m1b_phaseF_watch.log
RAW=${REPO}/results/raw

mkdir -p "${REPO}/logs"

START_TS=$(date +%s)
MAX_RUNTIME=$((20*3600))  # 20h
POLL=$((10*60))           # 10 min

ts() { date '+%F %H:%M:%S'; }

dump() {
  {
    echo "================================================================"
    echo "[snapshot $(ts)]"
    echo "--- procs ---"
    pgrep -af "run_m1b_freelunch|run_pdm_score_cot" || echo "NO PROCS"
    echo "--- GPU ---"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1
    echo "--- M1b_phaseF_* dirs (latest 6) ---"
    ls -td "${RAW}"/M1b_phaseF_* 2>/dev/null | head -6 | while read d; do
      mf="${d}/manifest.json"
      sh="${d}/shard0.log"
      if [[ -f "${mf}" ]]; then
        pd=$(python3 -c "import json; print(json.load(open('${mf}')).get('pdms'))" 2>/dev/null)
        rc=$(python3 -c "import json; print(json.load(open('${mf}')).get('rc'))" 2>/dev/null)
        nv=$(python3 -c "import json; print(json.load(open('${mf}')).get('n_valid'))" 2>/dev/null)
        echo "  $(basename "${d}")  pdms=${pd}  rc=${rc}  n_valid=${nv}"
      elif [[ -f "${sh}" ]]; then
        nlines=$(wc -l < "${sh}")
        # estimate progress by last token line
        last=$(grep -E 'token=|Token ' "${sh}" 2>/dev/null | tail -1)
        echo "  $(basename "${d}")  IN-PROGRESS  shard_lines=${nlines}  last:${last:0:80}"
      else
        echo "  $(basename "${d}")  no manifest, no shard0.log yet"
      fi
    done
    echo "--- manifest count ---"
    ls "${RAW}"/M1b_phaseF_*/manifest.json 2>/dev/null | wc -l
    echo ""
  } >> "${OUT}"
}

echo "=== watch_m1b_phaseF started $(ts) PID=$$ ===" >> "${OUT}"

while true; do
  dump
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TS))

  # are sweep procs still alive?
  if ! pgrep -af "run_m1b_freelunch_sweep" > /dev/null; then
    if ! pgrep -af "run_pdm_score_cot" > /dev/null; then
      N_MANIFEST=$(ls "${RAW}"/M1b_phaseF_*/manifest.json 2>/dev/null | wc -l)
      if [[ ${N_MANIFEST} -ge 4 ]]; then
        echo "[$(ts)] DONE — all 4 manifests written. Stopping watch." >> "${OUT}"
        exit 0
      else
        echo "[$(ts)] PARTIAL — dispatcher gone but only ${N_MANIFEST}/4 manifests. Stopping watch." >> "${OUT}"
        exit 1
      fi
    fi
  fi

  if [[ ${ELAPSED} -gt ${MAX_RUNTIME} ]]; then
    echo "[$(ts)] TIMEOUT — ${ELAPSED}s > ${MAX_RUNTIME}s. Sweep still running. Watch exits, sweep keeps going." >> "${OUT}"
    exit 2
  fi

  sleep ${POLL}
done
