#!/usr/bin/env bash
# ============================================================================
# watch_m1b_phaseF_2gpu.sh — passive monitor for 2-GPU Phase F dispatcher
# ----------------------------------------------------------------------------
# Read-only. Polls every 10 min, writes one snapshot per poll to
#   logs/m1b_phaseF_2gpu_watch.log
# Tracks a 4 variant × 4 shard = 16-cell matrix of (variant, shard) jobs.
#
# Stop conditions:
#   - dispatcher proc gone AND all 16 cells DONE → exit 0
#   - dispatcher proc gone AND some cells missing → exit 1 (PARTIAL)
#   - MAX_RUNTIME elapsed (default 22h) → exit 2 (watchdog timeout, sweep
#     keeps running independently)
#
# Usage:
#   nohup setsid bash scripts/watch_m1b_phaseF_2gpu.sh \
#       > logs/m1b_phaseF_2gpu_watch.boot.log 2>&1 &
#
# Knobs (env):
#   VARIANTS    default "V0 V1 V2 V3"
#   SHARDS      default "0 1 2 3"
#   POLL_SEC    default 600 (10 min)
#   MAX_HOURS   default 22
# ============================================================================
set -u

REPO=/apdcephfs/private_shayladeng/tokenrl_autoVLA
cd "${REPO}"

OUT=${REPO}/logs/m1b_phaseF_2gpu_watch.log
RAW=${REPO}/results/raw
mkdir -p "${REPO}/logs"

VARIANTS="${VARIANTS:-V0 V1 V2 V3}"
SHARDS="${SHARDS:-0 1 2 3}"
POLL=${POLL_SEC:-600}
MAX_HOURS=${MAX_HOURS:-22}
MAX_RUNTIME=$((MAX_HOURS * 3600))

START_TS=$(date +%s)

ts() { date '+%F %H:%M:%S'; }

# ---- build candidate dir patterns for (V, S) ----
# scripts/run_m1b_freelunch_sweep.sh hard-codes the output dir to
# M1b_freelunch_<V>_<TS> and IGNORES TAG_PREFIX. So every (V, *) job lands
# under M1b_freelunch_<V>_*, and we MUST read manifest.json to disambiguate
# by scene_filter — we cannot tell shards apart by dir name alone.
candidate_dirs() {
  local V="$1" S="$2"
  ls -d "${RAW}"/M1b_freelunch_${V}_* \
        "${RAW}"/M1b_phaseF_full_s${S}_${V}_* \
        "${RAW}"/M1b_phaseF_s${S}_${V}_* \
        "${RAW}"/M1b_phaseF_s0_${V}_* 2>/dev/null
}

# ---- locate a done aggregate for (V, S), matching manifest.scene_filter ----
cell_status() {
  local V="$1" S="$2"
  local want_sf="navtest_local_filtered_shard${S}_20260616_154858"
  local d
  while IFS= read -r d; do
    [[ -n "${d}" ]] || continue
    [[ -f "${d}/aggregate.json" && -f "${d}/manifest.json" ]] || continue
    if python3 - "$d" "$V" "$want_sf" <<'PY' 2>/dev/null
import json, sys
d, V, want_sf = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(f"{d}/manifest.json"))
a = json.load(open(f"{d}/aggregate.json"))
ok = (
    m.get("variant") == V
    and m.get("scene_filter") == want_sf
    and m.get("rc") == 0
    and a.get("pdms") is not None
    and a.get("n_valid", 0) > 100
)
sys.exit(0 if ok else 1)
PY
    then
      return 0
    fi
  done < <(candidate_dirs "${V}" "${S}")
  return 1
}

# ---- detect in-progress (V, S) by scanning live pdm_score cmdlines ----
# Returns "PROG=<n/n>" line if found, else empty.
# We parse train_test_split=... from /proc/<pid>/cmdline and locate the variant
# from agent.head_mask_verbose flag + experiment_name token.
running_cells() {
  # echo "V0:0 V0:1 ..." cells currently running, plus optional ":prog" suffix
  for pid in $(pgrep -f "run_pdm_score_cot.py" 2>/dev/null); do
    cmd=$(tr '\0' ' ' < /proc/${pid}/cmdline 2>/dev/null || true)
    [[ -n "${cmd}" ]] || continue
    # scene_filter -> shard id
    sf=$(echo "${cmd}" | grep -oE "train_test_split=[^ ]+" | head -1 | cut -d= -f2)
    [[ -n "${sf}" ]] || continue
    shard=$(echo "${sf}" | grep -oE "shard[0-9]+" | grep -oE "[0-9]+")
    [[ -n "${shard}" ]] || continue
    # variant from experiment_name (e.g. m1b_phaseF_s0_V0_20260623_...)
    expn=$(echo "${cmd}" | grep -oE "experiment_name=[^ ]+" | head -1 | cut -d= -f2)
    variant=$(echo "${expn}" | grep -oE "V[0-9]+" | head -1)
    [[ -n "${variant}" ]] || continue
    echo "${variant}:${shard}"
  done
}

# ---- given (V, S), find current progress "<n>/<total>" or "?" ----
inprogress_progress() {
  local V="$1" S="$2"
  # search candidate dirs in mtime order, pick the freshest one with shard0.log
  # but no manifest yet, AND newer than the dispatcher start
  local d prog
  while IFS= read -r d; do
    [[ -n "${d}" ]] || continue
    [[ -f "${d}/shard0.log" && ! -f "${d}/manifest.json" ]] || continue
    prog=$(grep -oE "Processing scenario [0-9]+ / [0-9]+" "${d}/shard0.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+ / [0-9]+")
    if [[ -n "${prog}" ]]; then
      echo "${prog}"
      return 0
    fi
  done < <(ls -td $(candidate_dirs "${V}" "${S}") 2>/dev/null)
  echo "starting"
}

# print one row "Vx | s0 s1 s2 s3" with status emojis (DONE / RUN / -)
matrix_dump() {
  # gather running cells once
  local running_set
  running_set=" $(running_cells | tr '\n' ' ')"
  echo "matrix (DONE(pdms) / RUN(progress) / -)"
  for V in ${VARIANTS}; do
    line="  ${V} |"
    for S in ${SHARDS}; do
      if cell_status "${V}" "${S}" > /dev/null; then
        pd=""
        local d
        while IFS= read -r d; do
          [[ -f "${d}/aggregate.json" && -f "${d}/manifest.json" ]] || continue
          # verify scene_filter matches
          local got_sf
          got_sf=$(python3 -c "import json; print(json.load(open('${d}/manifest.json')).get('scene_filter',''))" 2>/dev/null)
          if [[ "${got_sf}" == "navtest_local_filtered_shard${S}_20260616_154858" ]]; then
            pd="${d}/aggregate.json"; break
          fi
        done < <(candidate_dirs "${V}" "${S}")
        local pdms="?"
        if [[ -n "${pd}" ]]; then
          pdms=$(python3 -c "import json; v=json.load(open('${pd}')).get('pdms'); print(round(v,2) if v else 'NaN')" 2>/dev/null)
        fi
        line+=" s${S}:DONE(${pdms})"
      elif [[ "${running_set}" == *" ${V}:${S} "* ]]; then
        prog=$(inprogress_progress "${V}" "${S}")
        line+=" s${S}:RUN(${prog})"
      else
        line+=" s${S}:-"
      fi
    done
    echo "${line}"
  done
}

count_done() {
  local n=0
  for V in ${VARIANTS}; do
    for S in ${SHARDS}; do
      if cell_status "${V}" "${S}" > /dev/null; then
        n=$((n+1))
      fi
    done
  done
  echo "${n}"
}

total_cells() {
  local nv=0 ns=0
  for _ in ${VARIANTS}; do nv=$((nv+1)); done
  for _ in ${SHARDS}; do ns=$((ns+1)); done
  echo $((nv * ns))
}

dump() {
  {
    echo "================================================================"
    echo "[snapshot $(ts)]  elapsed=$(( ($(date +%s) - START_TS) / 60 ))min"
    echo "--- dispatcher / sweep procs ---"
    pgrep -af "run_m1b_phaseF_2gpu|run_m1b_freelunch_sweep" | head -10 || echo "NO DISPATCHER"
    echo "--- inner pdm_score procs ---"
    pgrep -af "run_pdm_score_cot" | head -6 || echo "(none)"
    echo "--- GPU ---"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -4
    echo "--- $(matrix_dump)"
    local nd=$(count_done)
    local nt=$(total_cells)
    echo "--- progress: ${nd} / ${nt} cells done ---"
    echo ""
  } >> "${OUT}"
}

echo "=== watch_m1b_phaseF_2gpu started $(ts) PID=$$ MAX_HOURS=${MAX_HOURS} POLL=${POLL}s ===" >> "${OUT}"

# total cell count
TOTAL=$(total_cells)

while true; do
  dump
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TS))

  N_DONE=$(count_done)

  # dispatcher still alive?
  if ! pgrep -af "run_m1b_phaseF_2gpu" > /dev/null; then
    # also no inner sweep procs left
    if ! pgrep -af "run_m1b_freelunch_sweep|run_pdm_score_cot" > /dev/null; then
      if [[ ${N_DONE} -ge ${TOTAL} ]]; then
        echo "[$(ts)] DONE — all ${TOTAL} cells complete. Stopping watch." >> "${OUT}"
        exit 0
      else
        echo "[$(ts)] PARTIAL — dispatcher gone, only ${N_DONE}/${TOTAL} cells done. Stopping watch." >> "${OUT}"
        exit 1
      fi
    fi
  fi

  if [[ ${ELAPSED} -gt ${MAX_RUNTIME} ]]; then
    echo "[$(ts)] TIMEOUT — ${ELAPSED}s > ${MAX_RUNTIME}s. Sweep keeps running. Watch exits." >> "${OUT}"
    exit 2
  fi

  sleep ${POLL}
done
