#!/usr/bin/env bash
# ============================================================================
# run_m1b_phase3_step2_eval.sh — evaluate dynamic per-head mask probes on navtest
# ----------------------------------------------------------------------------
# M1.b₂ Phase 3 Step 2 (走法 1) — EVAL path.
#
# For each (probe, shard) cell, runs run_pdm_score_cot.py with the
# AutoVLAWithDynamicMaskAgent: per scene a 2-pass forward picks a per-scene L12
# head-drop set via the trained PerHeadBinaryProbe. We collect:
#   - PDMS (navtest pdm_score, merged over shards)
#   - avg_K_eff / std / min / max (from per-scene maskstats jsonl)
# so the (avg_K_eff, PDMS) point can be plotted against the Step1+Pivot1 static
# (K, PDMS) curve and judged by Gate G_p3_1' (step1_results.md).
#
# Sibling launchers:
#   scripts/run_m1b_freelunch_sweep.sh         (static head_mask baselines)
#   scripts/run_m1b_phase3_step2_lambda_sweep.sh (train the probes)
#
# Granularity is PER-SCENE (not per-token); see docs/journal/2026-06-30.md 偏离 #2.
#
# Knobs (env):
#   PROBES        space-sep probe dir names under PROBE_ROOT, OR abs paths to a
#                 probe dir / model.pt. Default: all probe_l*_s* under PROBE_ROOT.
#   PROBE_ROOT    default exp/m1b2_phase2_v0/step2_probes
#   SHARDS        space-sep shard ids. Default "0 1" (= full navtest_local_filtered).
#   GPU           device index (default 0). Single-GPU, serialized for clean tags.
#   THRESHOLD     probe keep-prob threshold (default 0.5).
#   KEEP_FLOOR    min heads to keep (safety; default 0 = no floor).
#   PROBE_HIDDEN  override probe MLP hidden dim (default: read from ckpt args).
#   TIMEOUT       per-cell timeout seconds (default 18000 = 5h).
#   TAG_PREFIX    experiment_name prefix (default m1b_p3s2_dyn).
#   RESULTS_ROOT  where to drop results (default ${PROJECT_ROOT}/results/raw).
#   SKIP_DONE     1=skip cells whose aggregate.json exists. default 1.
#   DRY_RUN       1=print commands only.
#
# Outputs per (probe, shard):
#   ${RESULTS_ROOT}/M1b_p3s2_dyn_<probe>_s<shard>_g<gpu>_<TS>/
#     - merged.csv          (token-level pdm_score)
#     - aggregate.json      (pdms + n_valid/n_total + k_eff stats + sub_metrics)
#     - manifest.json       (provenance)
#     - maskstats/maskstats_pid*.jsonl  (per-scene K_eff log from the agent)
#     - shard.log
#
# Non-strict: a failing cell does NOT halt the loop; its manifest records the
# failure and we continue.
# ============================================================================
set -uo pipefail

PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
AUTOVLA_ROOT="${PROJECT_ROOT}/code/third_party/AutoVLA"
NAVSIM_ROOT="${AUTOVLA_ROOT}/navsim"

PY="${PY:-/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python}"
[[ -x "${PY}" ]] || { echo "[p3s2_eval] FATAL: no python at ${PY}" >&2; exit 2; }

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/setup_navsim_env_vars.sh"

# Make the wrapped agent + probe + head_mask_patch importable inside hydra.
export PYTHONPATH="${PROJECT_ROOT}/code:${NAVSIM_ROOT}:${AUTOVLA_ROOT}:${PYTHONPATH:-}"

# ---- knobs ----
PROBE_ROOT="${PROBE_ROOT:-${PROJECT_ROOT}/exp/m1b2_phase2_v0/step2_probes}"
SHARDS="${SHARDS:-0 1}"
GPU="${GPU:-0}"
THRESHOLD="${THRESHOLD:-0.5}"
KEEP_FLOOR="${KEEP_FLOOR:-0}"
PROBE_HIDDEN="${PROBE_HIDDEN:-}"
TIMEOUT="${TIMEOUT:-18000}"
TAG_PREFIX="${TAG_PREFIX:-m1b_p3s2_dyn}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/raw}"
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"
SHARD_SPLIT_PREFIX="${SHARD_SPLIT_PREFIX:-navtest_local_filtered_shard}"
SHARD_SPLIT_SUFFIX="${SHARD_SPLIT_SUFFIX:-_20260616_154858}"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${RESULTS_ROOT}" "${LOG_DIR}"

# ---- probes ----
if [[ -z "${PROBES:-}" ]]; then
  PROBES="$(cd "${PROBE_ROOT}" 2>/dev/null && ls -d probe_l*_s* 2>/dev/null | tr '\n' ' ')"
fi
[[ -n "${PROBES}" ]] || { echo "[p3s2_eval] FATAL: no probes found in ${PROBE_ROOT}" >&2; exit 2; }

# Resolve a PROBES entry to an absolute model.pt path and a short tag.
resolve_probe() {  # $1 = entry -> echoes "<ckpt>|<tag>"
  local entry="$1" ckpt tag
  if [[ "${entry}" == *.pt ]]; then
    ckpt="${entry}"
    tag="$(basename "$(dirname "${ckpt}")")"
  elif [[ -d "${entry}" ]]; then
    ckpt="${entry}/model.pt"; tag="$(basename "${entry}")"
  elif [[ -d "${PROBE_ROOT}/${entry}" ]]; then
    ckpt="${PROBE_ROOT}/${entry}/model.pt"; tag="${entry}"
  else
    ckpt="${entry}"; tag="$(basename "${entry%.pt}")"
  fi
  echo "${ckpt}|${tag}"
}

# ---- assets ----
CKPT="${PROJECT_ROOT}/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
QWEN_BASE="${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"
JSON_DIR="${JSON_DIR:-${PROJECT_ROOT}/data/navtest_nocot}"
METRIC_CACHE="${METRIC_CACHE:-${PROJECT_ROOT}/data/navtest_metric_cache}"
TRAIN_YAML="${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR_DATA="${PROJECT_ROOT}/data/navsim_v2_local"

for f in "${CKPT}" "${QWEN_BASE}" "${JSON_DIR}" "${METRIC_CACHE}" "${TRAIN_YAML}"; do
  [[ -e "${f}" ]] || { echo "[p3s2_eval] FATAL: asset not found: ${f}" >&2; exit 3; }
done

echo "[p3s2_eval] ============================================================"
echo "[p3s2_eval] M1.b₂ Phase 3 Step 2 dynamic-mask eval"
echo "[p3s2_eval] PROBES     = ${PROBES}"
echo "[p3s2_eval] SHARDS     = ${SHARDS}"
echo "[p3s2_eval] GPU        = ${GPU}"
echo "[p3s2_eval] THRESHOLD  = ${THRESHOLD}  KEEP_FLOOR=${KEEP_FLOOR}  HIDDEN='${PROBE_HIDDEN}'"
echo "[p3s2_eval] SKIP_DONE  = ${SKIP_DONE}  DRY_RUN=${DRY_RUN}"
echo "[p3s2_eval] git HEAD   = $(cd ${PROJECT_ROOT} && git rev-parse --short HEAD 2>/dev/null)"
echo "[p3s2_eval] host       = $(hostname)"
echo "[p3s2_eval] ============================================================"

GIT_HEAD="$(cd ${PROJECT_ROOT} && git rev-parse HEAD 2>/dev/null || echo unknown)"
HOST="$(hostname)"

for PROBE_ENTRY in ${PROBES}; do
  IFS='|' read -r PROBE_CKPT PROBE_TAG <<< "$(resolve_probe "${PROBE_ENTRY}")"
  if [[ ! -f "${PROBE_CKPT}" ]]; then
    echo "[p3s2_eval] SKIP: probe ckpt not found: ${PROBE_CKPT} (entry='${PROBE_ENTRY}')" >&2
    continue
  fi

  for SHARD in ${SHARDS}; do
    SCENE_FILTER="${SHARD_SPLIT_PREFIX}${SHARD}${SHARD_SPLIT_SUFFIX}"
    TS=$(date +%Y%m%d_%H%M%S)
    EXP_NAME="${TAG_PREFIX}_${PROBE_TAG}_s${SHARD}_g${GPU}_${TS}"
    CELL_DIR="${RESULTS_ROOT}/M1b_p3s2_dyn_${PROBE_TAG}_s${SHARD}_g${GPU}_${TS}"
    MASK_LOG_DIR="${CELL_DIR}/maskstats"

    # SKIP_DONE: glob a previous completed cell for this (probe, shard).
    if [[ "${SKIP_DONE}" == "1" ]]; then
      PREV=$(ls -d "${RESULTS_ROOT}"/M1b_p3s2_dyn_"${PROBE_TAG}"_s"${SHARD}"_g*_* 2>/dev/null | while read -r d; do
        [[ -s "${d}/aggregate.json" ]] && echo "${d}"; done | head -1)
      if [[ -n "${PREV}" ]]; then
        echo "[p3s2_eval] SKIP done: ${PROBE_TAG} shard${SHARD} (have ${PREV})"
        continue
      fi
    fi

    mkdir -p "${CELL_DIR}" "${MASK_LOG_DIR}"
    SHARD_LOG="${CELL_DIR}/shard.log"

    echo ""
    echo "[p3s2_eval] >>> probe=${PROBE_TAG}  shard=${SHARD}  (${SCENE_FILTER})"
    echo "[p3s2_eval]     ckpt=${PROBE_CKPT}"
    echo "[p3s2_eval]     exp=${EXP_NAME}  out=${CELL_DIR}"

    TS_START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    T_START_EPOCH=$(date +%s)

    HYDRA_ARGS=(
      "experiment_name=${EXP_NAME}"
      "train_test_split=${SCENE_FILTER}"
      "metric_cache_path=${METRIC_CACHE}"
      "+json_data_path=${JSON_DIR}"
      "agent._target_=rldrive.agents.autovla_with_dynamic_mask.AutoVLAWithDynamicMaskAgent"
      "+agent.config_path=${TRAIN_YAML}"
      "+agent.checkpoint_path=${CKPT}"
      "+agent.sensor_data_path=${SENSOR_DATA}"
      "+agent.codebook_cache_path=${AUTOVLA_ROOT}/codebook_cache/agent_vocab.pkl"
      "+agent.lora_conf.use_lora=false"
      "+agent.probe_ckpt_path=${PROBE_CKPT}"
      "+agent.mask_threshold=${THRESHOLD}"
      "+agent.keep_floor=${KEEP_FLOOR}"
      "+agent.mask_log_dir=${MASK_LOG_DIR}"
      "+agent.verbose=false"
      "worker=single_machine_thread_pool"
      "worker.max_workers=1"
    )
    if [[ -n "${PROBE_HIDDEN}" ]]; then
      HYDRA_ARGS+=("+agent.probe_hidden=${PROBE_HIDDEN}")
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[p3s2_eval DRY] cd ${NAVSIM_ROOT} && CUDA_VISIBLE_DEVICES=${GPU} \\"
      echo "[p3s2_eval DRY]   ${PY} ${NAVSIM_ROOT}/navsim/planning/script/run_pdm_score_cot.py \\"
      for a in "${HYDRA_ARGS[@]}"; do echo "[p3s2_eval DRY]     ${a}"; done
      continue
    fi

    RC=0
    (
      cd "${NAVSIM_ROOT}" || exit 99
      export CUDA_VISIBLE_DEVICES="${GPU}"
      exec timeout --signal=TERM --kill-after=30s "${TIMEOUT}s" \
        "${PY}" "${NAVSIM_ROOT}/navsim/planning/script/run_pdm_score_cot.py" \
        "${HYDRA_ARGS[@]}"
    ) > "${SHARD_LOG}" 2>&1 || RC=$?

    T_END_EPOCH=$(date +%s)
    TS_END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    WALL_S=$(( T_END_EPOCH - T_START_EPOCH ))
    echo "[p3s2_eval]     rc=${RC}  wall=${WALL_S}s"

    # ---- collect outputs (merge csv + maskstats avg_K_eff) ----
    EXP_DIR_NAVSIM="${NAVSIM_EXP_ROOT}/${EXP_NAME}"
    PDMS="null"; N_VALID="null"; N_TOTAL="null"
    AVG_KEFF="null"; KEFF_STD="null"; KEFF_MIN="null"; KEFF_MAX="null"; N_SCENE="null"
    FAIL_REASON="null"

    "${PY}" - "${EXP_DIR_NAVSIM}" "${CELL_DIR}" "${MASK_LOG_DIR}" <<'PYEOF'
import sys, json
from pathlib import Path
import pandas as pd

exp_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
mask_dir = Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

agg = {"pdms": None, "n_valid": 0, "n_total": 0, "sub_metrics": {},
       "k_eff": {"avg": None, "std": None, "min": None, "max": None, "n_scene": 0}}

# --- PDMS from navsim csv ---
csvs = sorted(exp_dir.rglob("*.csv"))
if csvs:
    frames = [pd.read_csv(c) for c in csvs]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["token"] != "average"].copy()
    m = m.drop_duplicates(subset="token")
    m.to_csv(out_dir / "merged.csv", index=False)
    valid = m.loc[m["valid"]]
    agg["pdms"] = float(valid["score"].mean()) if len(valid) else None
    agg["n_valid"] = int(len(valid))
    agg["n_total"] = int(len(m))
    for col in ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                "time_to_collision_within_bound", "comfort", "driving_direction_compliance"):
        if col in m.columns:
            agg["sub_metrics"][col] = float(valid[col].mean()) if len(valid) else None

# --- avg_K_eff from per-scene maskstats jsonl ---
ks = []
for jf in sorted(mask_dir.glob("maskstats_pid*.jsonl")):
    for line in jf.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ks.append(int(json.loads(line)["k_eff"]))
        except Exception:
            pass
if ks:
    import statistics as st
    agg["k_eff"] = {
        "avg": float(sum(ks) / len(ks)),
        "std": float(st.pstdev(ks)) if len(ks) > 1 else 0.0,
        "min": int(min(ks)),
        "max": int(max(ks)),
        "n_scene": int(len(ks)),
    }

(out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
ke = agg["k_eff"]
print(f"[merge] pdms={agg['pdms']}  n_valid={agg['n_valid']}/{agg['n_total']}  "
      f"avg_K_eff={ke['avg']} (n_scene={ke['n_scene']})", flush=True)
PYEOF

    if [[ -s "${CELL_DIR}/aggregate.json" ]]; then
      read -r PDMS N_VALID N_TOTAL AVG_KEFF KEFF_STD KEFF_MIN KEFF_MAX N_SCENE < <(
        "${PY}" -c "
import json
a=json.load(open('${CELL_DIR}/aggregate.json'))
k=a['k_eff']
print(a['pdms'], a['n_valid'], a['n_total'], k['avg'], k['std'], k['min'], k['max'], k['n_scene'])
" 2>/dev/null)
      : "${PDMS:=null}" "${N_VALID:=null}" "${N_TOTAL:=null}"
      : "${AVG_KEFF:=null}" "${KEFF_STD:=null}" "${KEFF_MIN:=null}" "${KEFF_MAX:=null}" "${N_SCENE:=null}"
    else
      FAIL_REASON='"no aggregate.json — eval likely failed (see shard.log)"'
    fi
    if [[ ${RC} -ne 0 ]]; then
      FAIL_REASON='"dispatcher rc='${RC}'"'
    fi

    # ---- write manifest.json ----
    cat > "${CELL_DIR}/manifest.json" <<EOM
{
  "spec_doc": "exp/m1b2_phase2_v0/m1b2_phase3_step2_spec.md",
  "spec_version": "step2-v1",
  "granularity": "per-scene",
  "agent": "AutoVLAWithDynamicMaskAgent",
  "probe_tag": "${PROBE_TAG}",
  "probe_ckpt": "${PROBE_CKPT}",
  "mask_threshold": ${THRESHOLD},
  "keep_floor": ${KEEP_FLOOR},
  "target_layer": 12,
  "git_commit": "${GIT_HEAD}",
  "ts_start_utc": "${TS_START_UTC}",
  "ts_end_utc":   "${TS_END_UTC}",
  "wall_seconds": ${WALL_S},
  "gpu":          ${GPU},
  "host":         "${HOST}",
  "scene_filter": "${SCENE_FILTER}",
  "shard":        ${SHARD},
  "exp_name":     "${EXP_NAME}",
  "exp_dir":      "${EXP_DIR_NAVSIM}",
  "rc":           ${RC},
  "n_valid":      ${N_VALID},
  "n_total":      ${N_TOTAL},
  "pdms":         ${PDMS},
  "avg_k_eff":    ${AVG_KEFF},
  "k_eff_std":    ${KEFF_STD},
  "k_eff_min":    ${KEFF_MIN},
  "k_eff_max":    ${KEFF_MAX},
  "n_scene":      ${N_SCENE},
  "failure_reason": ${FAIL_REASON},
  "shard_log":    "${SHARD_LOG}"
}
EOM

    echo "[p3s2_eval]     manifest: ${CELL_DIR}/manifest.json"
    echo "[p3s2_eval]     PDMS=${PDMS}  n_valid=${N_VALID}/${N_TOTAL}  avg_K_eff=${AVG_KEFF}"
  done
done

echo ""
echo "[p3s2_eval] DONE. results under ${RESULTS_ROOT}/M1b_p3s2_dyn_*"
echo "[p3s2_eval] per-cell summary:"
for d in $(ls -d "${RESULTS_ROOT}"/M1b_p3s2_dyn_* 2>/dev/null | tail -24); do
  if [[ -f "${d}/manifest.json" ]]; then
    "${PY}" -c "
import json
m = json.load(open('${d}/manifest.json'))
print(f'  {m[\"probe_tag\"]:24s} s{m[\"shard\"]}  pdms={m[\"pdms\"]}  avg_K_eff={m[\"avg_k_eff\"]}  rc={m[\"rc\"]}  wall={m[\"wall_seconds\"]}s')
" 2>/dev/null
  fi
done
echo ""
echo "[p3s2_eval] NOTE: PDMS here is PER-SHARD. To get the full navtest PDMS,"
echo "[p3s2_eval]       merge shard0+shard1 merged.csv (same as static sweep aggregation)."
