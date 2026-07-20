#!/usr/bin/env bash
# ============================================================================
# run_autovla_navtest_dual_gpu.sh — AutoVLA → NAVSIM navtest inference dispatch
#                                    across N GPUs (MA2.3 dispatcher).
# ----------------------------------------------------------------------------
# (Filename retained for backward compat; despite "dual_gpu" the dispatcher
#  supports any GPUS list, e.g. "0 1 2 3" for 4x H20.)
#
# Strategy:
#   1) Build a token list by intersecting navtest scene_filter with available
#      metric_cache; shard it by token-hash mod N (N = number of GPUs).
#   2) Write N scene_filter yamls (navtest_shard{0..N-1}) with that split.
#   3) Launch N hydra processes in parallel (one per GPU), each running
#      run_pdm_score_cot.py with its own shard yaml.
#   4) Merge per-shard csv into a single results table and report B0 PDMS.
#
# This script is reusable for MA2.4 (smoke) and MA2.5 (full); the only
# difference is SCENE_FILTER_SRC.
#
# Knobs (env-overridable):
#   SCENE_FILTER_SRC  navtest scene_filter source yaml name
#                     (default `navtest_local_filtered`; for smoke
#                      use `navtest_smoke5` etc.)
#   JSON_DIR          MA2.1 json dir (default data/navtest_nocot)
#   METRIC_CACHE      MA2.2 cache dir (default data/navtest_metric_cache)
#   GPUS              space-separated device indices (default "0 1";
#                     for 4-GPU full-run use "0 1 2 3")
#   TIMEOUT_PER_GPU   seconds per shard (default 86400 = 24h)
#   EXP_TAG           tag used in experiment_name (default "ma2_dual")
# ============================================================================
set -uo pipefail

PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
AUTOVLA_ROOT="${PROJECT_ROOT}/code/third_party/AutoVLA"
NAVSIM_ROOT="${AUTOVLA_ROOT}/navsim"
SCENE_FILTER_DIR="${NAVSIM_ROOT}/navsim/planning/script/config/common/train_test_split/scene_filter"
SPLIT_DIR="${NAVSIM_ROOT}/navsim/planning/script/config/common/train_test_split"

# ----- env vars / python -----
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/setup_navsim_env_vars.sh"
PY="${PY:-/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python}"
if [[ ! -x "${PY}" ]]; then echo "[dispatch] FATAL: no python at ${PY}" >&2; exit 2; fi

# Required so hydra can resolve `navsim.agents.autovla_agent.AutoVLAAgent`
# (matches single-GPU smoke script).
export PYTHONPATH="${NAVSIM_ROOT}:${AUTOVLA_ROOT}:${PYTHONPATH:-}"

# ----- knobs -----
SCENE_FILTER_SRC="${SCENE_FILTER_SRC:-navtest_local_filtered}"
JSON_DIR="${JSON_DIR:-${PROJECT_ROOT}/data/navtest_nocot}"
METRIC_CACHE="${METRIC_CACHE:-${PROJECT_ROOT}/data/navtest_metric_cache}"
read -ra GPU_ARR <<< "${GPUS:-0 1}"
TIMEOUT_PER_GPU="${TIMEOUT_PER_GPU:-86400}"
EXP_TAG="${EXP_TAG:-ma2_dual}"

CKPT="${PROJECT_ROOT}/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
QWEN_BASE="${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"
TRAIN_YAML="${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR_DATA="${PROJECT_ROOT}/data/navsim_v2_local"   # placeholder; agent reads abs paths from json

for f in "${CKPT}" "${QWEN_BASE}" "${JSON_DIR}" "${METRIC_CACHE}" "${TRAIN_YAML}" \
         "${SCENE_FILTER_DIR}/${SCENE_FILTER_SRC}.yaml"; do
  if [[ ! -e "${f}" ]]; then
    echo "[dispatch] FATAL: asset not found: ${f}" >&2
    exit 3
  fi
done

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "[dispatch] PROJECT_ROOT   = ${PROJECT_ROOT}"
echo "[dispatch] SCENE_FILTER   = ${SCENE_FILTER_SRC}"
echo "[dispatch] JSON_DIR       = ${JSON_DIR}"
echo "[dispatch] METRIC_CACHE   = ${METRIC_CACHE}"
echo "[dispatch] GPUS           = ${GPU_ARR[*]}"
echo "[dispatch] TIMEOUT/gpu    = ${TIMEOUT_PER_GPU}s"
echo "[dispatch] EXP_TAG        = ${EXP_TAG}"
echo "[dispatch] TS             = ${TS}"

# ---------------------------------------------------------------------------
# 1) Shard token list by hash mod N (intersected with metric_cache)
# ---------------------------------------------------------------------------
NSHARD="${#GPU_ARR[@]}"
if (( NSHARD < 1 )); then echo "[dispatch] FATAL: GPUS empty" >&2; exit 4; fi
echo "[dispatch] sharding token list into ${NSHARD} shards..."

"${PY}" - <<PYEOF || { echo "[dispatch] FATAL: sharding failed" >&2; exit 4; }
import sys, hashlib, yaml
from pathlib import Path

NSHARD = ${NSHARD}
src   = Path("${SCENE_FILTER_DIR}/${SCENE_FILTER_SRC}.yaml")
cache = Path("${METRIC_CACHE}")
jsons = Path("${JSON_DIR}")

with src.open() as f:
    cfg = yaml.safe_load(f)

src_tokens = set(cfg.get('tokens') or [])
cached = {p.name for p in cache.glob('*/unknown/*') if p.is_dir()}
json_tokens = {p.stem for p in jsons.glob('*.json')}
elig = sorted(src_tokens & cached & json_tokens)
print(f'[shard] src={len(src_tokens)} cached={len(cached)} json={len(json_tokens)} eligible={len(elig)} nshard={NSHARD}', flush=True)

if not elig:
    print('[shard] FATAL: no eligible tokens (intersection empty)', flush=True); sys.exit(5)

shards = {i: [] for i in range(NSHARD)}
for t in elig:
    h = int(hashlib.sha1(t.encode()).hexdigest(), 16)
    shards[h % NSHARD].append(t)

needed_logs = set(cfg.get('log_names') or [])

for sid in range(NSHARD):
    out = dict(cfg)
    out['tokens'] = shards[sid]
    out['log_names'] = sorted(needed_logs)
    p = Path("${SCENE_FILTER_DIR}") / f"${SCENE_FILTER_SRC}_shard{sid}_${TS}.yaml"
    with p.open('w') as f:
        yaml.safe_dump(out, f)
    sp = Path("${SPLIT_DIR}") / f"${SCENE_FILTER_SRC}_shard{sid}_${TS}.yaml"
    with sp.open('w') as f:
        yaml.safe_dump({
            'defaults': [{'scene_filter': f"${SCENE_FILTER_SRC}_shard{sid}_${TS}"}],
            'data_split': 'test',
        }, f)
    print(f'[shard{sid}] tokens={len(shards[sid])} yaml={p}', flush=True)
PYEOF

# ---------------------------------------------------------------------------
# 2) Launch N hydra processes in parallel
# ---------------------------------------------------------------------------
SHARD_PIDS=()
SHARD_LOGS=()
SHARD_EXPNAMES=()
for ((sid=0; sid<NSHARD; sid++)); do
  GPU="${GPU_ARR[${sid}]}"
  SPLIT_NAME="${SCENE_FILTER_SRC}_shard${sid}_${TS}"
  EXP_NAME="${EXP_TAG}_shard${sid}_${TS}"
  SHARD_LOG="${LOG_DIR}/${EXP_TAG}_shard${sid}_${TS}.log"
  echo "[dispatch] launching shard${sid} on GPU ${GPU}: ${EXP_NAME}"
  # Subshell so each process gets its own cwd=NAVSIM_ROOT (matches single-GPU
  # smoke). `exec` keeps PID stable so $! captures the actual python process.
  (
    cd "${NAVSIM_ROOT}" || exit 99
    export CUDA_VISIBLE_DEVICES="${GPU}"
    exec timeout --signal=TERM --kill-after=30s "${TIMEOUT_PER_GPU}s" \
      "${PY}" "${NAVSIM_ROOT}/navsim/planning/script/run_pdm_score_cot.py" \
        experiment_name="${EXP_NAME}" \
        train_test_split="${SPLIT_NAME}" \
        metric_cache_path="${METRIC_CACHE}" \
        +json_data_path="${JSON_DIR}" \
        +agent.config_path="${TRAIN_YAML}" \
        +agent.checkpoint_path="${CKPT}" \
        +agent.sensor_data_path="${SENSOR_DATA}" \
        +agent.codebook_cache_path="${AUTOVLA_ROOT}/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false \
        worker=single_machine_thread_pool \
        worker.max_workers=1
  ) > "${SHARD_LOG}" 2>&1 &
  SHARD_PIDS+=($!)
  SHARD_LOGS+=("${SHARD_LOG}")
  SHARD_EXPNAMES+=("${EXP_NAME}")
  echo "[dispatch]   pid=${SHARD_PIDS[-1]} log=${SHARD_LOG}"
done

echo "[dispatch] waiting for ${NSHARD} shards (monitor: tail -f ${SHARD_LOGS[0]}) ..."
SHARD_RCS=()
for ((sid=0; sid<NSHARD; sid++)); do
  RC=0
  wait "${SHARD_PIDS[${sid}]}" || RC=$?
  SHARD_RCS+=("${RC}")
  echo "[dispatch] shard${sid} rc=${RC}"
done

# ---------------------------------------------------------------------------
# 3) Merge per-shard csvs and report aggregate PDMS
# ---------------------------------------------------------------------------
echo "[dispatch] merging shard csvs..."
MERGE_OUT="${PROJECT_ROOT}/exp/${EXP_TAG}_merged_${TS}"
mkdir -p "${MERGE_OUT}"

# Build a comma-separated python literal list of shard exp names so the
# python heredoc can iterate regardless of N.
EXP_NAMES_PY_LIST="$(printf '"%s",' "${SHARD_EXPNAMES[@]}")"

"${PY}" - <<PYEOF || true
import sys
from pathlib import Path
import pandas as pd

exp_root = Path("${NAVSIM_EXP_ROOT}")
out_dir  = Path("${MERGE_OUT}")
out_dir.mkdir(parents=True, exist_ok=True)

exp_names = [${EXP_NAMES_PY_LIST}]
print(f'[merge] expecting {len(exp_names)} shards', flush=True)

frames = []
for exp_name in exp_names:
    base = exp_root / exp_name
    csvs = list(base.rglob('*.csv'))
    if not csvs:
        print(f'[merge] WARN: no csv under {base}', flush=True); continue
    for c in csvs:
        df = pd.read_csv(c)
        df['__shard'] = exp_name
        frames.append(df)
        print(f'[merge] loaded {c} rows={len(df)}', flush=True)

if not frames:
    print('[merge] FATAL: no csv loaded', flush=True); sys.exit(0)

m = pd.concat(frames, ignore_index=True)
# Each navsim per-shard csv ends with an extra aggregate row (token='average');
# drop it so the merge produces a clean per-token table.
m = m[m['token'] != 'average'].copy()
m = m.drop_duplicates(subset='token')
print(f'[merge] total unique tokens = {len(m)}', flush=True)
print(f'[merge] valid = {int(m["valid"].sum())}  invalid = {int((~m["valid"]).sum())}', flush=True)
print(f'[merge] mean score (valid) = {m.loc[m["valid"], "score"].mean():.4f}', flush=True)
for col in ('no_at_fault_collisions','drivable_area_compliance','ego_progress',
            'time_to_collision_within_bound','comfort','driving_direction_compliance'):
    if col in m.columns:
        print(f'[merge]   mean {col} = {m.loc[m["valid"], col].mean():.4f}', flush=True)

out_csv = out_dir / 'merged.csv'
m.to_csv(out_csv, index=False)
print(f'[merge] wrote {out_csv}', flush=True)
PYEOF

echo "[dispatch] DONE. tag=${EXP_TAG} ts=${TS}"
echo "[dispatch] shard logs:"
for L in "${SHARD_LOGS[@]}"; do echo "    ${L}"; done
echo "[dispatch] merged dir: ${MERGE_OUT}"
