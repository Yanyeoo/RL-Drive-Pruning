#!/usr/bin/env bash
# run_tokenprune_sweep.sh — S2 headroom-gate single-arm runner (clean, path B).
# Mirrors scripts/run_m1b_freelunch_sweep.sh but swaps the head_mask knobs for the
# token-prune agent's (keep_ratio, selector). Runs ONE (selector,keep_ratio) arm on
# ONE split/gpu, then copies the per-scene CSV into results/raw/tokenprune_S2/ for
# the oracle post-processing (docs/specs/dynamic_headroom_gate_S2_spec.md).
#
# Usage: run_tokenprune_sweep.sh <selector> <keep_ratio> <split> <gpu> <exp_name> [timeout_s]
#   selector    = attn_L12 | random
#   keep_ratio  = 1.0 | 0.75 | 0.5 | 0.25 ...
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="${ROOT}/code/third_party/AutoVLA"
NAVSIM_ROOT="${AUTOVLA_ROOT}/navsim"
source "${ROOT}/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="${ROOT}/code:${NAVSIM_ROOT}:${AUTOVLA_ROOT}:${PYTHONPATH:-}"
CKPT="${ROOT}/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
TRAIN_YAML="${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR_DATA="${ROOT}/data/navsim_v2_local"
OUTDIR="${ROOT}/results/raw/tokenprune_S2"; mkdir -p "$OUTDIR"

SELECTOR="$1"; KR="$2"; SPLIT="$3"; GPU="$4"; EXP="$5"; TMO="${6:-36000}"

HYDRA_ARGS=(
  experiment_name="$EXP"
  train_test_split="$SPLIT"
  metric_cache_path="${ROOT}/data/navtest_metric_cache"
  +json_data_path="${ROOT}/data/navtest_nocot"
  agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent
  +agent.config_path="$TRAIN_YAML"
  +agent.checkpoint_path="$CKPT"
  +agent.sensor_data_path="$SENSOR_DATA"
  +agent.codebook_cache_path="${AUTOVLA_ROOT}/codebook_cache/agent_vocab.pkl"
  +agent.lora_conf.use_lora=false
  +agent.keep_ratio="$KR"
  +agent.selector="$SELECTOR"
  +agent.prune_verbose=false
  worker=single_machine_thread_pool
  worker.max_workers=1
)

echo "[tp_sweep] arm selector=$SELECTOR keep_ratio=$KR split=$SPLIT gpu=$GPU exp=$EXP  $(date -Iseconds)"
T0=$(date +%s); RC=0
( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$GPU"
  exec timeout --signal=TERM --kill-after=30s "${TMO}s" \
    "$PY" navsim/planning/script/run_pdm_score_cot.py "${HYDRA_ARGS[@]}"
) ; RC=$?
WALL=$(( $(date +%s) - T0 ))
echo "[tp_sweep] exp=$EXP rc=$RC wall=${WALL}s  $(date -Iseconds)"

# harvest per-scene CSV
C=$(ls -t "${NAVSIM_EXP_ROOT}/${EXP}"/*/*.csv 2>/dev/null | head -1)
if [[ -n "$C" ]]; then
  cp -a "$C" "${OUTDIR}/${EXP}.csv"
  echo "[tp_sweep] csv -> ${OUTDIR}/${EXP}.csv"
else
  echo "[tp_sweep] WARN no csv for $EXP (rc=$RC)"
fi
exit $RC
