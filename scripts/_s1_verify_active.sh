#!/usr/bin/env bash
# _s1_verify_active.sh — S1 GPU verify launcher (active, self-driven loop).
# Usage: bash scripts/_s1_verify_active.sh <mode> <gpu> <exp_name>
#   mode = new_r10 | new_r05 | vanilla
# Runs ONE eval on navtest_smoke5_shard0 and exits with the child's rc.
# Env/args mirror scripts/run_m1b_freelunch_sweep.sh + _auto_continue_20260703.sh.
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
SMOKE="navtest_smoke5_shard0_20260616_154725"

MODE="$1"; GPU="$2"; EXP="$3"

COMMON=(
  experiment_name="$EXP"
  train_test_split="$SMOKE"
  metric_cache_path="${ROOT}/data/navtest_metric_cache"
  +json_data_path="${ROOT}/data/navtest_nocot"
  +agent.config_path="$TRAIN_YAML"
  +agent.checkpoint_path="$CKPT"
  +agent.sensor_data_path="$SENSOR_DATA"
  +agent.codebook_cache_path="${AUTOVLA_ROOT}/codebook_cache/agent_vocab.pkl"
  +agent.lora_conf.use_lora=false
  worker=single_machine_thread_pool
  worker.max_workers=1
)

case "$MODE" in
  vanilla)
    ARGS=( agent._target_=rldrive.agents.autovla_with_attention.AutoVLAWithAttentionAgent
           "${COMMON[@]}" +agent.attention_enabled=false ) ;;
  new_r10)
    ARGS=( agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent
           "${COMMON[@]}" +agent.keep_ratio=1.0 +agent.selector=attn_L12 +agent.prune_verbose=true ) ;;
  new_r05)
    ARGS=( agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent
           "${COMMON[@]}" +agent.keep_ratio=0.5 +agent.selector=attn_L12 +agent.prune_verbose=true ) ;;
  new_r01)
    ARGS=( agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent
           "${COMMON[@]}" +agent.keep_ratio=0.1 +agent.selector=attn_L12 +agent.prune_verbose=true ) ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac

echo "[s1v] mode=$MODE gpu=$GPU exp=$EXP  $(date -Iseconds)"
cd "$NAVSIM_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" timeout --signal=TERM --kill-after=30s 1800s \
  "$PY" navsim/planning/script/run_pdm_score_cot.py "${ARGS[@]}"
rc=$?
echo "[s1v] mode=$MODE exp=$EXP rc=$rc  $(date -Iseconds)"
exit $rc
