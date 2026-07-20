#!/usr/bin/env bash
# run_mse_scorer_eval.sh — Eval MSE scorer at r=0.5 on shard0 (quick ablation check)
# Purpose: C1 ablation evidence — compare MSE scorer PDMS vs LambdaRank scorer PDMS
# Launch: nohup bash scripts/run_mse_scorer_eval.sh > logs/_mse_scorer_eval.log 2>&1 &
# Stop:   touch STOP_MSE
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
OUT="$ROOT/results/raw/tokenprune_S3_full"; mkdir -p "$OUT" logs
SHARD_PREFIX="navtest_local_filtered_shard"; SHARD_SUFFIX="_20260616_154858"
MSE_CKPT="$ROOT/ckpt/s3_token_scorer_mse"
log(){ echo "[mse-eval $(date +%H:%M:%S)] $*"; }

if pgrep -f run_pdm_score_cot >/dev/null; then log "ABORT: pdm_score already running"; exit 1; fi

# Quick eval: MSE scorer r=0.5 on shard0 only (takes ~1h on 1 GPU)
GPU=0
EXP="MT_scorer_mse_r05_sh0"
CSV="$OUT/${EXP}.csv"

if [[ -f "$CSV" ]]; then
  log "SKIP $EXP (already done)"
  exit 0
fi

log "Start $EXP on GPU$GPU"
( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$GPU"
  timeout 20000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
    experiment_name="$EXP" train_test_split="${SHARD_PREFIX}0${SHARD_SUFFIX}" \
    metric_cache_path="$ROOT/data/navtest_metric_cache" +json_data_path="$ROOT/data/navtest_nocot" \
    agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
    +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" +agent.sensor_data_path="$SENSOR" \
    +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
    +agent.lora_conf.use_lora=false +agent.keep_ratio=0.5 +agent.selector=scorer \
    +agent.scorer_ckpt="$MSE_CKPT" \
    +agent.prune_verbose=false worker=single_machine_thread_pool worker.max_workers=1
) > "logs/_mt_${EXP}.log" 2>&1; rc=$?

c=$(ls -t "$NAVSIM_EXP_ROOT/$EXP"/*/*.csv 2>/dev/null | head -1)
[[ -n "$c" ]] && cp -a "$c" "$CSV" && log "DONE $EXP rc=$rc -> csv" || log "$EXP rc=$rc WARN no csv"

# Report
if [[ -f "$CSV" ]]; then
  "$PY" -c "
import pandas as pd
df = pd.read_csv('$CSV')
df = df[df['token'] != 'average']
print(f'MSE scorer r=0.5 shard0: N={len(df)}, PDMS={df[\"score\"].mean():.6f}')
print(f'Compare: LambdaRank scorer r=0.5 shard0 PDMS=0.891766 (from MT_scorer_r05_sh0.csv)')
"
fi
