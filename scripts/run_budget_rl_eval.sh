#!/usr/bin/env bash
# run_budget_rl_eval.sh — Evaluate budget RL scorer (dynamic keep_ratio)
# Then chain to ImpromptuVLA 7B eval
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SHARD_PREFIX="navtest_local_filtered_shard"
SHARD_SUFFIX="_20260616_154858"
OUTDIR="$ROOT/results/raw/tokenprune_S3_full"
LOGDIR="$ROOT/logs/_budget_rl_eval"
mkdir -p "$OUTDIR" "$LOGDIR"

log(){ echo "[budget-eval $(date +%H:%M:%S)] $*"; }

# Find budget RL ckpt
BUDGET_CKPT=$(ls -dt $ROOT/ckpt/s3_token_scorer_budget_rl_*_sh0/ckpt_best 2>/dev/null | head -1)
if [[ -z "$BUDGET_CKPT" || ! -f "$BUDGET_CKPT/checkpoint.pt" ]]; then
    BUDGET_CKPT=$(ls -dt $ROOT/ckpt/s3_token_scorer_budget_rl_*_sh0 2>/dev/null | head -1)
fi
if [[ -z "$BUDGET_CKPT" ]]; then
    log "FATAL: No budget RL ckpt found"; exit 2
fi
log "Using budget RL ckpt: $BUDGET_CKPT"

# TODO: Budget RL eval needs a special agent that uses TokenScorerWithBudget
# For now, eval with the token_net scores at various fixed ratios to see if selection improved
# Plus one τ-cut style eval using the budget head's natural threshold

# Eval at fixed ratios (using token_net from budget RL scorer)
PIDS=""
for GPU_RATIO in "0:0.50" "1:0.75" "2:0.25"; do
    GPU="${GPU_RATIO%%:*}"
    RATIO="${GPU_RATIO##*:}"
    RTAG=$(echo $RATIO | tr -d '.')
    EXP="MT_budget_rl_r${RTAG}_sh0"
    CSV="$OUTDIR/${EXP}.csv"
    [[ -f "$CSV" ]] && { log "SKIP $EXP"; continue; }

    log "GPU$GPU START $EXP (r=$RATIO)"
    ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES=$GPU
      timeout 20000 $PY navsim/planning/script/run_pdm_score_cot.py \
        experiment_name="$EXP" \
        train_test_split="${SHARD_PREFIX}0${SHARD_SUFFIX}" \
        metric_cache_path="$ROOT/data/navtest_metric_cache" \
        +json_data_path="$ROOT/data/navtest_nocot" \
        agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
        +agent.config_path="$YAML" \
        +agent.checkpoint_path="$CKPT" \
        +agent.sensor_data_path="$SENSOR" \
        +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false \
        +agent.keep_ratio=$RATIO \
        +agent.selector=scorer \
        +agent.scorer_ckpt="$BUDGET_CKPT" \
        +agent.prune_variant=drop \
        +agent.prune_verbose=false \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${EXP}.log" 2>&1 &
    PIDS="$PIDS $!"
done
[[ -n "$PIDS" ]] && wait $PIDS 2>/dev/null

# Copy CSVs
for RATIO in 0.50 0.75 0.25; do
    RTAG=$(echo $RATIO | tr -d '.')
    EXP="MT_budget_rl_r${RTAG}_sh0"
    CSV="$OUTDIR/${EXP}.csv"
    if [[ ! -f "$CSV" ]]; then
        FOUND=$(ls -t "$NAVSIM_EXP_ROOT/$EXP"/*/*.csv 2>/dev/null | head -1)
        [[ -n "$FOUND" ]] && cp -a "$FOUND" "$CSV" && log "DONE $EXP"
    fi
done

# Report
$PY -c "
import pandas as pd
from pathlib import Path
print('=== Budget RL Eval Results ===')
for rtag, r in [('025',0.25),('050',0.5),('075',0.75)]:
    p = Path('$OUTDIR/MT_budget_rl_r' + rtag + '_sh0.csv')
    if p.exists():
        df = pd.read_csv(p); df=df[df['token']!='average']
        print(f'  r={r}: N={len(df)}, PDMS={df[\"score\"].mean():.4f}')
" 2>/dev/null

log "=== Budget RL eval done. Chaining to 7B eval ==="

# Chain to 7B ImpromptuVLA eval
if [[ -f "$ROOT/scripts/run_7b_eval_dual.sh" ]]; then
    nohup bash "$ROOT/scripts/run_7b_eval_dual.sh" > "$ROOT/logs/7b_eval_dual.log" 2>&1 &
    log "7B eval launched (PID=$!)"
fi
