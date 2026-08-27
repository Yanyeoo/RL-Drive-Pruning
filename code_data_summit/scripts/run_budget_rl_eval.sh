#!/usr/bin/env bash
# run_budget_rl_eval.sh — Evaluate budget RL scorer with the DYNAMIC per-scene budget.
#
# Unlike the old version (which ignored the budget head and only evaluated fixed
# ratios), this now uses selector=scorer_budget: each scene gets its own keep_ratio
# from the learned budget head, then top-B tokens are pruned at that ratio.
# A fixed-r=0.5 run is kept alongside for apples-to-apples comparison.
#
# Then chains to ImpromptuVLA 7B eval.
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

# Find budget RL ckpt (prefer best; it carries the budget_net head)
BUDGET_CKPT=$(ls -dt $ROOT/ckpt/s3_token_scorer_budget_rl_*_sh0/ckpt_best 2>/dev/null | head -1)
if [[ -z "$BUDGET_CKPT" || ! -f "$BUDGET_CKPT/checkpoint.pt" ]]; then
    BUDGET_CKPT=$(ls -dt $ROOT/ckpt/s3_token_scorer_budget_rl_*_sh0 2>/dev/null | head -1)
fi
if [[ -z "$BUDGET_CKPT" ]]; then
    log "FATAL: No budget RL ckpt found"; exit 2
fi
log "Using budget RL ckpt: $BUDGET_CKPT"

# === (1) DYNAMIC per-scene budget on GPU0 ===
EXP_DYN="MT_budget_rl_dynamic_sh0"
CSV_DYN="$OUTDIR/${EXP_DYN}.csv"
if [[ -f "$CSV_DYN" ]]; then
    log "SKIP $EXP_DYN (exists)"
else
    log "GPU0 START $EXP_DYN (dynamic per-scene budget)"
    ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES=0
      timeout 30000 $PY navsim/planning/script/run_pdm_score_cot.py \
        experiment_name="$EXP_DYN" \
        train_test_split="${SHARD_PREFIX}0${SHARD_SUFFIX}" \
        metric_cache_path="$ROOT/data/navtest_metric_cache" \
        +json_data_path="$ROOT/data/navtest_nocot" \
        agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
        +agent.config_path="$YAML" \
        +agent.checkpoint_path="$CKPT" \
        +agent.sensor_data_path="$SENSOR" \
        +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false \
        +agent.selector=scorer_budget \
        +agent.scorer_ckpt="$BUDGET_CKPT" \
        +agent.keep_ratio=0.5 \
        +agent.prune_variant=drop \
        +agent.prune_verbose=true \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${EXP_DYN}.log" 2>&1 &
    PID_DYN=$!
fi

# === (2) FIXED r=0.5 on GPU1 (comparison) ===
EXP_FIX="MT_budget_rl_r050_sh0"
CSV_FIX="$OUTDIR/${EXP_FIX}.csv"
if [[ -f "$CSV_FIX" ]]; then
    log "SKIP $EXP_FIX (exists)"
else
    log "GPU1 START $EXP_FIX (fixed r=0.5, for comparison)"
    ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES=1
      timeout 30000 $PY navsim/planning/script/run_pdm_score_cot.py \
        experiment_name="$EXP_FIX" \
        train_test_split="${SHARD_PREFIX}0${SHARD_SUFFIX}" \
        metric_cache_path="$ROOT/data/navtest_metric_cache" \
        +json_data_path="$ROOT/data/navtest_nocot" \
        agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
        +agent.config_path="$YAML" \
        +agent.checkpoint_path="$CKPT" \
        +agent.sensor_data_path="$SENSOR" \
        +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false \
        +agent.selector=scorer \
        +agent.scorer_ckpt="$BUDGET_CKPT" \
        +agent.keep_ratio=0.5 \
        +agent.prune_variant=drop \
        +agent.prune_verbose=false \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${EXP_FIX}.log" 2>&1 &
    PID_FIX=$!
fi

[[ -n "$PID_DYN" ]] && wait $PID_DYN 2>/dev/null
[[ -n "$PID_FIX" ]] && wait $PID_FIX 2>/dev/null

# Copy CSVs
for EXP in "$EXP_DYN" "$EXP_FIX"; do
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
print('=== Budget RL Eval (navtest) ===')
for tag, exp in [('dynamic', '$EXP_DYN'), ('fixed_r050', '$EXP_FIX')]:
    p = Path('$OUTDIR/' + exp + '.csv')
    if p.exists():
        df = pd.read_csv(p); df = df[df['token']!='average']
        print(f'  {tag:12s}: N={len(df)}, PDMS={df[\"score\"].mean():.4f}')
" 2>/dev/null

# Summarize per-scene keep_ratio spread from the dynamic eval log
if [[ -f "$LOGDIR/_${EXP_DYN}.log" ]]; then
    KR=$(grep -oE "kr=[0-9.]+" "$LOGDIR/_${EXP_DYN}.log" 2>/dev/null | sed 's/kr=//' | \
         awk '{s+=$1; s2+=$1*$1; n++} END{if(n>0) printf "mean=%.3f std=%.3f n=%d", s/n, sqrt(s2/n-(s/n)^2), n}')
    log "Dynamic budget keep_ratio: $KR"
fi

log "=== Budget RL eval done. Chaining to 7B eval ==="

# Chain to 7B ImpromptuVLA eval
if [[ -f "$ROOT/scripts/run_7b_eval_dual.sh" ]]; then
    nohup bash "$ROOT/scripts/run_7b_eval_dual.sh" > "$ROOT/logs/7b_eval_dual.log" 2>&1 &
    log "7B eval launched (PID=$!)"
fi
