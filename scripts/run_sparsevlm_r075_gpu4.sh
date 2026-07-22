#!/usr/bin/env bash
# run_sparsevlm_r075_gpu4.sh — SparseVLM (text-guided) r=0.75, full navtest, 4 shards serial on GPU4
# Launch: nohup bash scripts/run_sparsevlm_r075_gpu4.sh > logs/sparsevlm_r075.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SHARD_PREFIX="navtest_local_filtered_shard"
SHARD_SUFFIX="_20260616_154858"
OUTDIR="$ROOT/results/raw/tokenprune_S3_full"
LOGDIR="$ROOT/logs/_sparsevlm_r075"
mkdir -p "$OUTDIR" "$LOGDIR"

log(){ echo "[sparse075 $(date +%H:%M:%S)] $*"; }

# Pre-flight: no double launch
if pgrep -f "run_pdm_score_cot.*sparsevlm" >/dev/null; then
    log "ABORT: sparsevlm eval already running"
    exit 1
fi

for SH in 0 1 2 3; do
    EXP="MT_sparsevlm_text_r075_sh${SH}"
    CSV="$OUTDIR/${EXP}.csv"
    
    # Skip if already done
    if [[ -f "$CSV" ]]; then
        log "SKIP $EXP (csv exists)"
        continue
    fi
    
    log "START shard$SH"
    ( cd "$NAVSIM_ROOT"
      timeout 40000 $PY navsim/planning/script/run_pdm_score_cot.py \
        experiment_name="$EXP" \
        train_test_split="${SHARD_PREFIX}${SH}${SHARD_SUFFIX}" \
        metric_cache_path="$ROOT/data/navtest_metric_cache" \
        +json_data_path="$ROOT/data/navtest_nocot" \
        agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
        +agent.config_path="$YAML" \
        +agent.checkpoint_path="$CKPT" \
        +agent.sensor_data_path="$SENSOR" \
        +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false \
        +agent.keep_ratio=0.75 \
        +agent.selector=sparsevlm_text \
        +agent.prune_verbose=false \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${EXP}.log" 2>&1
    RC=$?
    
    # Copy CSV from exp dir
    FOUND=$(ls -t "$NAVSIM_EXP_ROOT/$EXP"/*/*.csv 2>/dev/null | head -1)
    if [[ -n "$FOUND" ]]; then
        cp -a "$FOUND" "$CSV"
        log "DONE shard$SH rc=$RC -> $CSV"
        $PY -c "import pandas as pd; df=pd.read_csv('$CSV'); df=df[df['token']!='average']; print(f'  [PDMS] $EXP N={len(df)} PDMS={df[\"score\"].mean():.6f}')" 2>/dev/null || true
    else
        log "WARN shard$SH rc=$RC no csv (see $LOGDIR/_${EXP}.log)"
    fi
done

log "ALL DONE"

# Also run Variant B r=0.75 if time allows
log "=== Starting Variant B r=0.75 ==="
for SH in 0 1 2 3; do
    EXP="MT_varBsafe_scorer_r075_sh${SH}"
    CSV="$OUTDIR/${EXP}.csv"
    
    if [[ -f "$CSV" ]]; then
        log "SKIP $EXP (csv exists)"
        continue
    fi
    
    log "START VarB shard$SH"
    ( cd "$NAVSIM_ROOT"
      timeout 40000 $PY navsim/planning/script/run_pdm_score_cot.py \
        experiment_name="$EXP" \
        train_test_split="${SHARD_PREFIX}${SH}${SHARD_SUFFIX}" \
        metric_cache_path="$ROOT/data/navtest_metric_cache" \
        +json_data_path="$ROOT/data/navtest_nocot" \
        agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
        +agent.config_path="$YAML" \
        +agent.checkpoint_path="$CKPT" \
        +agent.sensor_data_path="$SENSOR" \
        +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false \
        +agent.keep_ratio=0.75 \
        +agent.selector=scorer \
        +agent.scorer_ckpt="$ROOT/ckpt/s3_token_scorer" \
        +agent.prune_variant=drop \
        +agent.prune_verbose=false \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${EXP}.log" 2>&1
    RC=$?
    
    FOUND=$(ls -t "$NAVSIM_EXP_ROOT/$EXP"/*/*.csv 2>/dev/null | head -1)
    if [[ -n "$FOUND" ]]; then
        cp -a "$FOUND" "$CSV"
        log "DONE VarB shard$SH rc=$RC -> $CSV"
        $PY -c "import pandas as pd; df=pd.read_csv('$CSV'); df=df[df['token']!='average']; print(f'  [PDMS] $EXP N={len(df)} PDMS={df[\"score\"].mean():.6f}')" 2>/dev/null || true
    else
        log "WARN VarB shard$SH rc=$RC no csv"
    fi
done
log "=== ALL EXPERIMENTS DONE ==="
# Auto-chain: baseline pareto after VarB
log "=== Chaining to baseline pareto ==="
bash "$ROOT/scripts/run_baseline_pareto_gpu4.sh"
