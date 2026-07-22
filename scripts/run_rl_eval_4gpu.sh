#!/usr/bin/env bash
# run_rl_eval_4gpu.sh — Evaluate RL-shaped scorer on full navtest (4 GPU × 4 shard)
# Called by progress_monitor.sh after RL training completes
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
LOGDIR="$ROOT/logs/_rl_eval"
mkdir -p "$OUTDIR" "$LOGDIR"

# Find best RL scorer ckpt (latest shaped training)
RL_CKPT_BASE=$(ls -dt $ROOT/ckpt/s3_token_scorer_rl_shaped_*_sh0 2>/dev/null | head -1)
if [[ -z "$RL_CKPT_BASE" ]]; then
    echo "[rl-eval] FATAL: no RL shaped ckpt found"; exit 1
fi
# Use the best checkpoint from shard0 (or whichever has best reward)
RL_SCORER="$RL_CKPT_BASE/ckpt_best"
if [[ ! -f "$RL_SCORER/checkpoint.pt" ]]; then
    RL_SCORER="$RL_CKPT_BASE"  # fallback to final
fi
echo "[rl-eval] Using scorer: $RL_SCORER"

log(){ echo "[rl-eval $(date +%H:%M:%S)] $*"; }

# Pre-flight
if pgrep -f "MT_rl_shaped" >/dev/null; then
    log "ABORT: rl eval already running"
    exit 1
fi

PIDS=""
for SH in 0 1 2 3; do
    EXP="MT_rl_shaped_r05_sh${SH}"
    CSV="$OUTDIR/${EXP}.csv"
    
    if [[ -f "$CSV" ]]; then
        log "SKIP $EXP (csv exists)"
        continue
    fi
    
    GPU=$SH
    log "GPU$GPU START $EXP"
    ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES=$GPU
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
        +agent.keep_ratio=0.5 \
        +agent.selector=scorer \
        +agent.scorer_ckpt="$RL_SCORER" \
        +agent.prune_verbose=false \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${EXP}.log" 2>&1 &
    PIDS="$PIDS $!"
done

echo "$PIDS" > "$ROOT/logs/rl_eval.pids"
log "PIDs: $PIDS"
wait
log "ALL SHARDS DONE"

# Report results
for SH in 0 1 2 3; do
    EXP="MT_rl_shaped_r05_sh${SH}"
    FOUND=$(ls -t "$NAVSIM_EXP_ROOT/$EXP"/*/*.csv 2>/dev/null | head -1)
    CSV="$OUTDIR/${EXP}.csv"
    if [[ -n "$FOUND" && ! -f "$CSV" ]]; then
        cp -a "$FOUND" "$CSV"
        log "DONE $EXP -> $CSV"
    fi
done

# Final PDMS calculation
$PY -c "
import pandas as pd
from pathlib import Path
dfs = []
for sh in range(4):
    p = Path('$OUTDIR/MT_rl_shaped_r05_sh' + str(sh) + '.csv')
    if p.exists():
        df = pd.read_csv(p)
        df = df[df['token']!='average']
        dfs.append(df)
if dfs:
    all_df = pd.concat(dfs)
    pdms = all_df['score'].mean()
    print(f'=== RL SHAPED FINAL: N={len(all_df)}, PDMS={pdms:.6f} ===')
    print(f'vs SFT (0.8920): delta={pdms-0.8920:+.4f}pt')
    if pdms > 0.8920:
        print('*** RL WINS! Ready for paper. ***')
    else:
        print('RL < SFT. Check reward shaping or try lr adjustment.')
"

# === RL eval at r=0.25 and r=0.75 (shard0 only, parallel on GPU0-1) ===
log "=== RL eval r=0.25 and r=0.75 (shard0) ==="
for GPU_RATIO in "0:0.25:r025" "1:0.75:r075"; do
    GPU="${GPU_RATIO%%:*}"
    REST="${GPU_RATIO#*:}"
    KR="${REST%%:*}"
    RTAG="${REST##*:}"
    EXP="MT_rl_shaped_${RTAG}_sh0"
    CSV="$OUTDIR/${EXP}.csv"
    [[ -f "$CSV" ]] && { log "SKIP $EXP"; continue; }
    log "GPU$GPU START $EXP"
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
        +agent.keep_ratio=$KR \
        +agent.selector=scorer \
        +agent.scorer_ckpt="$RL_SCORER" \
        +agent.prune_variant=drop \
        +agent.prune_verbose=false \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${EXP}.log" 2>&1 &
done
# Also run RL + τ-cut on GPU2
EXP_TC="MT_rl_taucut_kr060_sh0"
CSV_TC="$OUTDIR/${EXP_TC}.csv"
if [[ ! -f "$CSV_TC" ]]; then
    log "GPU2 START RL + τ-cut"
    ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES=2
      timeout 20000 $PY navsim/planning/script/run_pdm_score_cot.py \
        experiment_name="$EXP_TC" \
        train_test_split="${SHARD_PREFIX}0${SHARD_SUFFIX}" \
        metric_cache_path="$ROOT/data/navtest_metric_cache" \
        +json_data_path="$ROOT/data/navtest_nocot" \
        agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
        +agent.config_path="$YAML" \
        +agent.checkpoint_path="$CKPT" \
        +agent.sensor_data_path="$SENSOR" \
        +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false \
        +agent.keep_ratio=0.5 \
        +agent.selector=scorer_taucut \
        +agent.scorer_ckpt="$RL_SCORER" \
        +agent.tau=-0.1668 \
        +agent.tau_min_keep=36 \
        +agent.prune_variant=drop \
        +agent.prune_verbose=false \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${EXP_TC}.log" 2>&1 &
fi
wait
# Copy CSVs
for EXP in "MT_rl_shaped_r025_sh0" "MT_rl_shaped_r075_sh0" "$EXP_TC"; do
    CSV="$OUTDIR/${EXP}.csv"
    if [[ ! -f "$CSV" ]]; then
        FOUND=$(ls -t "$NAVSIM_EXP_ROOT/$EXP"/*/*.csv 2>/dev/null | head -1)
        [[ -n "$FOUND" ]] && cp -a "$FOUND" "$CSV" && log "DONE $EXP"
    fi
done
log "RL r=0.25/0.75/τ-cut done"

# === Auto-chain: Start Budget RL training ===
log "=== Chaining to Budget RL training ==="
if [[ -f "$ROOT/scripts/run_budget_rl_4gpu.sh" ]]; then
    chmod +x "$ROOT/scripts/run_budget_rl_4gpu.sh"
    nohup bash "$ROOT/scripts/run_budget_rl_4gpu.sh" > "$ROOT/logs/budget_rl_train.log" 2>&1 &
    log "Budget RL launched (PID=$!)"
else
    log "WARN: scripts/run_budget_rl_4gpu.sh not found"
fi
