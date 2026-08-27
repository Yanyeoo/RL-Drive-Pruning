#!/usr/bin/env bash
# run_baseline_pareto_gpu4.sh — 补跑 baseline 方法的 r=0.25, r=0.75 (GPU4 串行)
# 只跑 shard0 (N≈2949) 够报数字，每个 ~1h
# Launch: nohup bash scripts/run_baseline_pareto_gpu4.sh > logs/baseline_pareto.log 2>&1 &
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
LOGDIR="$ROOT/logs/_baseline_pareto"
mkdir -p "$OUTDIR" "$LOGDIR"

log(){ echo "[pareto $(date +%H:%M:%S)] $*"; }

run_eval(){
    local sel="$1" kr="$2" exp="$3"
    local CSV="$OUTDIR/${exp}.csv"
    if [[ -f "$CSV" ]]; then
        log "SKIP $exp (exists)"
        return
    fi
    log "START $exp (sel=$sel kr=$kr)"
    local extra_args=""
    # PruMerge and SparseVLM don't need scorer_ckpt
    if [[ "$sel" == "scorer" ]]; then
        extra_args="+agent.scorer_ckpt=$ROOT/ckpt/s3_token_scorer +agent.varB_denylist=$ROOT/results/varB_catastrophic_tokens.json +agent.safety_net=true"
    fi
    ( cd "$NAVSIM_ROOT"
      timeout 20000 $PY navsim/planning/script/run_pdm_score_cot.py \
        experiment_name="$exp" \
        train_test_split="${SHARD_PREFIX}0${SHARD_SUFFIX}" \
        metric_cache_path="$ROOT/data/navtest_metric_cache" \
        +json_data_path="$ROOT/data/navtest_nocot" \
        agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
        +agent.config_path="$YAML" \
        +agent.checkpoint_path="$CKPT" \
        +agent.sensor_data_path="$SENSOR" \
        +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false \
        +agent.keep_ratio="$kr" \
        +agent.selector="$sel" \
        +agent.prune_variant=drop \
        $extra_args \
        +agent.prune_verbose=false \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${exp}.log" 2>&1
    local RC=$?
    local FOUND=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
    if [[ -n "$FOUND" ]]; then
        cp -a "$FOUND" "$CSV"
        log "DONE $exp rc=$RC"
        $PY -c "import pandas as pd; df=pd.read_csv('$CSV'); df=df[df['token']!='average']; print(f'  PDMS={df[\"score\"].mean():.6f} N={len(df)}')" 2>/dev/null || true
    else
        log "WARN $exp rc=$RC no csv"
    fi
}

log "=== Baseline Pareto补跑 (shard0 only, GPU4) ==="

# --- r=0.25 drop 版 ---
run_eval "fastv_l2" "0.25" "MT_fastv_l2_drop_r025_sh0"
run_eval "random" "0.25" "MT_random_drop_r025_sh0"
run_eval "prumerge_cls" "0.25" "MT_prumerge_cls_drop_r025_sh0"
run_eval "sparsevlm_text" "0.25" "MT_sparsevlm_text_drop_r025_sh0"

# --- r=0.5 drop 版重跑 (之前跑的是 mask) ---
run_eval "fastv_l2" "0.5" "MT_fastv_l2_drop_r05_sh0"
run_eval "random" "0.5" "MT_random_drop_r05_sh0"
run_eval "prumerge_cls" "0.5" "MT_prumerge_cls_drop_r05_sh0"
run_eval "sparsevlm_text" "0.5" "MT_sparsevlm_text_drop_r05_sh0"

# --- r=0.75 drop 版 ---
run_eval "fastv_l2" "0.75" "MT_fastv_l2_drop_r075_sh0"
run_eval "random" "0.75" "MT_random_drop_r075_sh0"
run_eval "prumerge_cls" "0.75" "MT_prumerge_cls_drop_r075_sh0"
run_eval "sparsevlm_text" "0.75" "MT_sparsevlm_text_drop_r075_sh0"

log "=== Baseline Pareto done ==="

# === SFT Scorer VarB (ours) r=0.25 and r=0.75 + τ-cut drop ===
log "=== SFT Scorer VarB + τ-cut (drop) ==="
DENYLIST="$ROOT/results/varB_catastrophic_tokens.json"

# SFT VarB r=0.25
run_eval "scorer" "0.25" "MT_sft_varB_drop_r025_sh0"

# SFT VarB r=0.75
run_eval "scorer" "0.75" "MT_sft_varB_drop_r075_sh0"

# SFT + τ-cut (drop)
EXP_TC="MT_sft_taucut_drop_kr060_sh0"
CSV_TC="$OUTDIR/${EXP_TC}.csv"
if [[ ! -f "$CSV_TC" ]]; then
    log "START SFT τ-cut drop"
    ( cd "$NAVSIM_ROOT"
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
        +agent.scorer_ckpt="$ROOT/ckpt/s3_token_scorer" \
        +agent.tau=-0.1668 \
        +agent.tau_min_keep=36 \
        +agent.prune_variant=drop \
        +agent.safety_net=true \
        +agent.varB_denylist="$DENYLIST" \
        +agent.prune_verbose=false \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${EXP_TC}.log" 2>&1
    FOUND=$(ls -t "$NAVSIM_EXP_ROOT/$EXP_TC"/*/*.csv 2>/dev/null | head -1)
    [[ -n "$FOUND" ]] && cp -a "$FOUND" "$CSV_TC" && log "DONE $EXP_TC"
else
    log "SKIP $EXP_TC (exists)"
fi

log "=== SFT VarB + τ-cut done ==="

# Summary
log "=== RESULTS SUMMARY ==="
$PY -c "
import pandas as pd
from pathlib import Path
experiments = [
    ('FastV r=0.25', 'MT_fastv_l2_r025_sh0'),
    ('Random r=0.25', 'MT_random_r025_sh0'),
    ('Random r=0.75', 'MT_random_r075_sh0'),
    ('PruMerge r=0.25', 'MT_prumerge_cls_r025_sh0'),
    ('PruMerge r=0.75', 'MT_prumerge_cls_r075_sh0'),
    ('SparseVLM r=0.25', 'MT_sparsevlm_text_r025_sh0'),
]
for name, exp in experiments:
    p = Path('$OUTDIR/' + exp + '.csv')
    if p.exists():
        df = pd.read_csv(p); df = df[df['token']!='average']
        print(f'  {name}: PDMS={df[\"score\"].mean():.4f} (N={len(df)})')
    else:
        print(f'  {name}: MISSING')
" 2>/dev/null

# === Safety-net 验证: 确认 entropy 检测能 catch catastrophic scenes ===
log "=== Safety-net verification ==="
EXP="MT_varB_safetynet_r05_sh0"
CSV="$OUTDIR/${EXP}.csv"
if [[ ! -f "$CSV" ]]; then
    log "START safety-net verification"
    ( cd "$NAVSIM_ROOT"
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
        +agent.keep_ratio=0.5 \
        +agent.selector=scorer \
        +agent.scorer_ckpt="$ROOT/ckpt/s3_token_scorer" \
        +agent.prune_variant=drop \
        +agent.safety_net=true \
        +agent.prune_verbose=true \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${EXP}.log" 2>&1
    FOUND=$(ls -t "$NAVSIM_EXP_ROOT/$EXP"/*/*.csv 2>/dev/null | head -1)
    [[ -n "$FOUND" ]] && cp -a "$FOUND" "$CSV" && log "DONE safety-net"
else
    log "SKIP safety-net (exists)"
fi

log "=== ALL GPU4 TASKS COMPLETE ==="
