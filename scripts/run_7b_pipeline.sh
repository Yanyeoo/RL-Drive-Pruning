#!/usr/bin/env bash
# =============================================================================
# run_7b_pipeline.sh — Full 7B token pruning experiment pipeline
#
# End-to-end pipeline for AAAI 2027 7B scaling experiments:
#   Phase 1: Feature dump (7B forward → extract layer-0 vision embeddings)
#   Phase 2: Attention probe (7B forward → extract L12 attention for labels)
#   Phase 3: Train scorer (MLP, LambdaRank, ~30s)
#   Phase 4: Eval (scorer r=0.75, r=0.50 on navtest)
#
# Usage:
#   bash scripts/run_7b_pipeline.sh [phase]
#   phase: all | features | attention | train | eval
#
# Requirements:
#   - models/Qwen2.5-VL-7B-Instruct downloaded
#   - 4-8× H20 GPUs
#   - ~16h total wall time for full pipeline
# =============================================================================
set -euo pipefail

ROOT=/apdcephfs/private_shayladeng/tokenrl_autoVLA
cd "$ROOT"
source scripts/setup_navsim_env_vars.sh

PHASE="${1:-all}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Config ---
MODEL_7B="$ROOT/models/Qwen2.5-VL-7B-Instruct"
CONFIG_7B="$ROOT/code/third_party/AutoVLA/config/training/qwen2.5-vl-7B-navtest-grpo-nocot.yaml"
CODEBOOK="$ROOT/code/third_party/AutoVLA/codebook_cache/agent_vocab.pkl"
SENSOR_DATA="$ROOT/data/navsim_v2_local"

# 7B-specific paths
FEAT_DIR_7B="$ROOT/data/s3_scorer_7b/features"
ATTN_DIR_7B="$ROOT/exp/m1b2_navtrain_7b_alllayers"
SCORER_CKPT_7B="$ROOT/ckpt/s3_token_scorer_7b"
JSON_DIR="$ROOT/data/navtrain_nocot"

# Use navtrain subset for scorer training (same as 3B: 4000 scenes)
MAX_SCENES_TRAIN=4000
# Eval shards
EVAL_SHARDS=4

echo "========================================="
echo "[7B Pipeline] phase=$PHASE ts=$TIMESTAMP"
echo "[7B Pipeline] model=$MODEL_7B"
echo "========================================="

# --- Pre-flight checks ---
if [[ ! -d "$MODEL_7B" ]] || [[ ! -f "$MODEL_7B/config.json" ]]; then
    echo "ERROR: 7B model not found at $MODEL_7B"
    echo "Run: python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='$MODEL_7B')\""
    exit 1
fi

# Verify 7B config
HIDDEN=$(python3 -c "import json; print(json.load(open('$MODEL_7B/config.json'))['hidden_size'])")
echo "[7B Pipeline] hidden_size=$HIDDEN (expected 3584)"
if [[ "$HIDDEN" != "3584" ]]; then
    echo "WARNING: unexpected hidden_size=$HIDDEN (not 3584)"
fi

# ============================================================================
# Phase 1: Feature Dump (layer-0 vision embeddings for scorer training)
# ============================================================================
run_features() {
    echo ""
    echo "=== Phase 1: Feature Dump (7B) ==="
    mkdir -p "$FEAT_DIR_7B"
    local NGPU=4
    local pids=()
    for i in $(seq 0 $((NGPU-1))); do
        echo "[features] Starting shard $i/$NGPU on GPU $i ..."
        CUDA_VISIBLE_DEVICES=$i python3 -m rldrive.scoring.run_feature_dump \
            --save-dir "$FEAT_DIR_7B" \
            --feature-layer 0 \
            --gpu 0 \
            --checkpoint "" \
            --config "$CONFIG_7B" \
            --codebook "$CODEBOOK" \
            --sensor-data "$SENSOR_DATA" \
            --json-dir "$JSON_DIR" \
            --shard-stride $NGPU --shard-index $i \
            --max-scenes $MAX_SCENES_TRAIN \
            --skip-done \
            > "logs/_7b_featdump_sh${i}_${TIMESTAMP}.log" 2>&1 &
        pids+=($!)
    done
    echo "[features] Waiting for ${#pids[@]} shards..."
    local failed=0
    for pid in "${pids[@]}"; do
        wait $pid || ((failed++))
    done
    local n_feats=$(ls "$FEAT_DIR_7B"/*.pt 2>/dev/null | wc -l)
    echo "[features] DONE: $n_feats .pt files, $failed failures"
    if [[ $failed -gt 0 ]]; then
        echo "WARNING: $failed feature dump shards failed. Check logs."
    fi
}

# ============================================================================
# Phase 2: Attention Probe (L12 attention labels for scorer training)
# ============================================================================
run_attention() {
    echo ""
    echo "=== Phase 2: Attention Probe (7B, L12) ==="
    mkdir -p "$ATTN_DIR_7B"
    local NGPU=4
    local pids=()
    for i in $(seq 0 $((NGPU-1))); do
        echo "[attention] Starting shard $i/$NGPU on GPU $i ..."
        CUDA_VISIBLE_DEVICES=$i python3 -m rldrive.scoring.run_attention_probe \
            --save-dir "$ATTN_DIR_7B" \
            --layer-idx 12 \
            --all-layers --num-layers 28 \
            --per-head \
            --gpu 0 \
            --checkpoint "" \
            --config "$CONFIG_7B" \
            --codebook "$CODEBOOK" \
            --sensor-data "$SENSOR_DATA" \
            --json-dir "$JSON_DIR" \
            --shard-stride $NGPU --shard-index $i \
            --max-scenes $MAX_SCENES_TRAIN \
            --skip-done \
            > "logs/_7b_attnprobe_sh${i}_${TIMESTAMP}.log" 2>&1 &
        pids+=($!)
    done
    echo "[attention] Waiting for ${#pids[@]} shards..."
    local failed=0
    for pid in "${pids[@]}"; do
        wait $pid || ((failed++))
    done
    local n_attn=$(ls "$ATTN_DIR_7B"/*.pt 2>/dev/null | wc -l)
    echo "[attention] DONE: $n_attn .pt files, $failed failures"
}

# ============================================================================
# Phase 3: Train Scorer (LambdaRank MLP for 7B)
# ============================================================================
run_train() {
    echo ""
    echo "=== Phase 3: Train Scorer (7B, LambdaRank) ==="
    python3 scripts/s3_build_labels_train_scorer.py \
        --feat-dir "$FEAT_DIR_7B" \
        --label-dir "$ATTN_DIR_7B" \
        --out-dir "$SCORER_CKPT_7B" \
        --label-layer 12 \
        --max-scenes $MAX_SCENES_TRAIN \
        --epochs 20 \
        --pairs-per-scene 1024 \
        --batch-scenes 64 \
        --lr 3e-4 \
        --seed 42 \
        --device cuda:0 \
        2>&1 | tee "logs/_7b_scorer_train_${TIMESTAMP}.log"
    echo "[train] Scorer saved to $SCORER_CKPT_7B"
    cat "$SCORER_CKPT_7B/manifest.json" 2>/dev/null
}

# ============================================================================
# Phase 4: Eval (scorer r=0.75 and r=0.50 on navtest)
# ============================================================================
run_eval() {
    echo ""
    echo "=== Phase 4: Eval (7B scorer on navtest) ==="
    local RATIOS="0.75 0.50"
    local NGPU=$EVAL_SHARDS
    
    for ratio in $RATIOS; do
        local rtag=$(echo $ratio | tr -d '.')
        echo "[eval] Starting r=$ratio ($NGPU shards)..."
        local pids=()
        for sh in $(seq 0 $((NGPU-1))); do
            local exp_name="MT_7b_scorer_r${rtag}_sh${sh}"
            CUDA_VISIBLE_DEVICES=$sh python3 \
                "$ROOT/code/third_party/AutoVLA/navsim/navsim/planning/script/run_pdm_score_cot.py" \
                agent=autovla_with_token_prune \
                +agent.checkpoint_path="" \
                +agent.config_path="$CONFIG_7B" \
                +agent.codebook_cache_path="$CODEBOOK" \
                +agent.sensor_data_path="$SENSOR_DATA" \
                +agent.keep_ratio=$ratio \
                +agent.selector=scorer \
                +agent.scorer_ckpt="$SCORER_CKPT_7B" \
                +agent.prune_variant=attn_mask \
                experiment_name=$exp_name \
                scene_filter=navtest_local_filtered_shard${sh} \
                > "logs/_7b_eval_r${rtag}_sh${sh}_${TIMESTAMP}.log" 2>&1 &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            wait $pid || true
        done
        echo "[eval] r=$ratio DONE. Aggregating..."
        python3 scripts/s3_aggregate_maintable.py \
            --pattern "MT_7b_scorer_r${rtag}_sh*" 2>/dev/null || true
    done
    
    # Also run no-prune baseline for 7B
    echo "[eval] Running 7B no-prune baseline..."
    local pids=()
    for sh in $(seq 0 $((NGPU-1))); do
        local exp_name="MT_7b_noproune_r10_sh${sh}"
        CUDA_VISIBLE_DEVICES=$sh python3 \
            "$ROOT/code/third_party/AutoVLA/navsim/navsim/planning/script/run_pdm_score_cot.py" \
            agent=autovla_with_token_prune \
            +agent.checkpoint_path="" \
            +agent.config_path="$CONFIG_7B" \
            +agent.codebook_cache_path="$CODEBOOK" \
            +agent.sensor_data_path="$SENSOR_DATA" \
            +agent.keep_ratio=1.0 \
            +agent.selector=attn_L12 \
            +agent.prune_variant=attn_mask \
            experiment_name=$exp_name \
            scene_filter=navtest_local_filtered_shard${sh} \
            > "logs/_7b_eval_r10_sh${sh}_${TIMESTAMP}.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait $pid || true
    done
    echo "[eval] 7B no-prune DONE."
}

# ============================================================================
# Dispatch
# ============================================================================
mkdir -p logs

case "$PHASE" in
    all)
        run_features
        run_attention
        run_train
        run_eval
        ;;
    features)
        run_features
        ;;
    attention)
        run_attention
        ;;
    train)
        run_train
        ;;
    eval)
        run_eval
        ;;
    *)
        echo "Usage: $0 [all|features|attention|train|eval]"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "[7B Pipeline] $PHASE complete at $(date)"
echo "========================================="
