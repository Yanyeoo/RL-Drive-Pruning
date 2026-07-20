#!/usr/bin/env bash
# =============================================================================
# run_7b_21h_master.sh — 21h window master dispatcher for 7B scaling experiments
#
# Window: 2026-07-20 20:00 → 2026-07-21 17:00 (21h, 8× H20)
# Goal: Get 7B token pruning results for AAAI 2027 scaling law story
#
# Timeline:
#   Phase 0 (20:00-20:30): Verify 7B model, apply patches, smoke test
#   Phase 1 (20:30-03:30): Feature dump + Attention dump on 7B base (7h, 4+4 GPU)
#   Phase 2 (03:30-04:00): Train 7B scorer (30min, 1 GPU)
#   Phase 3 (04:00-10:00): 7B LoRA fine-tune on navtrain subset (6h, 4 GPU)
#                           + Run 3B remaining experiments (τ-cut etc) on other 4 GPU
#   Phase 4 (10:00-17:00): Eval 7B scorer on fine-tuned 7B (r=0.75, r=0.50, r=1.0)
#
# Fallback if no 7B fine-tune:
#   Use 7B base model directly — PDMS will be low (~0.5-0.7) but the 
#   RELATIVE pruning tolerance (r=0.75 vs r=1.0 delta) is still meaningful
#   for the scaling law argument.
# =============================================================================
set -euo pipefail

ROOT=/apdcephfs/private_shayladeng/tokenrl_autoVLA
cd "$ROOT"
source scripts/setup_navsim_env_vars.sh
export PYTHONPATH="$ROOT/code:$ROOT/code/third_party/AutoVLA:$ROOT/code/third_party/AutoVLA/navsim:$PYTHONPATH"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$ROOT/logs/7b_21h_${TIMESTAMP}"
mkdir -p "$LOGDIR"

MODEL_7B="$ROOT/models/Qwen2.5-VL-7B-Instruct"
CONFIG_7B="$ROOT/code/third_party/AutoVLA/config/training/qwen2.5-vl-7B-navtest-grpo-nocot.yaml"
CODEBOOK="$ROOT/code/third_party/AutoVLA/codebook_cache/agent_vocab.pkl"
SENSOR_DATA="$ROOT/data/navsim_v2_local"
JSON_DIR="$ROOT/data/navtrain_nocot"

FEAT_DIR="$ROOT/data/s3_scorer_7b/features"
ATTN_DIR="$ROOT/exp/m1b2_7b_attn_labels"
SCORER_CKPT="$ROOT/ckpt/s3_token_scorer_7b"

MAX_SCENES=4000  # Same as 3B experiment (fair comparison)

# Stop file
STOP_FILE="$ROOT/STOP_7B"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/master.log"; }

check_stop() {
    if [[ -f "$STOP_FILE" ]]; then
        log "STOP file detected. Halting."
        exit 0
    fi
}

# ==========================================================================
# Phase 0: Pre-flight
# ==========================================================================
log "=== Phase 0: Pre-flight checks ==="

if [[ ! -f "$MODEL_7B/config.json" ]]; then
    log "ERROR: 7B model not found at $MODEL_7B"
    log "Download should have completed. Check: ls $MODEL_7B/*.safetensors"
    exit 1
fi

HIDDEN=$(python3 -c "import json; print(json.load(open('$MODEL_7B/config.json'))['hidden_size'])")
NLAYERS=$(python3 -c "import json; print(json.load(open('$MODEL_7B/config.json'))['num_hidden_layers'])")
log "7B model verified: hidden=$HIDDEN, layers=$NLAYERS"

# Verify data paths
if [[ ! -d "$JSON_DIR" ]]; then
    log "ERROR: navtrain_nocot not found at $JSON_DIR"
    exit 1
fi
N_JSONS=$(ls "$JSON_DIR"/*.json 2>/dev/null | wc -l)
log "navtrain_nocot: $N_JSONS scenes available"

check_stop

# ==========================================================================
# Phase 1: Feature + Attention Dump (parallel, 4+4 GPU, ~7h)
# ==========================================================================
log "=== Phase 1: Feature + Attention Dump (7B base model) ==="

mkdir -p "$FEAT_DIR" "$ATTN_DIR"

# GPU 0-3: Feature dump (layer-0 vision embeddings)
log "Starting feature dump on GPU 0-3..."
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python3 -m rldrive.scoring.run_feature_dump \
        --save-dir "$FEAT_DIR" \
        --feature-layer 0 \
        --gpu 0 \
        --checkpoint "" \
        --config "$CONFIG_7B" \
        --codebook "$CODEBOOK" \
        --sensor-data "$SENSOR_DATA" \
        --json-dir "$JSON_DIR" \
        --shard-stride 4 --shard-index $i \
        --max-scenes $MAX_SCENES \
        --skip-done \
        > "$LOGDIR/feat_sh${i}.log" 2>&1 &
done

# GPU 4-7: Attention probe (L* attention for scorer labels)
# NOTE: For 7B (28 layers), L*=12 may still be optimal (middle-ish layer).
# If 7B has 28 layers, L12 is at position 12/28 ≈ 43% depth (vs 3B: 12/36 ≈ 33%).
# Alternative: L10 (10/28 ≈ 36%) to match relative depth. Start with L12 for comparability.
log "Starting attention probe (L12) on GPU 4-7..."
for i in 0 1 2 3; do
    gpu=$((i + 4))
    CUDA_VISIBLE_DEVICES=$gpu python3 -m rldrive.scoring.run_attention_probe \
        --save-dir "$ATTN_DIR" \
        --layer-idx 12 \
        --multi-layer \
        --gpu 0 \
        --checkpoint "" \
        --config "$CONFIG_7B" \
        --codebook "$CODEBOOK" \
        --sensor-data "$SENSOR_DATA" \
        --json-dir "$JSON_DIR" \
        --shard-stride 4 --shard-index $i \
        --max-scenes $MAX_SCENES \
        --skip-done \
        > "$LOGDIR/attn_sh${i}.log" 2>&1 &
done

log "Phase 1 launched (8 processes). Waiting..."
wait
log "Phase 1 complete."

N_FEAT=$(ls "$FEAT_DIR"/*.pt 2>/dev/null | wc -l)
N_ATTN=$(ls "$ATTN_DIR"/*.pt 2>/dev/null | wc -l)
log "Features: $N_FEAT .pt | Attention: $N_ATTN .pt"

check_stop

# ==========================================================================
# Phase 2: Train 7B Scorer (LambdaRank, ~30s)
# ==========================================================================
log "=== Phase 2: Train 7B Scorer ==="

python3 scripts/s3_build_labels_train_scorer.py \
    --feat-dir "$FEAT_DIR" \
    --label-dir "$ATTN_DIR" \
    --out-dir "$SCORER_CKPT" \
    --label-layer 12 \
    --max-scenes $MAX_SCENES \
    --epochs 20 \
    --pairs-per-scene 1024 \
    --batch-scenes 64 \
    --lr 3e-4 \
    --seed 42 \
    --device cuda:0 \
    2>&1 | tee "$LOGDIR/scorer_train.log"

if [[ -f "$SCORER_CKPT/checkpoint.pt" ]]; then
    log "Scorer trained successfully: $SCORER_CKPT"
    cat "$SCORER_CKPT/manifest.json"
else
    log "ERROR: Scorer training failed!"
    exit 1
fi

check_stop

# ==========================================================================
# Phase 3: 7B LoRA Fine-tune (background, for eval later)
# ==========================================================================
log "=== Phase 3: 7B LoRA Fine-tune (if patch applied) ==="
log "NOTE: Fine-tune requires run_rft.py patch. Skipping auto-launch."
log "Manual command (after patch):"
log "  cd $ROOT/code/third_party/AutoVLA && CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_rft.py --config training/qwen2.5-vl-7B-navtest-grpo-nocot"

# For now, proceed to Phase 4 using 7B base model (no fine-tune).
# The scorer + pruning tolerance comparison is still valid.

# ==========================================================================
# Phase 4: Eval 7B (scorer r=0.75, r=0.50, r=1.0 on navtest)
# ==========================================================================
log "=== Phase 4: Eval 7B scorer on navtest ==="

# Use the existing eval infrastructure but with 7B model + 7B scorer
RATIOS="0.75 0.50"
NGPU=4

for ratio in $RATIOS; do
    check_stop
    rtag=$(echo $ratio | tr -d '.')
    log "Eval r=$ratio ($NGPU shards)..."
    for sh in 0 1 2 3; do
        exp_name="MT_7b_scorer_r${rtag}_sh${sh}"
        CUDA_VISIBLE_DEVICES=$sh python3 \
            "$ROOT/code/third_party/AutoVLA/navsim/navsim/planning/script/run_pdm_score_cot.py" \
            agent=autovla_with_token_prune \
            +agent.checkpoint_path="" \
            +agent.config_path="$CONFIG_7B" \
            +agent.codebook_cache_path="$CODEBOOK" \
            +agent.sensor_data_path="$SENSOR_DATA" \
            +agent.keep_ratio=$ratio \
            +agent.selector=scorer \
            +agent.scorer_ckpt="$SCORER_CKPT" \
            +agent.prune_variant=attn_mask \
            experiment_name=$exp_name \
            scene_filter=navtest_local_filtered_shard${sh} \
            > "$LOGDIR/eval_r${rtag}_sh${sh}.log" 2>&1 &
    done
    wait
    log "Eval r=$ratio done."
done

# No-prune baseline
log "Eval r=1.0 (no-prune baseline)..."
for sh in 0 1 2 3; do
    exp_name="MT_7b_noprune_r10_sh${sh}"
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
        > "$LOGDIR/eval_r10_sh${sh}.log" 2>&1 &
done
wait
log "Eval r=1.0 done."

# ==========================================================================
# Aggregate
# ==========================================================================
log "=== Aggregating results ==="
python3 scripts/s3_aggregate_maintable.py --pattern "MT_7b_*" 2>&1 | tee -a "$LOGDIR/master.log" || true

log ""
log "========================================="
log "7B 21h Master Pipeline COMPLETE"
log "Results: results/raw/tokenprune_S3_full/MT_7b_*"
log "Scorer: $SCORER_CKPT"
log "========================================="
