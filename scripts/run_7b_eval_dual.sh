#!/usr/bin/env bash
# run_7b_eval_dual.sh — ImpromptuVLA 7B + nuScenes eval (cross-model zero-shot transfer)
# Uses 7B scorer trained on base Qwen2.5-VL-7B → zero-shot on ImpromptuVLA 7B (driving fine-tuned)
# Proves: scorer generalizes across models without retraining
# Launch: nohup bash scripts/run_7b_eval_dual.sh > logs/7b_eval_dual.log 2>&1 &
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
export PYTHONPATH="$ROOT/code:$ROOT/code/third_party/AutoVLA:$ROOT/code/third_party/AutoVLA/navsim:${PYTHONPATH:-}"

MODEL_PATH="$ROOT/models/ImpromptuVLA_7B/7B_AD_finetune"
SCORER_7B="$ROOT/ckpt/s3_token_scorer_7b"
DATA_DIR="$ROOT/data/nuscenes_impromptu_val/unpacked"
OUTDIR="$ROOT/results/impromptu7b"
LOGDIR="$ROOT/logs/_7b_impromptu"
mkdir -p "$OUTDIR" "$LOGDIR"

log(){ echo "[7b-impvla $(date +%H:%M:%S)] $*"; }

# Pre-flight
[[ -d "$MODEL_PATH" ]] || { log "FATAL: ImpromptuVLA model not found at $MODEL_PATH"; exit 2; }
[[ -f "$SCORER_7B/checkpoint.pt" ]] || { log "FATAL: 7B scorer not found at $SCORER_7B"; exit 2; }

# Find data json (use q1 which is the driving trajectory prediction task)
DATA_JSON=$(ls "$DATA_DIR"/*q1*.json 2>/dev/null | head -1)
[[ -n "$DATA_JSON" ]] || { log "FATAL: No nuScenes QA json found in $DATA_DIR"; exit 2; }

log "=== ImpromptuVLA 7B nuScenes Eval (Cross-Model Zero-Shot) ==="
log "Model: $MODEL_PATH"
log "Scorer: $SCORER_7B (trained on base Qwen2.5-VL-7B, zero-shot transfer)"
log "Data: $DATA_JSON"

# Run for multiple keep ratios: 0.5, 0.75, and 1.0 (baseline)
# Parallel: each ratio on a different GPU
PIDS=""

for GPU_RATIO in "0:1.0" "1:0.75" "2:0.50" "3:0.25"; do
    GPU="${GPU_RATIO%%:*}"
    RATIO="${GPU_RATIO##*:}"
    RTAG=$(echo $RATIO | tr -d '.')
    OUTPUT="$OUTDIR/pred_r${RTAG}.jsonl"
    
    if [[ -f "$OUTPUT" ]]; then
        log "SKIP r=$RATIO (output exists: $OUTPUT)"
        continue
    fi
    
    log "GPU$GPU START r=$RATIO -> $OUTPUT"
    (
        export CUDA_VISIBLE_DEVICES=$GPU
        $PY scripts/run_impromptu7b_nuscenes_eval.py \
            --model-path "$MODEL_PATH" \
            --scorer-ckpt "$SCORER_7B" \
            --keep-ratio $RATIO \
            --data-json "$DATA_JSON" \
            --output "$OUTPUT" \
            --device cuda:0
    ) > "$LOGDIR/_r${RTAG}.log" 2>&1 &
    PIDS="$PIDS $!"
done

if [[ -n "$PIDS" ]]; then
    log "PIDs: $PIDS"
    wait $PIDS 2>/dev/null
fi

log "=== All ratios done. Running evaluation... ==="

# Evaluate each ratio (compute L2, Collision, Intersection metrics)
for RATIO in 1.0 0.75 0.50 0.25; do
    RTAG=$(echo $RATIO | tr -d '.')
    PRED="$OUTDIR/pred_r${RTAG}.jsonl"
    EVAL_OUT="$OUTDIR/eval_r${RTAG}.json"
    
    if [[ ! -f "$PRED" ]]; then
        log "SKIP eval r=$RATIO (no predictions)"
        continue
    fi
    if [[ -f "$EVAL_OUT" ]]; then
        log "SKIP eval r=$RATIO (already evaluated)"
        continue
    fi
    
    log "Evaluating r=$RATIO..."
    $PY -c "
import json
from pathlib import Path

pred_file = Path('$PRED')
lines = pred_file.read_text().strip().split('\n')
preds = [json.loads(l) for l in lines if l.strip()]
print(f'  r=$RATIO: {len(preds)} predictions')

# Compute metrics if evaluation script available
eval_script = Path('$ROOT/code/third_party/ImpromptuVLA/data_qa_generate/data_engine/datasets/nuscenes/scripts/evaluation_nuscenes.py')
if eval_script.exists():
    import subprocess
    subprocess.run([
        '$PY', str(eval_script),
        '--jsonl_file', '$PRED',
        '--output_file', '$EVAL_OUT',
        '--mode', 'x-y'
    ], check=False)
    if Path('$EVAL_OUT').exists():
        results = json.loads(Path('$EVAL_OUT').read_text())
        print(f'  L2: {results}')
else:
    print('  [WARN] Evaluation script not found, manual eval needed')
" 2>&1 | tee -a "$LOGDIR/_eval.log"
done

# Summary table
log "=== SUMMARY ==="
$PY -c "
import json
from pathlib import Path

print('| r | L2 (cm) | Collision (%) | Intersection (%) | Rel. L2 |')
print('|---|---------|---------------|------------------|---------|')

baseline_l2 = None
for ratio in ['10', '075', '050', '025']:
    eval_f = Path('$OUTDIR/eval_r' + ratio + '.json')
    r_str = {'10': '1.0', '075': '0.75', '050': '0.50', '025': '0.25'}[ratio]
    if eval_f.exists():
        data = json.loads(eval_f.read_text())
        l2 = data.get('l2_avg', data.get('avg_l2', '?'))
        col = data.get('collision_avg', data.get('avg_collision', '?'))
        inter = data.get('intersection_avg', data.get('avg_intersection', '?'))
        if ratio == '10' and isinstance(l2, (int, float)):
            baseline_l2 = l2
        rel = f'{l2/baseline_l2*100:.1f}%' if baseline_l2 and isinstance(l2, (int, float)) else '?'
        print(f'| {r_str} | {l2} | {col} | {inter} | {rel} |')
    else:
        print(f'| {r_str} | (pending) | — | — | — |')

print()
print('Compare with FastDriveVLA Table 1:')
print('  FastDriveVLA r=0.75: L2=32.64, Collision Rel=83.0%, Intersection Rel=96.1%')
print('  FastDriveVLA r=0.50: L2=32.10, Collision Rel=97.3%, Intersection Rel=95.1%')
print('  FastDriveVLA r=0.25: L2=31.80, Collision Rel=93.6%, Intersection Rel=101.1%')
" 2>/dev/null

log "=== 7B ImpromptuVLA Eval COMPLETE ==="
