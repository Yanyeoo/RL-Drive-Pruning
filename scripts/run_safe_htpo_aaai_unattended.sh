#!/usr/bin/env bash
# One-command unattended AAAI Safe-HTPO-Lite run:
# synchronized 8-GPU joint training -> four-shard raw Variant-B eval -> aggregation.
set -Eeuo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
source "$ROOT/scripts/setup_navsim_env_vars.sh" >/dev/null
export PYTHONPATH="$ROOT/code:$ROOT/code/third_party/AutoVLA/navsim:$ROOT/code/third_party/AutoVLA:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

SAFE_HTPO_RUN_ID="${SAFE_HTPO_RUN_ID:-safehtpo_$(date +%Y%m%d_%H%M%S)}"
SAFE_EPOCHS="${SAFE_EPOCHS:-1}"
SAFE_GROUP="${SAFE_GROUP:-16}"
SAFE_EFFICIENCY_BETA="${SAFE_EFFICIENCY_BETA:-0.15}"
SAFE_SAFETY_BETA="${SAFE_SAFETY_BETA:-0.50}"
SAFE_SAFETY_MARGIN="${SAFE_SAFETY_MARGIN:-0.00}"
SAFE_DRIVING_SCALE="${SAFE_DRIVING_SCALE:-2.0}"
SAFE_SELECTION_PG_WEIGHT="${SAFE_SELECTION_PG_WEIGHT:-1.0}"
# Strict default wall-clock budget: smoke <=0.5h + train <=6h + eval <=4h + aggregation buffer.
SAFE_TRAIN_TIMEOUT_SEC="${SAFE_TRAIN_TIMEOUT_SEC:-21600}"
SAFE_MAX_TRAIN_ATTEMPTS="${SAFE_MAX_TRAIN_ATTEMPTS:-1}"
SAFE_RUN_SMOKE="${SAFE_RUN_SMOKE:-1}"
SAFE_SMOKE_SCENES="${SAFE_SMOKE_SCENES:-32}"
SAFE_SMOKE_GROUP="${SAFE_SMOKE_GROUP:-2}"
SAFE_SMOKE_TIMEOUT_SEC="${SAFE_SMOKE_TIMEOUT_SEC:-1800}"

OUT_DIR="$ROOT/ckpt/${SAFE_HTPO_RUN_ID}"
LOG_DIR="$ROOT/logs/${SAFE_HTPO_RUN_ID}"
STATUS_FILE="$LOG_DIR/status.json"
TRAIN_LOG="$LOG_DIR/train.log"
mkdir -p "$OUT_DIR" "$LOG_DIR"

log() { echo "[safe-htpo $(date '+%F %T')] $*" | tee -a "$LOG_DIR/orchestrator.log"; }
write_status() {
  cat > "$STATUS_FILE" <<EOF
{"run_id":"${SAFE_HTPO_RUN_ID}","stage":"$1","time":"$(date -Iseconds)","out_dir":"${OUT_DIR}","message":"$2"}
EOF
}
fail() { write_status "FAILED" "$1"; log "FATAL: $1"; exit 1; }
trap 'rc=$?; write_status "FAILED" "orchestrator exited with code ${rc}"; exit ${rc}' ERR

[[ -f "$ROOT/ckpt/s3_token_scorer/checkpoint.pt" ]] || fail "Missing SFT scorer checkpoint"
[[ -f "$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt" ]] || fail "Missing AutoVLA checkpoint"
[[ ! -e "$OUT_DIR/checkpoint.pt" ]] || fail "Refusing to overwrite completed run directory: $OUT_DIR"

"$PY" -m py_compile \
  "$ROOT/scripts/train_scorer_budget_rl.py" \
  "$ROOT/scripts/aggregate_safe_htpo_eval.py" \
  "$ROOT/code/third_party/AutoVLA/models/utils/score.py" || fail "Python syntax preflight failed"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  [[ "$GPU_COUNT" -ge 8 ]] || fail "Need 8 GPUs; nvidia-smi reports ${GPU_COUNT}"
  log "GPU preflight sees ${GPU_COUNT} devices"
else
  log "WARN: nvidia-smi unavailable; torchrun/CUDA is the final GPU availability check"
fi

cat > "$OUT_DIR/run_config.json" <<EOF
{
  "run_id": "${SAFE_HTPO_RUN_ID}",
  "algorithm": "Safe-HTPO-Lite: synchronized joint REINFORCE with random logistic-normal budget and Top-K selection surrogate",
  "epochs": ${SAFE_EPOCHS},
  "group_size": ${SAFE_GROUP},
  "keep_ratio_range": [0.2, 0.9],
  "selection_pg_weight": ${SAFE_SELECTION_PG_WEIGHT},
  "efficiency_beta": ${SAFE_EFFICIENCY_BETA},
  "safety_beta": ${SAFE_SAFETY_BETA},
  "safety_margin": ${SAFE_SAFETY_MARGIN},
  "driving_scale": ${SAFE_DRIVING_SCALE},
  "training_prune_variant": "attn_mask",
  "evaluation_prune_variant": "drop",
  "baseline_source": "same-scene unpruned VLA rollout captured during feature extraction",
  "denylist": null,
  "world_size": 8
}
EOF

echo "$OUT_DIR" > "$ROOT/logs/safe_htpo_active_outdir.txt"

if [[ "$SAFE_RUN_SMOKE" == "1" ]]; then
  write_status "SMOKE" "Running synchronized DDP smoke before final training"
  log "SMOKE: ${SAFE_SMOKE_SCENES} scenes, group=${SAFE_SMOKE_GROUP}, all 8 ranks"
  timeout --signal=TERM --kill-after=5m "$SAFE_SMOKE_TIMEOUT_SEC" \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$ROOT/scripts/train_scorer_budget_rl.py" \
      --distributed \
      --scorer-ckpt "$ROOT/ckpt/s3_token_scorer" \
      --out-dir "$OUT_DIR/smoke" \
      --json-dir "$ROOT/data/navtrain_nocot" \
      --metric-cache "$ROOT/data/navtrain_metric_cache" \
      --sensor-data-path "/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/sensor_blobs/test/openscene-v1.1/sensor_blobs/test" \
      --autovla-config "$ROOT/code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml" \
      --autovla-ckpt "$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt" \
      --num-epochs 1 --max-scenes "$SAFE_SMOKE_SCENES" --group-size "$SAFE_SMOKE_GROUP" \
      --lr 3e-5 --budget-lr 1e-4 --kl-beta 0.01 \
      --selection-pg-weight "$SAFE_SELECTION_PG_WEIGHT" \
      --efficiency-beta "$SAFE_EFFICIENCY_BETA" \
      --safety-beta "$SAFE_SAFETY_BETA" --safety-margin "$SAFE_SAFETY_MARGIN" \
      --driving-scale "$SAFE_DRIVING_SCALE" \
      --min-keep-ratio 0.2 --max-keep-ratio 0.9 \
      --prune-variant attn_mask \
      --save-every 1 --seed 42 --device cuda:0 >> "$LOG_DIR/smoke.log" 2>&1 \
    || fail "DDP smoke failed; final training was not started"
  [[ -f "$OUT_DIR/smoke/checkpoint.pt" ]] || fail "Smoke finished without checkpoint"
  [[ -f "$OUT_DIR/smoke/ddp_sync_audit.json" ]] || fail "Smoke finished without DDP sync audit"
  grep -q 'selection_log_prob_mean' "$OUT_DIR/smoke/train_log.jsonl" \
    || fail "Smoke produced no joint selection-policy training records"
  log "SMOKE passed: synchronized checkpoint, audit, and joint-policy records found"
fi

write_status "TRAINING" "Starting synchronized 8-GPU Safe-HTPO-Lite training"
log "RUN=${SAFE_HTPO_RUN_ID}; output=${OUT_DIR}; epochs=${SAFE_EPOCHS}; random k in [0.2,0.9]"

TRAIN_RC=1
for ATTEMPT in $(seq 1 "$SAFE_MAX_TRAIN_ATTEMPTS"); do
  log "TRAIN attempt ${ATTEMPT}/${SAFE_MAX_TRAIN_ATTEMPTS}"
  set +e
  timeout --signal=TERM --kill-after=10m "$SAFE_TRAIN_TIMEOUT_SEC" \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$ROOT/scripts/train_scorer_budget_rl.py" \
      --distributed \
      --scorer-ckpt "$ROOT/ckpt/s3_token_scorer" \
      --out-dir "$OUT_DIR" \
      --json-dir "$ROOT/data/navtrain_nocot" \
      --metric-cache "$ROOT/data/navtrain_metric_cache" \
      --sensor-data-path "/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/sensor_blobs/test/openscene-v1.1/sensor_blobs/test" \
      --autovla-config "$ROOT/code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml" \
      --autovla-ckpt "$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt" \
      --num-epochs "$SAFE_EPOCHS" \
      --group-size "$SAFE_GROUP" \
      --lr 3e-5 \
      --budget-lr 1e-4 \
      --kl-beta 0.01 \
      --selection-pg-weight "$SAFE_SELECTION_PG_WEIGHT" \
      --efficiency-beta "$SAFE_EFFICIENCY_BETA" \
      --safety-beta "$SAFE_SAFETY_BETA" \
      --safety-margin "$SAFE_SAFETY_MARGIN" \
      --driving-scale "$SAFE_DRIVING_SCALE" \
      --min-keep-ratio 0.2 \
      --max-keep-ratio 0.9 \
      --prune-variant attn_mask \
      --save-every 25 \
      --seed 42 \
      --device cuda:0 >> "$TRAIN_LOG" 2>&1
  TRAIN_RC=$?
  set -e
  [[ "$TRAIN_RC" -eq 0 ]] && break
  if [[ -f "$OUT_DIR/ckpt_resume/checkpoint.pt" && "$ATTEMPT" -lt "$SAFE_MAX_TRAIN_ATTEMPTS" ]]; then
    log "TRAIN failed with rc=${TRAIN_RC}; resume checkpoint exists, retrying once"
    sleep 30
  else
    fail "Training failed with rc=${TRAIN_RC}; inspect ${TRAIN_LOG}"
  fi
done

[[ -f "$OUT_DIR/checkpoint.pt" ]] || fail "Training exited without final checkpoint"
[[ -f "$OUT_DIR/manifest.json" ]] || fail "Training exited without manifest"
touch "$OUT_DIR/TRAIN_DONE"
write_status "EVALUATING" "Training complete; starting no-denylist Variant-B full navtest evaluation"
log "TRAIN complete; launching four-shard dynamic Variant-B evaluation"

export SAFE_HTPO_CHECKPOINT="$OUT_DIR"
export SAFE_HTPO_RUN_ID
export SAFE_EVAL_TIMEOUT_SEC="${SAFE_EVAL_TIMEOUT_SEC:-14400}"
bash "$ROOT/scripts/eval_safe_htpo_dynamic_4gpu.sh" >> "$LOG_DIR/eval_orchestrator.log" 2>&1

[[ -f "$ROOT/results/raw/safe_htpo_${SAFE_HTPO_RUN_ID}/EVAL_DONE" ]] || fail "Evaluation did not write EVAL_DONE"
write_status "DONE" "Training and full raw no-denylist dynamic evaluation complete"
log "DONE. Summary: $ROOT/results/raw/safe_htpo_${SAFE_HTPO_RUN_ID}/safe_htpo_summary.md"
