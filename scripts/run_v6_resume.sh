#!/usr/bin/env bash
# run_v6_resume.sh — Resume v6 counterfactual training from ckpt_step50
# 分支A：v6 shard0 quick eval PDMS > 0.890 时执行
# 用法: bash scripts/run_v6_resume.sh
set -euo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"
NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"

source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="s3_token_scorer_budget_rl_v6_resume_${TIMESTAMP}"
LOG_DIR="$ROOT/logs/v6_resume"
mkdir -p "$LOG_DIR"

# Resume from existing ckpt_step50
RESUME_BASE="$ROOT/ckpt/s3_token_scorer_budget_rl_v6_20260806_162439"

SCORER_CKPT="$ROOT/ckpt/s3_token_scorer"
AUTOVLA_CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
AUTOVLA_CONFIG="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
JSON_DIR="$ROOT/data/navtrain_nocot"
METRIC_CACHE="$ROOT/data/navtrain_metric_cache"
BASELINE_SCORES="$ROOT/results/baseline_sub_scores.json"

# v6 超参 (同原v6)
EFFICIENCY_BETA=0.0
DRIVING_SCALE=3.0
SAFETY_BETA=0.03
SAFETY_MARGIN=0.02
KL_BETA=0.01
BUDGET_KL_BETA=0.0
LR=3e-5
BUDGET_LR=1e-4
BUDGET_LOG_STD_INIT=-1.0
SELECTION_PG_WEIGHT=1.0
NUM_EPOCHS=1
GROUP_SIZE=8
MIN_KR=0.2
MAX_KR=0.9
SEED=42
CF_K=4

echo "============================================"
echo "  v6 RESUME: Per-Token Counterfactual REINFORCE"
echo "  Run: $RUN_NAME"
echo "============================================"
echo "  resume_from: $RESUME_BASE"
echo "  epochs: $NUM_EPOCHS"
echo "  counterfactual_k: $CF_K"
echo "============================================"

PIDS=""
for SH in 0 1 2 3; do
  RESUME_CKPT="${RESUME_BASE}_sh${SH}/ckpt_step50"
  SHARD_OUT="${ROOT}/ckpt/${RUN_NAME}_sh${SH}"
  mkdir -p "$SHARD_OUT"
  
  # Copy resume checkpoint to new dir
  cp -a "$RESUME_CKPT"/* "$SHARD_OUT/" 2>/dev/null || true
  
  LOGFILE="$LOG_DIR/train_sh${SH}.log"
  echo "[launch] Starting shard $SH on GPU $SH (resume from step 50)..."

  (
    export CUDA_VISIBLE_DEVICES=$SH
    cd "$ROOT"
    $PY scripts/train_scorer_budget_rl.py \
      --scorer-ckpt "$SCORER_CKPT" \
      --out-dir "$SHARD_OUT" \
      --json-dir "$JSON_DIR" \
      --metric-cache "$METRIC_CACHE" \
      --sensor-data-path "$SENSOR" \
      --autovla-config "$AUTOVLA_CONFIG" \
      --autovla-ckpt "$AUTOVLA_CKPT" \
      --num-epochs $NUM_EPOCHS \
      --group-size $GROUP_SIZE \
      --lr $LR \
      --budget-lr $BUDGET_LR \
      --kl-beta $KL_BETA \
      --budget-kl-beta $BUDGET_KL_BETA \
      --selection-pg-weight $SELECTION_PG_WEIGHT \
      --efficiency-beta $EFFICIENCY_BETA \
      --driving-scale $DRIVING_SCALE \
      --safety-beta $SAFETY_BETA \
      --safety-margin $SAFETY_MARGIN \
      --min-keep-ratio $MIN_KR \
      --max-keep-ratio $MAX_KR \
      --budget-log-std-init $BUDGET_LOG_STD_INIT \
      --num-shards 4 \
      --shard-id $SH \
      --seed $SEED \
      --prune-variant attn_mask \
      --baseline-scores "$BASELINE_SCORES" \
      --counterfactual-k $CF_K \
      --resume \
      > "$LOGFILE" 2>&1
  ) &
  PIDS="$PIDS $!"
done

echo "[launch] Training PIDs: $PIDS"
echo "$PIDS" > "$LOG_DIR/train.pids"
echo "$RUN_NAME" > "$LOG_DIR/outbase"
echo "[launch] Logs: $LOG_DIR/"
echo "[launch] Ckpt:  $ROOT/ckpt/${RUN_NAME}"
echo "[launch] All 4 shards resumed."
