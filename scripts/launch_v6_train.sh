#!/usr/bin/env bash
# launch_v6_train.sh — v6 Per-Token Counterfactual REINFORCE 训练启动脚本
#
# 改动 vs v5:
#   - Per-token counterfactual credit assignment: 每个 kept token 单独评估贡献
#   - token_net 用 per-token advantage 更新（不再是 scene-level shared advantage）
#   - budget_net 保持 Gaussian REINFORCE（scene-level）
#   - counterfactual_k=4: 每个 scene 采样 4 个 kept token 做 counterfactual
#   - 计算量: baseline(1) + pruned(1) + K*counterfactual(4) = 6 VLA forwards/scene
#     vs v5: 2 VLA forwards/scene → 3x slower per step, 但每个 step 信号更强
#
# 用法: bash scripts/launch_v6_train.sh
set -euo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"
NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"

source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="s3_token_scorer_budget_rl_v6_${TIMESTAMP}"
OUT_BASE="$ROOT/ckpt/${RUN_NAME}"
LOG_DIR="$ROOT/logs/v6_train"
mkdir -p "$LOG_DIR" "$OUT_BASE"

SCORER_CKPT="$ROOT/ckpt/s3_token_scorer"
AUTOVLA_CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
AUTOVLA_CONFIG="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
JSON_DIR="$ROOT/data/navtrain_nocot"
METRIC_CACHE="$ROOT/data/navtrain_metric_cache"
BASELINE_SCORES="$ROOT/results/baseline_sub_scores.json"

# v6 超参
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
NUM_EPOCHS=1                 # v6: 1 epoch (~6h), counterfactual 每步慢3x
GROUP_SIZE=8                 # 减小 group_size 因为每步更慢
MIN_KR=0.2
MAX_KR=0.9
SEED=42
CF_K=4                       # counterfactual tokens per scene

echo "============================================"
echo "  v6 Per-Token Counterfactual REINFORCE"
echo "  Run: $RUN_NAME"
echo "  Timestamp: $TIMESTAMP"
echo "============================================"
echo "  efficiency_beta = $EFFICIENCY_BETA (NO bonus)"
echo "  driving_scale   = $DRIVING_SCALE"
echo "  safety_beta     = $SAFETY_BETA"
echo "  epochs          = $NUM_EPOCHS"
echo "  group_size      = $GROUP_SIZE"
echo "  counterfactual_k= $CF_K"
echo "  reward          = True PDMS product"
echo "  token update    = Per-token counterfactual advantage"
echo "  budget update   = Scene-level Gaussian REINFORCE"
echo "============================================"

cat > "$OUT_BASE/v6_config.json" << EOF
{
  "run_name": "$RUN_NAME",
  "timestamp": "$TIMESTAMP",
  "version": "v6",
  "description": "Per-token counterfactual REINFORCE + Gaussian budget, true PDMS reward",
  "efficiency_beta": $EFFICIENCY_BETA,
  "driving_scale": $DRIVING_SCALE,
  "safety_beta": $SAFETY_BETA,
  "safety_margin": $SAFETY_MARGIN,
  "num_epochs": $NUM_EPOCHS,
  "group_size": $GROUP_SIZE,
  "lr": $LR,
  "budget_lr": $BUDGET_LR,
  "kl_beta": $KL_BETA,
  "budget_kl_beta": $BUDGET_KL_BETA,
  "selection_pg_weight": $SELECTION_PG_WEIGHT,
  "budget_log_std_init": $BUDGET_LOG_STD_INIT,
  "min_keep_ratio": $MIN_KR,
  "max_keep_ratio": $MAX_KR,
  "seed": $SEED,
  "counterfactual_k": $CF_K,
  "reward_type": "true_pdms_product",
  "token_update": "per_token_counterfactual",
  "budget_update": "scene_level_gaussian"
}
EOF

PIDS=""
for SH in 0 1 2 3; do
  SHARD_OUT="${OUT_BASE}_sh${SH}"
  mkdir -p "$SHARD_OUT"
  LOGFILE="$LOG_DIR/train_sh${SH}.log"

  echo "[launch] Starting shard $SH on GPU $SH ..."

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
      > "$LOGFILE" 2>&1
  ) &
  PIDS="$PIDS $!"
done

echo "[launch] Training PIDs: $PIDS"
echo "$PIDS" > "$LOG_DIR/train.pids"
echo "$OUT_BASE" > "$LOG_DIR/outbase"
echo "[launch] Logs: $LOG_DIR/"
echo "[launch] Ckpt:  $OUT_BASE"
echo "[launch] All 4 shards launched."
