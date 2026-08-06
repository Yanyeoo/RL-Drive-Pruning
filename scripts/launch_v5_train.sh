#!/usr/bin/env bash
# launch_v5_train.sh — v5 True PDMS RL 训练启动脚本
# 
# 改动 vs v4:
#   - use_true_pdms=True: reward 直接使用真实 PDMS product（与 eval 完全一致）
#   - 去掉 efficiency_bonus: budget head 从纯 driving quality signal 学习最优 kr
#   - safety_beta=0.03（降低安全惩罚，让 driving reward 主导）
#   - 2 epochs: 给模型更多时间从 true PDMS signal 学习
#
# 用法: bash scripts/launch_v5_train.sh
set -euo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"
NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"

# 环境
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="s3_token_scorer_budget_rl_v5_${TIMESTAMP}"
OUT_BASE="$ROOT/ckpt/${RUN_NAME}"
LOG_DIR="$ROOT/logs/v5_train"
mkdir -p "$LOG_DIR" "$OUT_BASE"

# 固定配置
SCORER_CKPT="$ROOT/ckpt/s3_token_scorer"              # SFT 初始化
AUTOVLA_CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
AUTOVLA_CONFIG="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
JSON_DIR="$ROOT/data/navtrain_nocot"
METRIC_CACHE="$ROOT/data/navtrain_metric_cache"
BASELINE_SCORES="$ROOT/results/baseline_sub_scores.json"

# v5 超参
EFFICIENCY_BETA=0.0         # v5: 不用 efficiency bonus
DRIVING_SCALE=3.0            # driving reward 缩放
SAFETY_BETA=0.03             # 温和 safety penalty
SAFETY_MARGIN=0.02
KL_BETA=0.01
BUDGET_KL_BETA=0.0
LR=3e-5
BUDGET_LR=1e-4
BUDGET_LOG_STD_INIT=-1.0
SELECTION_PG_WEIGHT=1.0
NUM_EPOCHS=2                 # v5: 2 epochs 给 true PDMS signal 更多时间
GROUP_SIZE=16
MIN_KR=0.2
MAX_KR=0.9
SEED=42

echo "============================================"
echo "  v5 True PDMS RL Training"
echo "  Run: $RUN_NAME"
echo "  Timestamp: $TIMESTAMP"
echo "============================================"
echo "  efficiency_beta = $EFFICIENCY_BETA (NO bonus)"
echo "  driving_scale   = $DRIVING_SCALE"
echo "  safety_beta     = $SAFETY_BETA"
echo "  epochs          = $NUM_EPOCHS"
echo "  reward          = True PDMS product (use_true_pdms=True)"
echo "============================================"

# 保存配置
cat > "$OUT_BASE/v5_config.json" << EOF
{
  "run_name": "$RUN_NAME",
  "timestamp": "$TIMESTAMP",
  "version": "v5",
  "description": "True PDMS RL — reward = real PDMS product, no efficiency bonus",
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
  "reward_type": "true_pdms_product",
  "efficiency_bonus": false
}
EOF

# 启动 4 shard 训练 (每个 GPU 一个独立进程)
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
      --min-keep-ratio $MIN_KR \
      --max-keep-ratio $MAX_KR \
      --budget-log-std-init $BUDGET_LOG_STD_INIT \
      --num-shards 4 \
      --shard-id $SH \
      --seed $SEED \
      --prune-variant attn_mask \
      --baseline-scores "$BASELINE_SCORES" \
      > "$LOGFILE" 2>&1
  ) &
  PIDS="$PIDS $!"
done

echo "[launch] Training PIDs: $PIDS"
echo "$PIDS" > "$LOG_DIR/train.pids"
echo "$OUT_BASE" > "$LOG_DIR/outbase"

echo "[launch] Logs: $LOG_DIR/"
echo "[launch] Ckpt:  $OUT_BASE"
echo "[launch] All 4 shards launched. Training in progress."

# 不等待 — 让 auto_chain 脚本来轮询
