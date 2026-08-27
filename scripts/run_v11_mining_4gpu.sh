#!/usr/bin/env bash
# ============================================================================
# run_v11_mining_4gpu.sh — v11 阶段 1：hard-example mining（4×H20）
#
# 背景（v10 r1 结论）：
#   full_hi 已追平 no-prune SOTA（full 0.898274 vs 0.898845，-0.000571），
#   但 90 个「灾难场景」（剪枝后 PDMS 崩到 0 而 no-prune 成功）贡献了
#   87.6% 的总负 delta。救回这批场景即可反超（+0.005 量级）。
#
# 本脚本用 v10 winner（full_hi）的确定性策略在 navtrain 上逐场景 rollout，
# 记录 (pruned_pdms, noprune_pdms, delta, keep_ratio)，供阶段 2 组装
# hard-enriched 训练场景表。
#
# 关键点：
#   - deterministic=True：keep_ratio 用 policy mean，与 eval 口径一致
#   - navtrain（19225 scenes）与 navtest 评估集完全隔离，无 train-on-test
#   - 结果 append 到 jsonl 且断点续跑（被回收后重跑同命令即继续）
#
# 用法：
#   bash scripts/run_v11_mining_4gpu.sh [n_scenes_total]
# ============================================================================
set -uo pipefail
ROOT=/apdcephfs/private_shayladeng/tokenrl_autoVLA
cd "$ROOT"
PY=/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"
NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh" >/dev/null
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

N_TOTAL=${1:-4096}
NUM_SHARDS=4
CYCLE_ID=20260826_v11
MINE_DIR="$ROOT/results/mining_${CYCLE_ID}"
LOG_DIR="$ROOT/logs/v7_surrogate_v11_${CYCLE_ID}"
mkdir -p "$MINE_DIR" "$LOG_DIR"

# v10 winner：追平 SOTA 的 full_hi final checkpoint
INIT_CKPT="$ROOT/ckpt/v7_surrogate_20260825_v10r1/full_hi"
SCORER="$ROOT/ckpt/s3_token_scorer"
VLA_CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
VLA_CFG="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
JSON_DIR="$ROOT/data/navtrain_nocot"
METRIC="$ROOT/data/navtrain_metric_cache"

# 策略超参必须与 full_hi 训练时一致，否则 keep_ratio 映射区间会错
POLICY="--min-keep-ratio 0.35 --max-keep-ratio 0.85 \
--selection-mode st_topk --selection-tau 0.1 --use-value-baseline \
--prune-variant attn_mask --driving-scale 3.0"

mine_shard() {
  local gpu=$1 shard=$2
  echo "[$(date)] mine shard$shard on gpu$gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_scorer_budget_rl.py \
    --scorer-ckpt "$SCORER" --init-budget-ckpt "$INIT_CKPT" \
    --out-dir "$MINE_DIR" --mine-mode \
    --mine-out "$MINE_DIR/mine_shard${shard}.jsonl" \
    --json-dir "$JSON_DIR" --metric-cache "$METRIC" \
    --sensor-data-path "$SENSOR" --autovla-config "$VLA_CFG" --autovla-ckpt "$VLA_CKPT" \
    --max-scenes "$N_TOTAL" --num-shards "$NUM_SHARDS" --shard-id "$shard" \
    $POLICY \
    > "$LOG_DIR/mine_shard${shard}.log" 2>&1
}

echo "[$(date)] v11 mining: $N_TOTAL navtrain scenes over $NUM_SHARDS GPUs"
pids=()
for s in 0 1 2 3; do
  mine_shard "$s" "$s" & pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$LOG_DIR/mining.pids"
for p in "${pids[@]}"; do wait "$p" || true; done

echo "[$(date)] mining done. records per shard:"
wc -l "$MINE_DIR"/mine_shard*.jsonl
echo "[$(date)] MINING_COMPLETE" | tee "$LOG_DIR/MINING_COMPLETE"
