#!/usr/bin/env bash
# ============================================================================
# run_v10_methodA_4gpu.sh — 方法 A：value baseline + keep_ratio 上浮（4×H20）
#
# 目标（用户 2026-08-25 指令）：
#   允许 budget head 输出更高 keep_ratio（0.6~0.7）逼近 SOTA，最终目标 = 比不剪枝
#   （no-prune 0.8988）还高。
#
# 诊断依据（scripts/diagnose_v10_sota_headroom.py）：
#   - SFT r0.75 = 0.898353（距 no-prune 仅 -0.0005，delta std 0.091，是稳定免费午餐）
#   - RL v7 (kr=0.54) = 0.894789，损失集中在 shard1(-0.0065)/shard3(-0.0072)
#   - 173 个灾难 scene（delta<-0.3）贡献 89.4% 总损失（hard-mining 铁证）
#   - 906/11572 scene（7.8%）剪枝后反超 no-prune ——「比不剪枝还高」是可达的
#
# 方法 A 三件套（本轮实现）：
#   1. Value baseline (critic)：advantage = reward - V(s)，替代 group 标准化，降方差
#   2. keep_ratio 上浮：efficiency floor（只罚 kr<target，kr>=target 自由上浮）
#      + budget head 初始化锚定到 0.65~0.70
#   3. hard-example mining：本轮先做诊断（脚本已产出），mining 采样留 v11
#
# 2×2 消融（基础 = v9 winner st_topk_ep2）：
#   value_only  : + value baseline（max_kr 仍 0.70，隔离 value 纯贡献）
#   floor_init  : + floor efficiency + budget_init 0.65 + max_kr 0.80（隔离 kr 上浮）
#   full        : value + floor + init（完整方法 A）
#   full_hi     : full + budget_init 0.70 + max_kr 0.85 + target 0.65（更激进冲 SOTA）
#
# 用法：
#   bash scripts/run_v10_methodA_4gpu.sh [round_name] [gpu_offset]
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

ROUND=${1:-r1}
OFFSET=${2:-0}
CYCLE_ID="20260825_v10${ROUND}"
CYCLE_DIR="$ROOT/logs/v7_surrogate_${CYCLE_ID}"
RUN_ROOT="$ROOT/ckpt/v7_surrogate_${CYCLE_ID}"
mkdir -p "$CYCLE_DIR" "$RUN_ROOT" "$ROOT/results/raw"
echo "$CYCLE_ID" > "$ROOT/logs/v7_surrogate_latest"

SCORER="$ROOT/ckpt/s3_token_scorer"
VLA_CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
VLA_CFG="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
JSON_DIR="$ROOT/data/navtrain_nocot"
METRIC="$ROOT/data/navtrain_metric_cache"
BASE="$ROOT/results/baseline_sub_scores.json"

# 公共参数（= v9 winner st_topk_ep2）
COMMON="--scorer-ckpt $SCORER --json-dir $JSON_DIR --metric-cache $METRIC \
--sensor-data-path $SENSOR --autovla-config $VLA_CFG --autovla-ckpt $VLA_CKPT \
--num-epochs 2 --max-scenes 512 --group-size 8 \
--lr 3e-5 --budget-lr 1e-4 --kl-beta 0.01 --budget-kl-beta 0.0 \
--selection-pg-weight 1.0 --selection-mode st_topk --selection-tau 0.1 \
--driving-scale 3.0 --delta-reward \
--safety-beta 0.5 --safety-margin 0.0 --min-keep-ratio 0.35 \
--budget-log-std-init -1 --seed 3407 --prune-variant attn_mask \
--baseline-scores $BASE --counterfactual-k 0 --save-every 8 --log-every 1"

train_arm() {
  local gpu=$1 name=$2 extra="$3"
  local out="$RUN_ROOT/$name"
  echo "[$(date)] launch $name on gpu$gpu: $extra"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_scorer_budget_rl.py \
    $COMMON --out-dir "$out" $extra \
    > "$CYCLE_DIR/train_${name}.log" 2>&1
}

# value_only：value baseline，其余完全 = v9 winner（max_kr 0.70, linear eff 0.005）
VALUE_ONLY="--use-value-baseline --value-lr 1e-4 --value-loss-weight 1.0 \
--efficiency-beta 0.005 --efficiency-mode linear --max-keep-ratio 0.70"

# floor_init：kr 上浮（floor + init 0.65 + max_kr 0.80），无 value
FLOOR_INIT="--efficiency-mode floor --target-keep-ratio 0.60 --efficiency-beta 0.02 \
--budget-init-kr 0.65 --max-keep-ratio 0.80"

# full：value + floor + init
FULL="--use-value-baseline --value-lr 1e-4 --value-loss-weight 1.0 \
--efficiency-mode floor --target-keep-ratio 0.60 --efficiency-beta 0.02 \
--budget-init-kr 0.65 --max-keep-ratio 0.80"

# full_hi：更激进（init 0.70, max_kr 0.85, target 0.65）
FULL_HI="--use-value-baseline --value-lr 1e-4 --value-loss-weight 1.0 \
--efficiency-mode floor --target-keep-ratio 0.65 --efficiency-beta 0.02 \
--budget-init-kr 0.70 --max-keep-ratio 0.85"

echo "[$(date)] v10${ROUND} method A: value-baseline x keep_ratio-uprising (512 scenes, ep2), offset=$OFFSET"
pids=()
train_arm "$((OFFSET+0))" value_only "$VALUE_ONLY" & pids+=("$!")
train_arm "$((OFFSET+1))" floor_init "$FLOOR_INIT" & pids+=("$!")
train_arm "$((OFFSET+2))" full       "$FULL"       & pids+=("$!")
train_arm "$((OFFSET+3))" full_hi    "$FULL_HI"    & pids+=("$!")
printf '%s\n' "${pids[@]}" > "$CYCLE_DIR/phase1.pids"
for p in "${pids[@]}"; do wait "$p" || true; done

echo "[$(date)] phase1 training done. Summarizing train logs:"
"$PY" - "$CYCLE_DIR" "$RUN_ROOT" <<'PY'
import json, os, sys
out, run = sys.argv[1:]
print(f"{'arm':14s} {'steps':>6s} {'reward_mean':>12s} {'kr_mean':>8s} "
      f"{'sel_logp':>10s} {'grad_norm':>10s}")
for arm in ['value_only','floor_init','full','full_hi']:
    p = os.path.join(run, arm, 'train_log.jsonl')
    if not os.path.exists(p):
        print(f"{arm:14s}  NO LOG"); continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        print(f"{arm:14s}  EMPTY"); continue
    r = rows[-1]
    print(f"{arm:14s} {len(rows):6d} {r.get('reward_mean', float('nan')):12.5f} "
          f"{r.get('keep_ratio_mean', float('nan')):8.4f} "
          f"{r.get('selection_log_prob_mean', float('nan')):10.4f} "
          f"{r.get('grad_norm', float('nan')):10.4f}")
PY

echo "[$(date)] cycle complete" | tee "$CYCLE_DIR/CYCLE_COMPLETE"
