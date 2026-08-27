#!/usr/bin/env bash
# ============================================================================
# run_v11_hardmine_4gpu.sh — v11 阶段 3：hard-example mining 训练（4×H20）
#
# 起点：v10 winner full_hi（navtest full 0.898274，距 no-prune 0.898845 仅 -0.000571）
# 目标：救回灾难场景以反超 no-prune SOTA，同时守住 efficiency（kr ≈ 0.72）。
#
# 全部臂都从 full_hi 续训（--init-budget-ckpt），沿用其方法配置
# （value baseline + efficiency floor + max_kr 0.85），只改训练分布/强度：
#
#   hard50     : 50% 灾难场景 + 50% 正常场景（主力配置，hard 过采样 x2）
#   hard50_lr  : 同上，但 budget_lr 加倍 —— 灾难场景需要 budget head 快速上抬 kr
#   hard75     : 75% 灾难场景（更激进，检验 hard 比例的边际收益）
#   hard50_kl  : 同 hard50，但 token_net KL 收紧到 0.02（防止 token 排序被少量
#                灾难场景带偏，保住正常场景表现）
#
# 用法：
#   bash scripts/run_v11_hardmine_4gpu.sh <scene_list_hard50> <scene_list_hard75> [round]
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

LIST50=${1:?usage: run_v11_hardmine_4gpu.sh <list_hard50> <list_hard75> [round]}
LIST75=${2:?need hard75 scene list}
ROUND=${3:-r1}
CYCLE_ID="20260826_v11${ROUND}"
CYCLE_DIR="$ROOT/logs/v7_surrogate_${CYCLE_ID}"
RUN_ROOT="$ROOT/ckpt/v7_surrogate_${CYCLE_ID}"
mkdir -p "$CYCLE_DIR" "$RUN_ROOT" "$ROOT/results/raw"
echo "$CYCLE_ID" > "$ROOT/logs/v7_surrogate_latest"

INIT_CKPT="$ROOT/ckpt/v7_surrogate_20260825_v10r1/full_hi"
SCORER="$ROOT/ckpt/s3_token_scorer"
VLA_CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
VLA_CFG="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
JSON_DIR="$ROOT/data/navtrain_nocot"
METRIC="$ROOT/data/navtrain_metric_cache"
BASE="$ROOT/results/baseline_sub_scores.json"

# 公共参数 = v10 full_hi 方法配置（value baseline + floor + max_kr 0.85），
# 唯一区别：warm-start 自 full_hi，训练分布由 --scene-list 指定。
COMMON="--scorer-ckpt $SCORER --init-budget-ckpt $INIT_CKPT \
--json-dir $JSON_DIR --metric-cache $METRIC \
--sensor-data-path $SENSOR --autovla-config $VLA_CFG --autovla-ckpt $VLA_CKPT \
--num-epochs 2 --group-size 8 \
--lr 3e-5 --selection-pg-weight 1.0 --selection-mode st_topk --selection-tau 0.1 \
--driving-scale 3.0 --delta-reward \
--safety-beta 0.5 --safety-margin 0.0 \
--min-keep-ratio 0.35 --max-keep-ratio 0.85 \
--use-value-baseline --value-lr 1e-4 --value-loss-weight 1.0 \
--efficiency-mode floor --target-keep-ratio 0.65 --efficiency-beta 0.02 \
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

HARD50="--scene-list $LIST50 --budget-lr 1e-4 --kl-beta 0.01"
HARD50_LR="--scene-list $LIST50 --budget-lr 2e-4 --kl-beta 0.01"
HARD75="--scene-list $LIST75 --budget-lr 1e-4 --kl-beta 0.01"
HARD50_KL="--scene-list $LIST50 --budget-lr 1e-4 --kl-beta 0.02"

echo "[$(date)] v11${ROUND} hard-example mining, warm-start from v10 full_hi"
pids=()
train_arm 0 hard50     "$HARD50"    & pids+=("$!")
train_arm 1 hard50_lr  "$HARD50_LR" & pids+=("$!")
train_arm 2 hard75     "$HARD75"    & pids+=("$!")
train_arm 3 hard50_kl  "$HARD50_KL" & pids+=("$!")
printf '%s\n' "${pids[@]}" > "$CYCLE_DIR/phase1.pids"
for p in "${pids[@]}"; do wait "$p" || true; done

echo "[$(date)] v11 training done. Summary:"
"$PY" - "$RUN_ROOT" <<'PY'
import json, os, sys
run = sys.argv[1]
print(f"{'arm':14s} {'steps':>6s} {'reward_mean':>12s} {'kr_mean':>8s} "
      f"{'sel_logp':>10s} {'grad_norm':>10s}")
for arm in ['hard50','hard50_lr','hard75','hard50_kl']:
    p = os.path.join(run, arm, 'train_log.jsonl')
    if not os.path.exists(p):
        print(f"{arm:14s}  NO LOG"); continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        print(f"{arm:14s}  EMPTY"); continue
    r = rows[-1]
    kr = [x.get('keep_ratio_mean', 0) for x in rows[-16:]]
    print(f"{arm:14s} {len(rows):6d} {r.get('reward_mean', float('nan')):12.5f} "
          f"{sum(kr)/max(1,len(kr)):8.4f} "
          f"{r.get('selection_log_prob_mean', float('nan')):10.4f} "
          f"{r.get('grad_norm', float('nan')):10.4f}")
PY

echo "[$(date)] cycle complete" | tee "$CYCLE_DIR/CYCLE_COMPLETE"
