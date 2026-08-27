#!/usr/bin/env bash
# ============================================================================
# run_v8_scale_ablation_4gpu.sh — v8 规模消融：数据量 × 步数（4×H20）
#
# 结论（v7 已实锤）：winner = st_topk（tau=0.1, lr=3e-5, pg_w=1.0），
#   full 4-shard mean 0.894798 / min-shard 0.892037。
#   超 SFT(+0.0028) ✅，差 SOTA(0.89879) ~0.004 ❌。
#
# 瓶颈分析：v7 只有 64 梯度步（512 scenes ÷ group 8 × 1 epoch），且数据仅用
#   sorted 前 512/19225（2.7%）。min-shard 0.892037 几乎等于 SFT，说明 shard2
#   覆盖场景的排序没学好 → 训练不足 + 数据多样性不足，是规模问题而非结构问题。
#
# 本周期 4 个 arm（各占 1 卡，全部保持 st_topk 超参不变，只动数据量/步数）：
#   A  st_topk_s512      : 512 scenes,  1 epoch, group 8  → 64 步（复现 v7 基线）
#   B  st_topk_s2000     : 2000 scenes, 1 epoch, group 8  → 250 步
#   C  st_topk_s4000     : 4000 scenes, 1 epoch, group 8  → 500 步
#   D  st_topk_s4000_e3  : 4000 scenes, 3 epoch, group 16 → 750 步（冲 SOTA 主力）
#
# 预期训练时长：A≈38min, B≈2.5h, C≈5h, D≈15h（按 4.45 s/scene 计）。
# 训练完成后用 eval_v7_folds_4gpu.sh <CYCLE_ID> gate <arms...> 做 gate 筛选，
# 胜者再 full 定论。
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

CYCLE_ID=${CYCLE_ID:-20260819_scale}
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

# arm 名 / scenes / epochs / group-size
NAMES=(st_topk_s512 st_topk_s2000 st_topk_s4000 st_topk_s4000_e3)
SCENES=(512 2000 4000 4000)
EPOCHS=(1 1 1 3)
GRP=(8 8 8 16)

train_arm() {
  local gpu=$1 i=$2 out="$RUN_ROOT/${NAMES[$i]}"
  echo "[$(date)] launch ${NAMES[$i]} on gpu$gpu (scenes=${SCENES[$i]} epochs=${EPOCHS[$i]} group=${GRP[$i]})"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_scorer_budget_rl.py \
    --scorer-ckpt "$SCORER" --out-dir "$out" --json-dir "$JSON_DIR" \
    --metric-cache "$METRIC" --sensor-data-path "$SENSOR" \
    --autovla-config "$VLA_CFG" --autovla-ckpt "$VLA_CKPT" \
    --num-epochs "${EPOCHS[$i]}" --max-scenes "${SCENES[$i]}" --group-size "${GRP[$i]}" \
    --lr 3e-5 --budget-lr 1e-4 --kl-beta 0.01 --budget-kl-beta 0.0 \
    --selection-pg-weight 1.0 --selection-mode st_topk --selection-tau 0.1 \
    --efficiency-beta 0.005 --driving-scale 3.0 --delta-reward \
    --safety-beta 0.5 --safety-margin 0.0 --min-keep-ratio 0.35 --max-keep-ratio 0.70 \
    --budget-log-std-init -1 --seed 3407 --prune-variant attn_mask \
    --baseline-scores "$BASE" --counterfactual-k 0 --save-every 50 --log-every 1 \
    > "$CYCLE_DIR/train_${NAMES[$i]}.log" 2>&1
}

echo "[$(date)] v8 scale ablation: data × steps (st_topk fixed), 4-arm"
pids=()
for i in 0 1 2 3; do train_arm "$i" "$i" & pids+=("$!"); done
printf '%s\n' "${pids[@]}" > "$CYCLE_DIR/phase1.pids"
for p in "${pids[@]}"; do wait "$p" || true; done

echo "[$(date)] phase1 training done. Summarizing train logs:"
"$PY" - "$CYCLE_DIR" "$RUN_ROOT" <<'PY'
import json, os, sys
out, run = sys.argv[1:]
print(f"{'arm':18s} {'steps':>6s} {'reward_mean':>12s} {'kr_mean':>8s} "
      f"{'sel_logp':>10s} {'grad_norm':>10s}")
for arm in ['st_topk_s512','st_topk_s2000','st_topk_s4000','st_topk_s4000_e3']:
    p = os.path.join(run, arm, 'train_log.jsonl')
    if not os.path.exists(p):
        print(f"{arm:18s}  NO LOG"); continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        print(f"{arm:18s}  EMPTY"); continue
    r = rows[-1]
    print(f"{arm:18s} {len(rows):6d} {r.get('reward_mean', float('nan')):12.5f} "
          f"{r.get('keep_ratio_mean', float('nan')):8.4f} {r.get('selection_log_prob_mean', float('nan')):10.4f} "
          f"{r.get('grad_norm', float('nan')):10.4f}")
PY

echo "[$(date)] cycle complete" | tee "$CYCLE_DIR/CYCLE_COMPLETE"
