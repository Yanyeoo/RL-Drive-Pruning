#!/usr/bin/env bash
# ============================================================================
# run_v7_surrogate_ablation_4gpu.sh — v7 可微 Top-K surrogate 对比（4×H20）
#
# 目标（ICLR 主线）：打开 token_net 梯度，用「有界、保序」的可微 Top-K
# surrogate 替换旧的「无界 softmax surrogate」，配合 delta reward，验证 RL 能否
# 首次真正突破 SFT 0.89199。
#
# 根因（上周期已实锤）：
#   - 旧 softmax surrogate 的 logsumexp 项让 token score 尺度爆炸漂移
#     （logp 从 -10 漂到 -63），破坏 SFT ranking。
#   - 更关键：旧 sweep 用 --selection-pg-weight 0.0，token_net 完全冻结，
#     只有 budget head 在随机游走 → RL 永远无法超过 SFT。
#
# 4 个 arm（各占 1 卡，512 scene 小规模）：
#   arm0 softmax_open : softmax surrogate + token 梯度打开（旧方法对照，预期漂移）
#   arm1 st_topk      : straight-through Top-K, tau=0.1（方法 A 主推）
#   arm2 gumbel       : gumbel-sigmoid 软 mask, tau=1.0（方法 A 备选）
#   arm3 st_topk_tau1 : straight-through Top-K, tau=1.0（tau 对比）
#
# 全部启用 --delta-reward（方向 C）。
#
# 用法：bash scripts/run_v7_surrogate_ablation_4gpu.sh
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

CYCLE_ID=${CYCLE_ID:-$(date +%Y%m%d_%H%M%S)}
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
SWEEP_SCENES=${SWEEP_SCENES:-512}
GROUP=${GROUP:-8}

# arm 名 / selection-mode / selection-tau / token_net lr / selection-pg-weight
NAMES=(softmax_open st_topk gumbel st_topk_tau1)
MODE=(softmax st_topk gumbel st_topk)
TAU=(1.0 0.1 1.0 1.0)
TOK_LR=(1e-5 3e-5 3e-5 3e-5)
PG_W=(1.0 1.0 1.0 1.0)

train_arm() {
  local gpu=$1 i=$2 out="$RUN_ROOT/${NAMES[$i]}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_scorer_budget_rl.py \
    --scorer-ckpt "$SCORER" --out-dir "$out" --json-dir "$JSON_DIR" \
    --metric-cache "$METRIC" --sensor-data-path "$SENSOR" \
    --autovla-config "$VLA_CFG" --autovla-ckpt "$VLA_CKPT" \
    --num-epochs 1 --max-scenes "$SWEEP_SCENES" --group-size "$GROUP" \
    --lr "${TOK_LR[$i]}" --budget-lr 1e-4 --kl-beta 0.01 --budget-kl-beta 0.0 \
    --selection-pg-weight "${PG_W[$i]}" --selection-mode "${MODE[$i]}" --selection-tau "${TAU[$i]}" \
    --efficiency-beta 0.005 --driving-scale 3.0 --delta-reward \
    --safety-beta 0.5 --safety-margin 0.0 --min-keep-ratio 0.35 --max-keep-ratio 0.70 \
    --budget-log-std-init -1 --seed 3407 --prune-variant attn_mask \
    --baseline-scores "$BASE" --counterfactual-k 0 --save-every 8 --log-every 1 \
    > "$CYCLE_DIR/train_${NAMES[$i]}.log" 2>&1
}

echo "[$(date)] v7 surrogate ablation: 4-arm token-gradient sweep (512 scenes, delta reward)"
pids=()
for i in {0..3}; do train_arm "$i" "$i" & pids+=("$!"); done
printf '%s\n' "${pids[@]}" > "$CYCLE_DIR/phase1.pids"
for p in "${pids[@]}"; do wait "$p" || true; done

echo "[$(date)] phase1 training done. Summarizing train logs:"
"$PY" - "$CYCLE_DIR" "$RUN_ROOT" <<'PY'
import glob,json,os,sys
out,run=sys.argv[1:]
print(f"{'arm':16s} {'steps':>6s} {'reward_mean':>12s} {'kr_mean':>8s} "
      f"{'sel_logp':>10s} {'grad_norm':>10s}")
for arm in ['softmax_open','st_topk','gumbel','st_topk_tau1']:
    p=os.path.join(run,arm,'train_log.jsonl')
    if not os.path.exists(p): print(f"{arm:16s}  NO LOG"); continue
    rows=[json.loads(l) for l in open(p) if l.strip()]
    if not rows: print(f"{arm:16s}  EMPTY"); continue
    r=rows[-1]
    print(f"{arm:16s} {len(rows):6d} {r.get('reward_mean',float('nan')):12.5f} "
          f"{r.get('keep_ratio_mean',float('nan')):8.4f} {r.get('selection_log_prob_mean',float('nan')):10.4f} "
          f"{r.get('grad_norm',float('nan')):10.4f}")
PY

echo "[$(date)] cycle complete" | tee "$CYCLE_DIR/CYCLE_COMPLETE"
