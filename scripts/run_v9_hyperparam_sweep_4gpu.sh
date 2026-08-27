#!/usr/bin/env bash
# ============================================================================
# run_v9_hyperparam_sweep_4gpu.sh — v9 超参 sweep：st_topk 单 knob 微调（4×H20）
#
# 背景：v8 规模消融已实锤「加数据量 512→2000/4000 有害」：
#   s2000 gate 0.8895、s4000 gate 0.8911，均 < SFT 0.89199（更差）。
#   → 数据多样性是毒药，512 scenes / 64 steps 是唯一甜点位。
#   因此 v9 回到 512 scenes 甜点位，只做「单 knob」微调，找冲 SOTA 的方向。
#
# v7 winner（基线，已 full 实锤）：
#   st_topk  tau=0.1 lr=3e-5 pg_w=1.0 kl=0.01  ep=1  group=8
#   full 4-shard mean 0.894798 / min-shard 0.892037（超 SFT +0.0028，差 SOTA 0.004）。
#
# Round1 4 arm（每个只改一个 knob，其余 = v7 winner）：
#   st_topk_ep2     : num-epochs 1→2（128 步，同 512 scenes，训练更充分，非加数据）
#   st_topk_tau005  : selection-tau 0.1→0.05（st_topk sigmoid 更锐利）
#   st_topk_pgw2    : selection-pg-weight 1.0→2.0（token 选择梯度更强）
#   st_topk_kl005   : kl-beta 0.01→0.05（token_net 更强锚定 SFT，防漂移）
#
# 用法：
#   bash scripts/run_v9_hyperparam_sweep_4gpu.sh [round_name] [gpu_offset]
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
CYCLE_ID="20260824_v9${ROUND}"
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

# arm 名 / epochs / tau / lr / pg_weight / kl_beta
NAMES=(st_topk_ep2 st_topk_tau005 st_topk_pgw2 st_topk_kl005)
EPOCHS=(2 1 1 1)
TAU=(0.1 0.05 0.1 0.1)
LR=(3e-5 3e-5 3e-5 3e-5)
PGW=(1.0 1.0 2.0 1.0)
KL=(0.01 0.01 0.01 0.05)

train_arm() {
  local gpu=$1 i=$2 out="$RUN_ROOT/${NAMES[$i]}"
  echo "[$(date)] launch ${NAMES[$i]} on gpu$gpu (ep=${EPOCHS[$i]} tau=${TAU[$i]} lr=${LR[$i]} pgw=${PGW[$i]} kl=${KL[$i]})"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_scorer_budget_rl.py \
    --scorer-ckpt "$SCORER" --out-dir "$out" --json-dir "$JSON_DIR" \
    --metric-cache "$METRIC" --sensor-data-path "$SENSOR" \
    --autovla-config "$VLA_CFG" --autovla-ckpt "$VLA_CKPT" \
    --num-epochs "${EPOCHS[$i]}" --max-scenes 512 --group-size 8 \
    --lr "${LR[$i]}" --budget-lr 1e-4 --kl-beta "${KL[$i]}" --budget-kl-beta 0.0 \
    --selection-pg-weight "${PGW[$i]}" --selection-mode st_topk --selection-tau "${TAU[$i]}" \
    --efficiency-beta 0.005 --driving-scale 3.0 --delta-reward \
    --safety-beta 0.5 --safety-margin 0.0 --min-keep-ratio 0.35 --max-keep-ratio 0.70 \
    --budget-log-std-init -1 --seed 3407 --prune-variant attn_mask \
    --baseline-scores "$BASE" --counterfactual-k 0 --save-every 8 --log-every 1 \
    > "$CYCLE_DIR/train_${NAMES[$i]}.log" 2>&1
}

echo "[$(date)] v9${ROUND} hyperparam sweep: st_topk single-knob (512 scenes), 4-arm, offset=$OFFSET"
pids=()
for i in 0 1 2 3; do train_arm "$((OFFSET+i))" "$i" & pids+=("$!"); done
printf '%s\n' "${pids[@]}" > "$CYCLE_DIR/phase1.pids"
for p in "${pids[@]}"; do wait "$p" || true; done

echo "[$(date)] phase1 training done. Summarizing train logs:"
"$PY" - "$CYCLE_DIR" "$RUN_ROOT" <<'PY'
import json, os, sys
out, run = sys.argv[1:]
print(f"{'arm':20s} {'steps':>6s} {'reward_mean':>12s} {'kr_mean':>8s} "
      f"{'sel_logp':>10s} {'grad_norm':>10s}")
for arm in ['st_topk_ep2','st_topk_tau005','st_topk_pgw2','st_topk_kl005']:
    p = os.path.join(run, arm, 'train_log.jsonl')
    if not os.path.exists(p):
        print(f"{arm:20s}  NO LOG"); continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        print(f"{arm:20s}  EMPTY"); continue
    r = rows[-1]
    print(f"{arm:20s} {len(rows):6d} {r.get('reward_mean', float('nan')):12.5f} "
          f"{r.get('keep_ratio_mean', float('nan')):8.4f} {r.get('selection_log_prob_mean', float('nan')):10.4f} "
          f"{r.get('grad_norm', float('nan')):10.4f}")
PY

echo "[$(date)] cycle complete" | tee "$CYCLE_DIR/CYCLE_COMPLETE"
