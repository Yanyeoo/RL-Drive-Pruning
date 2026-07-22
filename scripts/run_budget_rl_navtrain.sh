#!/usr/bin/env bash
# run_budget_rl_navtrain.sh — Budget RL on NAVTRAIN (clean split), resumable.
# 4 GPU × 4 shard parallel; auto-chains to dynamic-budget eval.
#
# Usage:
#   BUDGET_EPOCHS=1 bash scripts/run_budget_rl_navtrain.sh      # validation (~5-6h)
#   BUDGET_EPOCHS=3 bash scripts/run_budget_rl_navtrain.sh      # full    (~16-17h)
#
# If reclaimed, just re-run the SAME command — it resumes from ckpt_resume/.
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

# === Tunables (override via env) ===
NUM_EPOCHS=${BUDGET_EPOCHS:-3}
GROUP_SIZE=${BUDGET_GROUP:-16}
EFFICIENCY_BETA=${BUDGET_EFF:-0.15}
DRIVING_SCALE=${BUDGET_DRV:-2.0}
BUDGET_LR=${BUDGET_LR:-1e-4}
LR=${BUDGET_TOKLR:-3e-5}
KL_BETA=${BUDGET_KL:-0.01}
PRUNE_VARIANT=${BUDGET_PRUNE:-attn_mask}   # attn_mask (train proxy) ; eval uses drop

# === Config ===
# NEW STORY (2026-07-22): SFT -> Budget RL (RL SCORER deleted as a method branch).
# Budget RL warm-starts DIRECTLY from the clean SFT scorer (navtrain-trained,
# LambdaRank-distilled from L12 attention), NOT from rl_shaped. rl_shaped is the
# deleted intermediate branch and may be navtest-polluted, so using it would (a) break
# the paper claim "Budget RL warm-starts from SFT" and (b) leak navtest into Budget RL.
# FORCE SFT — do NOT prefer rl_shaped.
SCORER_CKPT="$ROOT/ckpt/s3_token_scorer"
echo "[budget-rl] Init from SFT scorer (clean, navtrain): $SCORER_CKPT"

OUT_DIR="$ROOT/ckpt/s3_token_scorer_budget_rl_$(date +%Y%m%d_%H%M%S)"
# Resume: if the previous run left a ckpt_resume, reuse its out_dir and continue.
PREV=$(cat "$ROOT/logs/budget_rl_outdir.txt" 2>/dev/null)
if [[ -n "$PREV" && -d "${PREV}_sh0/ckpt_resume" ]]; then
    OUT_DIR="$PREV"
    echo "[budget-rl] RESUME into existing out_dir: $OUT_DIR"
else
    echo "$OUT_DIR" > "$ROOT/logs/budget_rl_outdir.txt"
fi
BASELINE="$ROOT/results/baseline_sub_scores.json"

echo "=========================================="
echo "[budget-rl] navtrain | epochs=$NUM_EPOCHS group=$GROUP_SIZE"
echo "  eff_beta=$EFFICIENCY_BETA drive_scale=$DRIVING_SCALE budget_lr=$BUDGET_LR kl=$KL_BETA"
echo "  out_dir: $OUT_DIR"
echo "=========================================="

PIDS=""
# 8-shard parallel across all 8 GPUs (GPU0-7). Each shard trains on 1/8 of navtrain;
# eval uses shard0's ckpt (run_budget_rl_eval.sh picks *_sh0/ckpt_best).
for SH in 0 1 2 3 4 5 6 7; do
    GPU=$SH
    SHARD_OUT="${OUT_DIR}_sh${SH}"
    echo "[budget-rl] GPU$GPU shard$SH -> $SHARD_OUT"
    (
        export CUDA_VISIBLE_DEVICES=$GPU
        $PY scripts/train_scorer_budget_rl.py \
            --scorer-ckpt "$SCORER_CKPT" \
            --out-dir "$SHARD_OUT" \
            --json-dir "$ROOT/data/navtrain_nocot" \
            --metric-cache "$ROOT/data/navtrain_metric_cache" \
            --efficiency-beta $EFFICIENCY_BETA \
            --driving-scale $DRIVING_SCALE \
            --num-epochs $NUM_EPOCHS \
            --group-size $GROUP_SIZE \
            --lr $LR \
            --budget-lr $BUDGET_LR \
            --kl-beta $KL_BETA \
            --min-keep-ratio 0.2 \
            --max-keep-ratio 0.9 \
            --shaped-reward \
            --baseline-scores "$BASELINE" \
            --num-shards 8 \
            --shard-id $SH \
            --seed $((42 + SH)) \
            --prune-variant $PRUNE_VARIANT \
            --device cuda:0
    ) > "$ROOT/logs/budget_rl_sh${SH}.log" 2>&1 &
    PIDS="$PIDS $!"
done

echo "$PIDS" > "$ROOT/logs/budget_rl_train.pids"
echo "[budget-rl] PIDs: $PIDS"
echo "[budget-rl] Waiting..."
wait
echo "[budget-rl] ALL DONE at $(date)"

echo "[budget-rl] Chaining to budget RL eval (dynamic budget)..."
chmod +x "$ROOT/scripts/run_budget_rl_eval.sh"
nohup bash "$ROOT/scripts/run_budget_rl_eval.sh" > "$ROOT/logs/budget_rl_eval.log" 2>&1 &
echo "[budget-rl] Budget RL eval launched (PID=$!)"
