#!/usr/bin/env bash
# run_7b_budget_rl_train.sh — Budget RL on 7B (base Qwen2.5-VL-7B features), resumable.
# Mirrors run_budget_rl_navtrain.sh but for 7B: warm-starts from the 7B SFT scorer
# (ckpt/s3_token_scorer_7b) and trains on base-7B features (matching s3_token_scorer_7b).
#
# This completes contribution (3) (3B->7B method generalization) for the FULL method:
# SFT scorer (already done) + Budget RL, retrained on 7B, then evaluated on ImpromptuVLA_7B.
#
# Self-guarded: runs a 1-shard SMOKE test (small max-scenes) first; only launches the
# full 8-shard run if the smoke test loads 7B and produces a training step. Prevents
# burning ~20h on a broken config.
#
# Usage:
#   BUDGET_EPOCHS=3 bash scripts/run_7b_budget_rl_train.sh
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$ROOT/code/third_party/AutoVLA:$ROOT/code/third_party/AutoVLA/navsim:${PYTHONPATH:-}"

# === Tunables ===
NUM_EPOCHS=${BUDGET_EPOCHS:-3}
GROUP_SIZE=${BUDGET_GROUP:-16}
EFFICIENCY_BETA=${BUDGET_EFF:-0.15}
DRIVING_SCALE=${BUDGET_DRV:-2.0}
BUDGET_LR=${BUDGET_LR:-1e-4}
LR=${BUDGET_TOKLR:-3e-5}
KL_BETA=${BUDGET_KL:-0.01}
PRUNE_VARIANT=${BUDGET_PRUNE:-attn_mask}   # attn_mask (train proxy) ; eval uses drop

# === 7B-specific config ===
CONFIG_7B="$ROOT/code/third_party/AutoVLA/config/training/qwen2.5-vl-7B-navtest-grpo-nocot.yaml"
AUTOVLA_CKPT_7B=""   # empty => load base Qwen2.5-VL-7B (HF dir); matches s3_token_scorer_7b training
SCORER_CKPT="$ROOT/ckpt/s3_token_scorer_7b"   # warm-start from 7B SFT scorer
echo "[7b-budget-rl] warm-start from 7B SFT scorer: $SCORER_CKPT"
if [[ ! -f "$SCORER_CKPT/checkpoint.pt" ]]; then
    echo "[7b-budget-rl] FATAL: 7B SFT scorer not found at $SCORER_CKPT"; exit 2
fi
if [[ ! -f "$CONFIG_7B" ]]; then
    echo "[7b-budget-rl] FATAL: 7B config not found at $CONFIG_7B"; exit 2
fi

OUT_DIR="$ROOT/ckpt/s3_token_scorer_budget_rl_7b_$(date +%Y%m%d_%H%M%S)"
echo "$OUT_DIR" > "$ROOT/logs/budget_rl_7b_outdir.txt"
BASELINE="$ROOT/results/baseline_sub_scores.json"

echo "=========================================="
echo "[7b-budget-rl] 7B navtrain | epochs=$NUM_EPOCHS group=$GROUP_SIZE"
echo "  eff_beta=$EFFICIENCY_BETA drive_scale=$DRIVING_SCALE budget_lr=$BUDGET_LR kl=$KL_BETA"
echo "  autovla_config=$CONFIG_7B  autovla_ckpt='$AUTOVLA_CKPT_7B' (base 7B)"
echo "  out_dir: $OUT_DIR"
echo "=========================================="

# ---------- SMOKE TEST (1 shard, tiny) ----------
echo "[7b-budget-rl] SMOKE: shard0, max-scenes=200, 1 epoch..."
SMOKE_OUT="${OUT_DIR}_sh0_smoke"
rm -rf "$SMOKE_OUT"
(
    export CUDA_VISIBLE_DEVICES=0
    $PY scripts/train_scorer_budget_rl.py \
        --scorer-ckpt "$SCORER_CKPT" \
        --out-dir "$SMOKE_OUT" \
        --json-dir "$ROOT/data/navtrain_nocot" \
        --metric-cache "$ROOT/data/navtrain_metric_cache" \
        --autovla-config "$CONFIG_7B" \
        --autovla-ckpt "$AUTOVLA_CKPT_7B" \
        --efficiency-beta $EFFICIENCY_BETA \
        --driving-scale $DRIVING_SCALE \
        --num-epochs 1 \
        --group-size $GROUP_SIZE \
        --lr $LR --budget-lr $BUDGET_LR --kl-beta $KL_BETA \
        --min-keep-ratio 0.2 --max-keep-ratio 0.9 \
        --shaped-reward --baseline-scores "$BASELINE" \
        --num-shards 8 --shard-id 0 \
        --max-scenes 200 --seed 42 \
        --prune-variant $PRUNE_VARIANT --device cuda:0
) > "$ROOT/logs/budget_rl_7b_smoke.log" 2>&1
SMOKE_RC=$?
if [[ $SMOKE_RC -ne 0 ]] || grep -qE "Traceback|RuntimeError|CUDA out of memory|KeyError" "$ROOT/logs/budget_rl_7b_smoke.log"; then
    echo "[7b-budget-rl] SMOKE FAILED (rc=$SMOKE_RC). Last 25 lines:"
    tail -25 "$ROOT/logs/budget_rl_7b_smoke.log"
    echo "[7b-budget-rl] ABORT full launch. Fix config and re-run."
    exit 3
fi
echo "[7b-budget-rl] SMOKE OK (7B loaded, training step produced). Launching full 8-shard run."

# ---------- FULL 8-SHARD RUN ----------
PIDS=""
for SH in 0 1 2 3 4 5 6 7; do
    GPU=$SH
    SHARD_OUT="${OUT_DIR}_sh${SH}"
    echo "[7b-budget-rl] GPU$GPU shard$SH -> $SHARD_OUT"
    (
        export CUDA_VISIBLE_DEVICES=$GPU
        $PY scripts/train_scorer_budget_rl.py \
            --scorer-ckpt "$SCORER_CKPT" \
            --out-dir "$SHARD_OUT" \
            --json-dir "$ROOT/data/navtrain_nocot" \
            --metric-cache "$ROOT/data/navtrain_metric_cache" \
            --autovla-config "$CONFIG_7B" \
            --autovla-ckpt "$AUTOVLA_CKPT_7B" \
            --efficiency-beta $EFFICIENCY_BETA \
            --driving-scale $DRIVING_SCALE \
            --num-epochs $NUM_EPOCHS \
            --group-size $GROUP_SIZE \
            --lr $LR --budget-lr $BUDGET_LR --kl-beta $KL_BETA \
            --min-keep-ratio 0.2 --max-keep-ratio 0.9 \
            --shaped-reward --baseline-scores "$BASELINE" \
            --num-shards 8 --shard-id $SH \
            --seed $((42 + SH)) \
            --prune-variant $PRUNE_VARIANT --device cuda:0
    ) > "$ROOT/logs/budget_rl_7b_sh${SH}.log" 2>&1 &
    PIDS="$PIDS $!"
done
echo "$PIDS" > "$ROOT/logs/budget_rl_7b_train.pids"
echo "[7b-budget-rl] PIDs: $PIDS"
echo "[7b-budget-rl] Waiting (~20h expected)..."
wait
echo "[7b-budget-rl] ALL DONE at $(date)"
echo "[7b-budget-rl] 7B budget scorer shards: ${OUT_DIR}_sh{0..7}/checkpoint.pt"
