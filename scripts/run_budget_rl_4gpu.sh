#!/usr/bin/env bash
# run_budget_rl_4gpu.sh — Budget RL training (scorer learns WHAT + HOW MANY to prune)
# 4 GPU × 4 shard parallel
# Launch: nohup bash scripts/run_budget_rl_4gpu.sh > logs/budget_rl_train.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

# === Config ===
# Use RL-shaped scorer as init (if available), else fall back to SFT
RL_BEST=$(ls -d $ROOT/ckpt/s3_token_scorer_rl_shaped_*_sh0/ckpt_best 2>/dev/null | head -1)
if [[ -n "$RL_BEST" && -f "$RL_BEST/checkpoint.pt" ]]; then
    SCORER_CKPT="$RL_BEST"
    echo "[budget-rl] Init from RL best: $SCORER_CKPT"
else
    SCORER_CKPT="$ROOT/ckpt/s3_token_scorer"
    echo "[budget-rl] Init from SFT: $SCORER_CKPT"
fi

OUT_DIR="$ROOT/ckpt/s3_token_scorer_budget_rl_$(date +%Y%m%d_%H%M%S)"
BASELINE="$ROOT/results/baseline_sub_scores.json"
EFFICIENCY_BETA=0.05  # trade-off: how much to reward pruning more
NUM_EPOCHS=3
GROUP_SIZE=8
LR=3e-5
KL_BETA=0.01
SEED=42

# Pre-flight
if pgrep -f "train_scorer_budget_rl" >/dev/null; then
    echo "[ABORT] budget RL already running"; exit 1
fi

echo "=========================================="
echo "[budget-rl] Starting 4-shard parallel training"
echo "  init_ckpt:       $SCORER_CKPT"
echo "  out_dir:         $OUT_DIR"
echo "  efficiency_beta: $EFFICIENCY_BETA"
echo "  keep_ratio:      [0.2, 0.9] (learned)"
echo "=========================================="

PIDS=""
for SH in 0 1 2 3; do
    GPU=$SH
    SHARD_OUT="${OUT_DIR}_sh${SH}"
    echo "[budget-rl] GPU$GPU shard$SH -> $SHARD_OUT"
    (
        export CUDA_VISIBLE_DEVICES=$GPU
        $PY scripts/train_scorer_budget_rl.py \
            --scorer-ckpt "$SCORER_CKPT" \
            --out-dir "$SHARD_OUT" \
            --efficiency-beta $EFFICIENCY_BETA \
            --num-epochs $NUM_EPOCHS \
            --group-size $GROUP_SIZE \
            --lr $LR \
            --kl-beta $KL_BETA \
            --min-keep-ratio 0.2 \
            --max-keep-ratio 0.9 \
            --shaped-reward \
            --baseline-scores "$BASELINE" \
            --num-shards 4 \
            --shard-id $SH \
            --seed $((SEED + SH)) \
            --device cuda:0
    ) > "$ROOT/logs/budget_rl_sh${SH}.log" 2>&1 &
    PIDS="$PIDS $!"
done

echo "$PIDS" > "$ROOT/logs/budget_rl_train.pids"
echo "[budget-rl] PIDs: $PIDS"
echo "[budget-rl] Waiting..."
wait
echo "[budget-rl] ALL DONE at $(date)"

# Auto-chain: Budget RL eval after training
echo "[budget-rl] Chaining to budget RL eval..."
chmod +x "$ROOT/scripts/run_budget_rl_eval.sh"
nohup bash "$ROOT/scripts/run_budget_rl_eval.sh" > "$ROOT/logs/budget_rl_eval.log" 2>&1 &
echo "[budget-rl] Budget RL eval launched (PID=$!)"

