#!/usr/bin/env bash
# run_rl_shaped_4gpu.sh — RL shaped reward 训练 (4 GPU × 4 shard 并行)
# 每个 GPU 跑一个 shard，共 4 个 shard 并行
# Launch: nohup bash scripts/run_rl_shaped_4gpu.sh > logs/rl_shaped_train.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

# === Config ===
SCORER_CKPT="$ROOT/ckpt/s3_token_scorer"       # LambdaRank SFT ckpt (init)
OUT_DIR="$ROOT/ckpt/s3_token_scorer_rl_shaped_$(date +%Y%m%d_%H%M%S)"
BASELINE="$ROOT/results/baseline_sub_scores.json"
KEEP_RATIO=0.5
NUM_EPOCHS=3
GROUP_SIZE=8
LR=3e-5
KL_BETA=0.01
SEED=42

# === Pre-flight ===
if pgrep -f "train_scorer_grpo" >/dev/null; then
    echo "[ABORT] train_scorer_grpo already running. Exiting."
    exit 1
fi
[[ -f "$SCORER_CKPT/checkpoint.pt" ]] || { echo "FATAL: scorer ckpt not found at $SCORER_CKPT"; exit 2; }
[[ -f "$BASELINE" ]] || { echo "FATAL: baseline sub-scores not found at $BASELINE"; exit 2; }

echo "=========================================="
echo "[RL-shaped] Starting 4-shard parallel training"
echo "  scorer_ckpt: $SCORER_CKPT"
echo "  out_dir:     $OUT_DIR"
echo "  baseline:    $BASELINE"
echo "  keep_ratio:  $KEEP_RATIO"
echo "  lr:          $LR"
echo "  epochs:      $NUM_EPOCHS"
echo "  group_size:  $GROUP_SIZE"
echo "=========================================="

# Backup scorer ckpt before RL
cp -a "$SCORER_CKPT" "${SCORER_CKPT}_backup_before_rl_$(date +%Y%m%d)" 2>/dev/null || true

PIDS=""
for SH in 0 1 2 3; do
    GPU=$SH
    SHARD_OUT="${OUT_DIR}_sh${SH}"
    echo "[RL-shaped] GPU$GPU shard$SH -> $SHARD_OUT"
    (
        export CUDA_VISIBLE_DEVICES=$GPU
        $PY scripts/train_scorer_grpo.py \
            --scorer-ckpt "$SCORER_CKPT" \
            --out-dir "$SHARD_OUT" \
            --keep-ratio $KEEP_RATIO \
            --num-epochs $NUM_EPOCHS \
            --group-size $GROUP_SIZE \
            --lr $LR \
            --kl-beta $KL_BETA \
            --shaped-reward \
            --baseline-scores "$BASELINE" \
            --num-shards 4 \
            --shard-id $SH \
            --seed $((SEED + SH)) \
            --device cuda:0
    ) > "$ROOT/logs/rl_shaped_sh${SH}.log" 2>&1 &
    PIDS="$PIDS $!"
done

# Save PIDs
echo "$PIDS" > "$ROOT/logs/rl_shaped_train.pids"
echo "[RL-shaped] PIDs: $PIDS"
echo "[RL-shaped] Waiting for all shards..."
wait
echo "[RL-shaped] ALL DONE at $(date)"

# Pick best shard
echo "[RL-shaped] Checking best reward across shards..."
$PY -c "
import json
from pathlib import Path
best_r, best_sh = -999, -1
for sh in range(4):
    log = Path('$OUT_DIR' + f'_sh{sh}/train_log.jsonl')
    if log.exists():
        lines = log.read_text().strip().split('\n')
        if lines:
            last = json.loads(lines[-1])
            r = last.get('running_reward', -999)
            print(f'  shard{sh}: final running_reward={r:.4f}')
            if r > best_r:
                best_r, best_sh = r, sh
if best_sh >= 0:
    print(f'  BEST: shard{best_sh} (reward={best_r:.4f})')
"

# === HARD CHAIN: directly start RL eval (backup for heartbeat) ===
echo "[RL-shaped] Training done. Starting RL eval directly..."
chmod +x "$ROOT/scripts/run_rl_eval_4gpu.sh"
nohup bash "$ROOT/scripts/run_rl_eval_4gpu.sh" > "$ROOT/logs/rl_eval_auto.log" 2>&1 &
echo "[RL-shaped] RL eval launched (PID=$!)"
