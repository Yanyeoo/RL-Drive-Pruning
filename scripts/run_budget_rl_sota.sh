#!/usr/bin/env bash
# run_budget_rl_sota.sh — 2026-08-03 主线 RL 训练
# 目标：训练 Budget RL 超越 SFT baseline (PDMS 0.892) → 达到可发表水平
#
# 与原始 run_budget_rl_navtrain.sh 的区别：
#   efficiency_beta: 0.15 → 0.05   (降低过度剪枝激励)
#   driving_scale:   2.0  → 3.0    (增大驾驶质量权重)
#   num_epochs:      3    → 1      (今晚可跑完)
#   其他超参不变（lr, kl, budget_lr, min/max_keep_ratio）
#
# 启动时间：Block A 完成后 (≈15:30)
# ETA：~5-6h (8 GPU, 1 epoch, ~300 steps)
# 断点续训：支持，被 kill 后重新运行自动 resume
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

# === 策略 A 超参 ===
NUM_EPOCHS=1
GROUP_SIZE=16
EFFICIENCY_BETA=0.05       # 原 0.15 → 0.05
DRIVING_SCALE=3.0           # 原 2.0  → 3.0
BUDGET_LR=1e-4
LR=3e-5
KL_BETA=0.01
PRUNE_VARIANT=attn_mask     # 训练用 mask 代理

SCORER_CKPT="$ROOT/ckpt/s3_token_scorer"
OUT_DIR="$ROOT/ckpt/s3_token_scorer_budget_rl_$(date +%Y%m%d_%H%M%S)"
BASELINE="$ROOT/results/baseline_sub_scores.json"

# Resume 支持
PREV=$(cat "$ROOT/logs/budget_rl_sota_outdir.txt" 2>/dev/null)
if [[ -n "$PREV" && -d "${PREV}_sh0/ckpt_resume" ]]; then
    OUT_DIR="$PREV"
    echo "[sota-rl] RESUME into existing out_dir: $OUT_DIR"
else
    echo "$OUT_DIR" > "$ROOT/logs/budget_rl_sota_outdir.txt"
fi

echo "=========================================="
echo "[sota-rl] SOTA Budget RL training"
echo "  epochs=$NUM_EPOCHS group=$GROUP_SIZE"
echo "  eff_beta=$EFFICIENCY_BETA drive_scale=$DRIVING_SCALE"
echo "  budget_lr=$BUDGET_LR lr=$LR kl=$KL_BETA"
echo "  out_dir: $OUT_DIR"
echo "=========================================="

PIDS=""
for SH in 0 1 2 3 4 5 6 7; do
    GPU=$SH
    SHARD_OUT="${OUT_DIR}_sh${SH}"
    echo "[sota-rl] GPU$GPU shard$SH -> $SHARD_OUT"
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
    ) > "$ROOT/logs/budget_rl_sota_sh${SH}.log" 2>&1 &
    PIDS="$PIDS $!"
done

echo "$PIDS" > "$ROOT/logs/budget_rl_sota.pids"
echo "[sota-rl] PIDs: $PIDS"
echo "[sota-rl] Waiting..."
wait
echo "[sota-rl] ALL DONE at $(date)"

# Auto-chain: 训练完成后自动 eval
echo "[sota-rl] Training done. Launching eval..."
mkdir -p "$ROOT/results/raw"
bash "$ROOT/scripts/run_budget_rl_eval_quick.sh" "$OUT_DIR" > "$ROOT/logs/budget_rl_sota_eval.log" 2>&1 &
echo "[sota-rl] Eval launched (PID=$!)"
