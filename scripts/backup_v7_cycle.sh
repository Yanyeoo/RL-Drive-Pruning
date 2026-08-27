#!/usr/bin/env bash
# ============================================================================
# backup_v7_cycle.sh — v7 周期收尾备份（4×H20 回收前 13:50 执行）
#
# 只做文件拷贝，不执行任何 git push / 删除等不可逆操作。
# 用法：bash scripts/backup_v7_cycle.sh [CYCLE_ID]
# ============================================================================
set -euo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
CYCLE_ID="${1:-$(cat logs/v7_surrogate_latest 2>/dev/null || echo unknown)}"
STAMP=$(date +%Y%m%d_%H%M)
BACKUP_DIR="$ROOT/backups/v7_cycle_${CYCLE_ID}_${STAMP}"
mkdir -p "$BACKUP_DIR"
echo "[backup] $(date) -> $BACKUP_DIR"

# 1. RL checkpoints（4 个 arm）
if [[ -d "ckpt/v7_surrogate_${CYCLE_ID}" ]]; then
    cp -a "ckpt/v7_surrogate_${CYCLE_ID}" "$BACKUP_DIR/ckpt_v7_surrogate"
    echo "[backup] ckpt copied"
fi

# 2. 评估 CSV（gate / full）
if [[ -d "results/raw/v7_surrogate_${CYCLE_ID}_gate" ]]; then
    cp -a "results/raw/v7_surrogate_${CYCLE_ID}_gate" "$BACKUP_DIR/results_gate"
    echo "[backup] gate csv copied"
fi
if [[ -d "results/raw/v7_surrogate_${CYCLE_ID}_full" ]]; then
    cp -a "results/raw/v7_surrogate_${CYCLE_ID}_full" "$BACKUP_DIR/results_full"
    echo "[backup] full csv copied"
fi

# 3. 训练/评估日志 + 编排日志
if [[ -d "logs/v7_surrogate_${CYCLE_ID}" ]]; then
    cp -a "logs/v7_surrogate_${CYCLE_ID}" "$BACKUP_DIR/logs_v7_surrogate"
    echo "[backup] logs copied"
fi
cp -a logs/v7_surrogate_orchestrator.log "$BACKUP_DIR/" 2>/dev/null || true

# 4. 关键代码改动（训练脚本 + eval 脚本 + 编排脚本）
mkdir -p "$BACKUP_DIR/scripts"
cp -a scripts/train_scorer_budget_rl.py "$BACKUP_DIR/scripts/"
cp -a scripts/run_v7_surrogate_ablation_4gpu.sh "$BACKUP_DIR/scripts/"
cp -a scripts/eval_v7_folds_4gpu.sh "$BACKUP_DIR/scripts/"
cp -a scripts/smoke_surrogate.py "$BACKUP_DIR/scripts/"
echo "[backup] scripts copied"

# 5. 状态文档
cp -a STATUS.md "$BACKUP_DIR/STATUS.md" 2>/dev/null || true
cp -a docs/journal/2026-08-18_v7_diff_topk.md "$BACKUP_DIR/journal_v7.md" 2>/dev/null || true

# 6. 记录备份清单 + 校验
echo "[backup] generating manifest..."
{
    echo "cycle_id=$CYCLE_ID"
    echo "stamp=$STAMP"
    echo "backup_dir=$BACKUP_DIR"
    echo "--- ckpt tree ---"
    find "$BACKUP_DIR/ckpt_v7_surrogate" -maxdepth 2 -type f 2>/dev/null | head -50
    echo "--- csv count ---"
    find "$BACKUP_DIR" -name "*.csv" 2>/dev/null | wc -l
} > "$BACKUP_DIR/MANIFEST.txt"
cat "$BACKUP_DIR/MANIFEST.txt"

echo "[backup] $(date) DONE. Size:"
du -sh "$BACKUP_DIR" 2>/dev/null
