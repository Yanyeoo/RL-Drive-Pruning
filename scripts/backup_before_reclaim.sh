#!/bin/bash
set -euo pipefail
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
BACKUP_DIR="backups/cycle_end_20260717_0950"
mkdir -p "$BACKUP_DIR"

echo "[backup] $(date) Starting pre-reclaim backup..."

# 1. Copy scorer RL checkpoint
if [ -d "ckpt/s3_token_scorer_rl_v1" ]; then
    cp -a ckpt/s3_token_scorer_rl_v1 "$BACKUP_DIR/scorer_rl_v1"
    echo "[backup] scorer_rl_v1 copied"
fi

# 2. Copy logs
cp -a logs/scorer_grpo_formal_ep1_20260716.log "$BACKUP_DIR/" 2>/dev/null || true
cp -a logs/varB_diag_20260716.log "$BACKUP_DIR/" 2>/dev/null || true

# 3. Record final training state
echo "[backup] Final training log tail:" >> "$BACKUP_DIR/status.txt"
tail -10 logs/scorer_grpo_formal_ep1_20260716.log >> "$BACKUP_DIR/status.txt" 2>/dev/null || true
echo "" >> "$BACKUP_DIR/status.txt"
echo "[backup] VarB diag tail:" >> "$BACKUP_DIR/status.txt"
tail -20 logs/varB_diag_20260716.log >> "$BACKUP_DIR/status.txt" 2>/dev/null || true

# 4. Git commit + push
git add -A
git commit -m "pre-reclaim backup 07-17 09:50: scorer RL v1 + varB diag" || true
git push origin main || true

echo "[backup] $(date) Done."
