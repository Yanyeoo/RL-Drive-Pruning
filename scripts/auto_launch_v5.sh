#!/usr/bin/env bash
# auto_launch_v5.sh — 消融 eval 完成后自动启动 v5 训练
# 等 GPU0/1/2 的 eval 进程结束后，启动 4 GPU v5 训练
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$AUTOVLA_ROOT/navsim:$AUTOVLA_ROOT:${PYTHONPATH:-}"

LOGF="$ROOT/logs/auto_launch_v5.log"
log(){ echo "[v5-launch $(date +%H:%M:%S)] $*" | tee -a "$LOGF"; }

log "Waiting for ablation evals to finish..."

# Wait for eval processes (look for run_pdm_score_cot.py)
while ps aux | grep -q "[r]un_pdm_score_cot"; do
  sleep 120
done
log "All evals done at $(date)"

# Copy CSVs
for tag in SOTAV4_R1_SAFENET SOTAV3_R1_FINAL SFT_r0355; do
  f=$(ls -t "$ROOT/exp/${tag}_sh0"/*/*.csv 2>/dev/null | head -1)
  if [[ -n "$f" ]]; then
    cp -a "$f" "$ROOT/results/raw/${tag}_sh0.csv"
    pdms=$($PY -c "import pandas as pd;d=pd.read_csv('$f');d=d[d['token']!='average'];print(f'{d[\"score\"].mean():.5f}')" 2>/dev/null)
    log "$tag: PDMS=$pdms"
  fi
done

# Aggregate all results so far
log "=== Quick aggregate ==="
$PY - << 'PYEOF'
import glob, pandas as pd
baselines = {"SFT r=0.5": 0.89008, "SFT r=0.355": 0.85575, "no-prune": 0.89886, "v3 old": 0.87066}
print(f"{'Method':<35} {'PDMS':>8}")
print("-"*45)
# v4
for tag in ["SOTAV4_R1_FINAL", "SOTAV4_R1_BEST", "SOTAV4_R1_SAFENET", "SOTAV3_R1_FINAL", "SFT_r0355"]:
    for f in sorted(glob.glob(f"/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/{tag}_sh*.csv")):
        d = pd.read_csv(f); d = d[d['token']!='average']
        print(f"{tag:<35} {d['score'].mean():>8.5f} (N={len(d)})")
        break
for name, pdms in baselines.items():
    print(f"{name:<35} {pdms:>8.5f}")
PYEOF

# Launch v5 training (4 GPU, 1 epoch, true PDMS reward, no efficiency bonus)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="$ROOT/ckpt/s3_token_scorer_budget_rl_${TIMESTAMP}"
SCORER_CKPT="$ROOT/ckpt/s3_token_scorer"
BASELINE="$ROOT/results/baseline_sub_scores.json"

log "=== Launching v5 training ==="
log "true_pdms=True, efficiency_beta=0, driving_scale=1.0"
log "out: $OUT_BASE"

PIDS=""
for SH in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES=$SH
    $PY scripts/train_scorer_budget_rl.py \
      --scorer-ckpt "$SCORER_CKPT" \
      --out-dir "${OUT_BASE}_sh${SH}" \
      --json-dir "$ROOT/data/navtrain_nocot" \
      --metric-cache "$ROOT/data/navtrain_metric_cache" \
      --efficiency-beta 0.0 \
      --driving-scale 1.0 \
      --num-epochs 1 \
      --group-size 16 \
      --lr 3e-5 --budget-lr 1e-4 \
      --kl-beta 0.01 \
      --min-keep-ratio 0.2 --max-keep-ratio 0.9 \
      --budget-log-std-init -0.5 \
      --safety-beta 0.05 --safety-margin 0.02 \
      --shaped-reward --baseline-scores "$BASELINE" \
      --num-shards 4 --shard-id $SH \
      --seed $((42 + SH)) \
      --prune-variant attn_mask \
      --device cuda:0
  ) > "$ROOT/logs/sota_v5_R1_sh${SH}.log" 2>&1 &
  PIDS="$PIDS $!"
  log "sh${SH} PID=$!"
done

echo "$PIDS" > "$ROOT/logs/sota_v5_R1.pids"
echo "$OUT_BASE" > "$ROOT/logs/sota_v5_R1.outbase"
log "v5 PIDs: $PIDS"
log "Launch complete at $(date)"
