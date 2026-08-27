#!/usr/bin/env bash
# ============================================================================
# orchestrate_v10_pipeline.sh — v10 方法 A 全自动流水线（无人值守）
#
# 流程：
#   1. 等待 v9 full eval（ep2+kl005, 8 卡）完成
#   2. 启动 v10 训练（value_only / floor_init / full / full_hi，4 卡）
#   3. gate 评估（4 arm × 2 shard，8 卡并行）
#   4. 选 gate 胜者 → full 4-shard 评估
#   5. 汇总写结果
#
# 用法：bash scripts/orchestrate_v10_pipeline.sh
# ============================================================================
set -uo pipefail
ROOT=/apdcephfs/private_shayladeng/tokenrl_autoVLA
cd "$ROOT"

ROUND="r1"
CYCLE_ID="20260825_v10${ROUND}"
ARMS=(value_only floor_init full full_hi)
OUTROOT="$ROOT/results/raw/v7_surrogate_${CYCLE_ID}"

log() { echo "[v10-orch $(date +%H:%M:%S)] $*"; }

# ---- 1. 等待 v9 full eval 完成 -------------------------------------------
log "step1: waiting for v9 full eval (ep2+kl005) to finish..."
while true; do
  n=$(pgrep -f run_pdm_score_cot 2>/dev/null | wc -l)
  n_csv=$(ls "$ROOT/results/raw/v7_surrogate_20260824_v9r1_full/"*.csv 2>/dev/null | wc -l)
  if [[ "$n" -eq 0 && "$n_csv" -ge 8 ]]; then break; fi
  log "  still running: $n procs, $n_csv/8 csv"
  sleep 120
done
log "step1 done: v9 full eval finished ($n_csv csv)"

# ---- 1.5 smoke test（1 卡，验证 value baseline + floor + init 不 crash） ----
log "step1.5: smoke test (full_hi config, 4 scenes, 2 steps, gpu 0)"
SMOKE_DIR="$ROOT/logs/v7_surrogate_${CYCLE_ID}/smoke"
mkdir -p "$SMOKE_DIR"
PY=/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"
NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh" >/dev/null
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_scorer_budget_rl.py \
  --scorer-ckpt "$ROOT/ckpt/s3_token_scorer" \
  --json-dir "$ROOT/data/navtrain_nocot" \
  --metric-cache "$ROOT/data/navtrain_metric_cache" \
  --sensor-data-path "$ROOT/data/navsim_v2_local" \
  --autovla-config "$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml" \
  --autovla-ckpt "$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt" \
  --num-epochs 1 --max-scenes 4 --group-size 2 \
  --lr 3e-5 --budget-lr 1e-4 --kl-beta 0.01 --budget-kl-beta 0.0 \
  --selection-pg-weight 1.0 --selection-mode st_topk --selection-tau 0.1 \
  --driving-scale 3.0 --delta-reward \
  --safety-beta 0.5 --safety-margin 0.0 --min-keep-ratio 0.35 \
  --budget-log-std-init -1 --seed 3407 --prune-variant attn_mask \
  --baseline-scores "$ROOT/results/baseline_sub_scores.json" \
  --counterfactual-k 0 --save-every 8 --log-every 1 \
  --use-value-baseline --value-lr 1e-4 --value-loss-weight 1.0 \
  --efficiency-mode floor --target-keep-ratio 0.65 --efficiency-beta 0.02 \
  --budget-init-kr 0.70 --max-keep-ratio 0.85 \
  --out-dir "$SMOKE_DIR" > "$SMOKE_DIR/smoke.log" 2>&1
if [[ -f "$SMOKE_DIR/train_log.jsonl" ]] && [[ -f "$SMOKE_DIR/checkpoint.pt" ]]; then
  log "step1.5 smoke test PASSED (train_log + checkpoint generated)"
else
  log "step1.5 smoke test FAILED — aborting; check $SMOKE_DIR/smoke.log"
  tail -40 "$SMOKE_DIR/smoke.log"
  exit 1
fi

# ---- 2. v10 训练 ---------------------------------------------------------
log "step2: launching v10 training (4 arms on gpu 0-3)"
bash scripts/run_v10_methodA_4gpu.sh "$ROUND" 0
log "step2 done: v10 training finished"

# ---- 3. gate 评估（8 卡并行） --------------------------------------------
log "step3: launching gate eval (4 arms x shard0+1, 8 gpu x 1 worker)"
# NOTE: EVAL_WORKERS must be 1 — AutoVLA loads weights via meta-tensor, and
# multi-threaded loading crashes with "Cannot copy out of meta tensor".
EVAL_GPUS="0 1 2 3 4 5 6 7" EVAL_WORKERS=1 \
  bash scripts/eval_v7_folds_4gpu.sh "$CYCLE_ID" gate "${ARMS[@]}"
log "step3 done: gate eval finished"

# ---- 4. 选 winner --------------------------------------------------------
WINNER=$(CYCLE_ID="$CYCLE_ID" /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python - <<'PY'
import os, pandas as pd
outdir = f"results/raw/v7_surrogate_{os.environ['CYCLE_ID']}_gate"
files = sorted(f for f in os.listdir(outdir) if f.endswith('.csv'))
rows = {}
for f in files:
    parts = f[:-4].split('_')          # v7_gate_<arm>_sh<sh>
    sh = parts[-1]
    arm = '_'.join(parts[2:-1])
    df = pd.read_csv(os.path.join(outdir, f))
    df = df[df['token'] != 'average']
    rows.setdefault(arm, {})[sh] = df['score'].mean()
best, best_p = None, -1
print("gate summary:")
for arm, shmap in sorted(rows.items()):
    mean = sum(shmap.values()) / len(shmap)
    minv = min(shmap.values())
    print(f"  {arm:12s} mean={mean:.6f} min_shard={minv:.6f}")
    if mean > best_p:
        best, best_p = arm, mean
print(f"WINNER={best}")
PY
)
WINNER=$(echo "$WINNER" | grep -oP '(?<=WINNER=).*' | tr -d '[:space:]')
log "step4: gate winner = $WINNER"

# ---- 5. full 评估（4 卡并行） --------------------------------------------
log "step5: launching full eval for winner [$WINNER] (4 shards, 4 gpu x 1 worker)"
EVAL_GPUS="0 1 2 3" EVAL_WORKERS=1 \
  bash scripts/eval_v7_folds_4gpu.sh "$CYCLE_ID" full "$WINNER"
log "step5 done: full eval finished"

# ---- 6. 汇总 -------------------------------------------------------------
log "step6: final summary"
CYCLE_ID="$CYCLE_ID" WINNER="$WINNER" /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python - <<'PY'
import os, pandas as pd
cid = os.environ['CYCLE_ID']; winner = os.environ['WINNER']
outdir = f"results/raw/v7_surrogate_{cid}_full"
files = [f for f in os.listdir(outdir) if f.endswith('.csv')]
shard_scores = []
for f in sorted(files):
    df = pd.read_csv(os.path.join(outdir, f))
    df = df[df['token'] != 'average']
    sh = f[:-4].split('_')[-1]
    shard_scores.append((sh, len(df), df['score'].mean()))
print(f"=== v10 {cid} full result (winner={winner}) ===")
for sh, n, s in shard_scores:
    print(f"  shard {sh}: n={n} PDMS={s:.6f}")
mean = sum(s for _,_,s in shard_scores) / len(shard_scores)
mins = min(s for _,_,s in shard_scores)
print(f"  mean PDMS = {mean:.6f}")
print(f"  min shard = {mins:.6f}")
print(f"  vs no-prune 0.898845: {mean - 0.898845:+.6f}")
print(f"  vs v7 full  0.894789: {mean - 0.894789:+.6f}")
print(f"  vs SFT r0.75 0.898353: {mean - 0.898353:+.6f}")
PY

log "ALL DONE"
