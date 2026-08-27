#!/usr/bin/env bash
# ============================================================================
# orchestrate_v11_pipeline.sh — v11 全流程无人值守（4×H20，deadline 8/27 20:00）
#
# 阶段：
#   0. 等 mining 结束（run_v11_mining_4gpu.sh 已在跑）
#   1. 组装 hard-enriched 场景表（hard50 / hard75）
#   2. 4 卡 4 臂训练（全部 warm-start 自 v10 full_hi）
#   3. gate 评估：4 臂 × shard0/1 = 8 job，4 卡分 2 波
#   4. 选 winner → full 4-shard 定论
#   5. 汇总对比 no-prune SOTA
#
# 时间预算（实测：mine 4.5s/scene，train ~1.3h/臂，eval ~1.8h/shard/卡）：
#   mining 结束 ~01:30 → 训练 ~03:00 → gate ~06:40 → full ~08:30，余量 >11h
#
# 用法：nohup bash scripts/orchestrate_v11_pipeline.sh > logs/v11_orchestrate.log 2>&1 &
# ============================================================================
set -uo pipefail
ROOT=/apdcephfs/private_shayladeng/tokenrl_autoVLA
cd "$ROOT"
PY=/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python

MINE_CYCLE=20260826_v11
MINE_DIR="$ROOT/results/mining_${MINE_CYCLE}"
MINE_LOG_DIR="$ROOT/logs/v7_surrogate_v11_${MINE_CYCLE}"
ROUND=r1
CYCLE_ID="20260826_v11${ROUND}"
CYCLE_DIR="$ROOT/logs/v7_surrogate_${CYCLE_ID}"
RUN_ROOT="$ROOT/ckpt/v7_surrogate_${CYCLE_ID}"
mkdir -p "$CYCLE_DIR"

log(){ echo "[v11-orch $(date +%m-%d_%H:%M:%S)] $*"; }

# ---------- 阶段 0：等 mining ----------
log "waiting for mining to finish..."
while [[ ! -f "$MINE_LOG_DIR/MINING_COMPLETE" ]]; do
  n=$(cat "$MINE_DIR"/mine_shard*.jsonl 2>/dev/null | wc -l)
  log "mining in progress: $n records"
  sleep 300
done
log "mining complete: $(cat "$MINE_DIR"/mine_shard*.jsonl | wc -l) records"

# ---------- 阶段 1：组装场景表 ----------
LIST50="$MINE_DIR/scenes_hard50.txt"
LIST75="$MINE_DIR/scenes_hard75.txt"
log "building scene lists"
"$PY" scripts/build_v11_scene_list.py --mine-glob "$MINE_DIR/mine_shard*.jsonl" \
  --out "$LIST50" --n-total 512 --hard-frac 0.5 --hard-repeat 2 \
  2>&1 | tee "$CYCLE_DIR/scene_list_hard50.txt"
"$PY" scripts/build_v11_scene_list.py --mine-glob "$MINE_DIR/mine_shard*.jsonl" \
  --out "$LIST75" --n-total 512 --hard-frac 0.75 --hard-repeat 2 \
  2>&1 | tee "$CYCLE_DIR/scene_list_hard75.txt"
[[ -s "$LIST50" && -s "$LIST75" ]] || { log "FATAL scene list empty"; exit 1; }

# ---------- 阶段 2：训练 ----------
log "phase2: training 4 arms"
bash scripts/run_v11_hardmine_4gpu.sh "$LIST50" "$LIST75" "$ROUND" \
  > "$CYCLE_DIR/train_driver.log" 2>&1
log "training done"

ARMS=()
for a in hard50 hard50_lr hard75 hard50_kl; do
  [[ -f "$RUN_ROOT/$a/checkpoint.pt" ]] && ARMS+=("$a") || log "WARN missing arm $a"
done
[[ ${#ARMS[@]} -gt 0 ]] || { log "FATAL no trained arms"; exit 1; }
log "arms to gate: ${ARMS[*]}"

# ---------- 阶段 3：gate 评估（4 卡，eval 脚本内部按 GPU 队列派发）----------
log "phase3: gate eval on shard0+1"
EVAL_WORKERS=1 EVAL_GPUS="0 1 2 3" \
  bash scripts/eval_v7_folds_4gpu.sh "$CYCLE_ID" gate "${ARMS[@]}" \
  > "$CYCLE_DIR/gate_eval.log" 2>&1
log "gate eval done"

# ---------- 阶段 4：选 winner ----------
WINNER=$("$PY" - "$ROOT/results/raw/v7_surrogate_${CYCLE_ID}_gate" <<'PY'
import os, sys
from collections import defaultdict
import pandas as pd
d = sys.argv[1]
acc = defaultdict(list)
for f in sorted(os.listdir(d)):
    if not f.endswith('.csv'):
        continue
    parts = f[:-4].split('_')          # v7_gate_<arm>_sh<n>
    arm = '_'.join(parts[2:-1])
    df = pd.read_csv(os.path.join(d, f))
    df = df[df['token'] != 'average']
    acc[arm].append(df['score'].mean())
best, best_p = "", -1.0
for arm, ps in acc.items():
    m = sum(ps) / len(ps)
    print(f"# {arm} gate PDMS {m:.6f} ({len(ps)} shards)", file=sys.stderr)
    if m > best_p:
        best, best_p = arm, m
print(best)
PY
)
log "gate winner: $WINNER"
[[ -n "$WINNER" ]] || { log "FATAL no winner"; exit 1; }

# ---------- 阶段 5：full 4-shard ----------
log "phase5: full eval for $WINNER"
EVAL_WORKERS=1 EVAL_GPUS="0 1 2 3" \
  bash scripts/eval_v7_folds_4gpu.sh "$CYCLE_ID" full "$WINNER" \
  > "$CYCLE_DIR/full_eval.log" 2>&1
log "full eval done"

# ---------- 汇总 ----------
log "final summary vs no-prune SOTA"
"$PY" - "$ROOT/results/raw/v7_surrogate_${CYCLE_ID}_full" "$WINNER" <<'PY'
import glob, os, sys
import pandas as pd
d, winner = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(d, '*.csv')))
if not files:
    print("no full csv"); raise SystemExit
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df[df['token'] != 'average'].drop_duplicates('token')
pdms = df['score'].mean()
SOTA = 0.898845   # no-prune attn_L12 r=1.0, N=11576
print(f"v11 winner   : {winner}")
print(f"N scenes     : {len(df)}")
print(f"v11 PDMS     : {pdms:.6f}")
print(f"no-prune SOTA: {SOTA:.6f}")
print(f"delta        : {pdms - SOTA:+.6f}   {'*** BEATS SOTA ***' if pdms > SOTA else '(below)'}")
print(f"v10 full_hi  : 0.898274  (delta {0.898274 - SOTA:+.6f})")
PY

log "V11 PIPELINE COMPLETE" | tee "$CYCLE_DIR/V11_COMPLETE"

# ---------- 阶段 6：efficiency 收尾 —— 实测 latency（GPU 此时空闲）----------
# efficiency 主张需要三件套：keep_ratio（已有）、FLOPs（已算）、实测 latency。
log "phase6: wall-clock latency profile for winner"
"$PY" scripts/profile_wallclock.py --n-scenes 30 --gpu 0 \
  --budget-ckpt "$RUN_ROOT/$WINNER" \
  --output "results/profiling/v11_${WINNER}_wallclock.json" \
  > "$CYCLE_DIR/latency_profile.log" 2>&1 \
  && log "latency profile written" || log "WARN latency profile failed (see log)"

log "ALL DONE"
