#!/usr/bin/env bash
# auto_chain_v6.sh — v6 无人值守全链脚本
#
# 链结构:
#   wait_train → v6 full eval (4GPU, 4shard, ~3.6h)
#   → ablation (ckpt_best / safety_net / SFT r=0.355 / v3对比)
#   → 7B nuScenes cross-model eval (best-effort)
#   → 汇总报告
#
# 用法: bash scripts/auto_chain_v6.sh
# 前提: v6 训练已启动 (PIDs 在 logs/v6_train/train.pids)
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"
NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CHAINLOG="$ROOT/logs/v6_train/auto_chain_$(date +%Y%m%d_%H%M%S).log"
LOGDIR="$ROOT/logs/auto_chain_v6"
mkdir -p "$LOGDIR"
log(){ echo "[chain $(date +%H:%M:%S)] $*" | tee -a "$CHAINLOG"; }

# Config
CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
PREFIX="navtest_local_filtered_shard"; SUFFIX="_20260616_154858"
OUTDIR="$ROOT/results/raw"

###############################################################################
# Helper: eval one shard on one GPU
###############################################################################
eval_shard(){
  local gpu="$1" ckpt="$2" tag="$3" sh="$4"
  local extra_flags="${5:-}"  # optional extra hydra flags (e.g. +agent.safety_net=true)
  local exp="${tag}_sh${sh}"
  local csv="$OUTDIR/${exp}.csv"
  local jlog="$LOGDIR/_${exp}.log"
  if [[ -f "$csv" ]]; then
    log "SKIP $exp (csv exists)"
    return
  fi
  log "GPU$gpu START $exp"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    cd "$NAVSIM_ROOT"
    # shellcheck disable=SC2086
    $PY navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" \
      train_test_split="${PREFIX}${sh}${SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" \
      +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false \
      +agent.selector=scorer_budget \
      +agent.scorer_ckpt="$ckpt" \
      +agent.keep_ratio=0.5 \
      +agent.prune_variant=drop +agent.prune_verbose=true \
      $extra_flags \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$jlog" 2>&1
  local found; found=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    cp -a "$found" "$csv"
    local pdms; pdms=$($PY -c "import pandas as pd;d=pd.read_csv('$found');d=d[d['token']!='average'];print(f'{d[\"score\"].mean():.5f}')" 2>/dev/null)
    log "GPU$gpu DONE  $exp: PDMS=$pdms N=$(wc -l < "$csv")"
  else
    log "GPU$gpu FAIL  $exp: no CSV produced"
  fi
}

aggregate_csvs(){
  local tag="$1"
  $PY - << PYEOF
import glob, pandas as pd
rows = []
for f in sorted(glob.glob(f"/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/${tag}_sh*.csv")):
    d = pd.read_csv(f); d = d[d['token'] != 'average']
    nc = d['num_penalties_collisions'].mean() if 'num_penalties_collisions' in d.columns else float('nan')
    ep = d['ego_progress'].mean() if 'ego_progress' in d.columns else float('nan')
    rows.append({'shard': f.split('_sh')[-1].replace('.csv',''), 'N': len(d), 'PDMS': d['score'].mean(), 'NC': nc, 'EP': ep})
    print(f"  {f.split('/')[-1]}: N={len(d)} PDMS={d['score'].mean():.5f} NC={nc:.4f} EP={ep:.4f}")
if len(rows) > 1:
    tn = sum(r['N'] for r in rows)
    wp = sum(r['PDMS']*r['N'] for r in rows)/tn
    print(f"\n  {tag} OVERALL: N={tn} PDMS={wp:.5f}")
elif len(rows) == 1:
    print(f"\n  {tag}: N={rows[0]['N']} PDMS={rows[0]['PDMS']:.5f}")
else:
    print(f"\n  {tag}: NO DATA")
PYEOF
}

###############################################################################
# Step 0: Wait for v6 training
###############################################################################
log "=== Step 0: Waiting for v6 training ==="
TRAIN_PIDFILE="$ROOT/logs/v6_train/train.pids"
OUTBASE_FILE="$ROOT/logs/v6_train/outbase"

if [[ -f "$TRAIN_PIDFILE" ]]; then
  for pid in $(cat "$TRAIN_PIDFILE"); do
    log "Waiting for PID $pid..."
    while kill -0 "$pid" 2>/dev/null; do sleep 120; done
    log "PID $pid done"
  done
fi
log "Training complete at $(date)"

# Find ckpt
OUT_BASE=$(cat "$OUTBASE_FILE" 2>/dev/null)
V6_SH0="${OUT_BASE}_sh0"
V6_BEST="${OUT_BASE}_sh0/ckpt_best"
log "v6 ckpt: $V6_SH0"

if [[ ! -f "$V6_SH0/checkpoint.pt" ]]; then
  LATEST=$(ls -d "${OUT_BASE}_sh0/ckpt_step"* 2>/dev/null | sort -V | tail -1)
  if [[ -n "$LATEST" && -f "$LATEST/checkpoint.pt" ]]; then
    V6_SH0="$LATEST"
    log "v6 fallback to latest step: $V6_SH0"
  else
    log "FATAL: no v6 checkpoint found"
    exit 2
  fi
fi

###############################################################################
# Step 1: v6 FINAL full eval (4 GPU, 4 shard, ~3.6h)
###############################################################################
log "=== Step 1: v6 FINAL full eval (4 GPU x 4 shard) ==="
TAG_V6F="SOTAV6_R1_FINAL"
for SH in 0 1 2 3; do
  eval_shard $SH "$V6_SH0" "$TAG_V6F" $SH &
done
wait
log "v6 eval done at $(date)"
aggregate_csvs "$TAG_V6F" 2>&1 | tee -a "$CHAINLOG"

###############################################################################
# Step 2: v6 ckpt_best eval + 消融
###############################################################################
log "=== Step 2: Ablation experiments ==="

# 2a: v6 ckpt_best eval (GPU0)
if [[ -f "$V6_BEST/checkpoint.pt" ]]; then
  eval_shard 0 "$V6_BEST" "SOTAV6_R1_BEST" 0 &
  PID_BEST=$!
  log "GPU0 START v6 BEST"
else
  log "SKIP v6 BEST: no ckpt_best"
  PID_BEST=""
fi

# 2b: v6 FINAL + safety_net (GPU1)
eval_shard 1 "$V6_SH0" "SOTAV6_R1_SAFENET" 0 "+agent.safety_net=true" &
PID_SAFE=$!
log "GPU1 START v6 SAFENET"

# 2c: SFT r=0.355 shard0 — matched-compute baseline (GPU2)
SFT_CKPT="$ROOT/ckpt/s3_token_scorer"
eval_shard 2 "$SFT_CKPT" "SFT_r0355_sh0" 0 &
PID_SFT=$!
log "GPU2 START SFT r=0.355"

# 2d: v3 old ckpt shard0 — v3 vs v6 直接对比 (GPU3)
V3_CKPT="$ROOT/ckpt/s3_token_scorer_budget_rl_20260803_151608_sh0"
if [[ -f "$V3_CKPT/checkpoint.pt" ]]; then
  eval_shard 3 "$V3_CKPT" "SOTAV3_R1_FINAL" 0 &
  PID_V3=$!
  log "GPU3 START v3 (eff_beta=0.15)"
else
  log "SKIP v3: no ckpt"
  PID_V3=""
fi

wait ${PID_BEST:-} ${PID_SAFE:-} ${PID_SFT:-} ${PID_V3:-}
log "Ablation done at $(date)"

aggregate_csvs "SOTAV6_R1_BEST" 2>&1 | tee -a "$CHAINLOG"
aggregate_csvs "SOTAV6_R1_SAFENET" 2>&1 | tee -a "$CHAINLOG"
aggregate_csvs "SFT_r0355_sh0" 2>&1 | tee -a "$CHAINLOG"
aggregate_csvs "SOTAV3_R1_FINAL" 2>&1 | tee -a "$CHAINLOG"

###############################################################################
# Step 3: kr distribution analysis
###############################################################################
log "=== Step 3: kr distribution analysis ==="
$PY - << 'PYEOF' 2>&1 | tee -a "$CHAINLOG"
import re, glob, statistics, json

def extract_kr(logfile):
    kr = {}
    pat = re.compile(r'\[token_budget\] scene=(\w+).*?kr=([0-9.]+)')
    try:
        with open(logfile) as f:
            for line in f:
                m = pat.search(line)
                if m: kr[m.group(1)] = float(m.group(2))
    except: pass
    return kr

all_kr = {}
for f in sorted(glob.glob("/apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/auto_chain_v6/_SOTAV6_R1_FINAL_sh*.log")):
    sh = f.split('_sh')[-1].replace('.log','')
    kr = extract_kr(f)
    all_kr.update(kr)
    vals = list(kr.values())
    if vals:
        print(f"  sh{sh}: N={len(vals)} mean_kr={statistics.mean(vals):.4f} median_kr={statistics.median(vals):.4f}")

vals_all = list(all_kr.values())
if vals_all:
    print(f"\n  OVERALL: N={len(vals_all)} mean_kr={statistics.mean(vals_all):.4f} median_kr={statistics.median(vals_all):.4f}")
    bins = [(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.9)]
    for lo, hi in bins:
        cnt = sum(1 for v in vals_all if lo <= v < hi)
        print(f"    kr [{lo:.1f},{hi:.1f}): {cnt:>5} ({100*cnt/len(vals_all):.1f}%)")
    with open("/apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/auto_chain_v6/kr_distribution.json", "w") as f:
        json.dump({"mean": statistics.mean(vals_all), "median": statistics.median(vals_all),
                   "std": statistics.pstdev(vals_all), "N": len(vals_all)}, f)
PYEOF

###############################################################################
# Step 4: Training curve analysis
###############################################################################
log "=== Step 4: Training curve analysis ==="
$PY - << 'PYEOF' 2>&1 | tee -a "$CHAINLOG"
import json, glob, statistics
for sh in range(4):
    pattern = f"/apdcephfs/private_shayladeng/tokenrl_autoVLA/ckpt/s3_token_scorer_budget_rl_v6_*_sh{sh}/train_log.jsonl"
    files = sorted(glob.glob(pattern))
    if not files: continue
    rewards, krs = [], []
    with open(files[-1]) as f:
        for line in f:
            d = json.loads(line)
            if 'reward_mean' in d:
                rewards.append(d['reward_mean'])
                krs.append(d['keep_ratio_mean'])
    if len(rewards) > 10:
        n = len(rewards); q = n // 4
        print(f"sh{sh}: steps={n}")
        for start, end, label in [(0, q, "Q1"), (q, 2*q, "Q2"), (2*q, 3*q, "Q3"), (3*q, n, "Q4")]:
            print(f"  {label}: R={statistics.mean(rewards[start:end]):.4f} kr={statistics.mean(krs[start:end]):.3f}")
        print(f"  Final: R={rewards[-1]:.4f} kr={krs[-1]:.3f}")
PYEOF

###############################################################################
# Step 5: Final report
###############################################################################
log "=== Step 5: Final report ==="
$PY - << 'PYEOF' 2>&1 | tee -a "$CHAINLOG"
import json, glob, pandas as pd, time

report = {
    "run": "v6_TruePDMS_RL",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "method": "True PDMS product reward, no efficiency bonus, 2 epochs",
    "results": {}
}

baselines = {
    "SFT r=0.5": 0.89008,
    "v3 Budget RL (raw)": 0.87066,
    "v4 Budget RL (FINAL)": 0.86094,
    "no-prune": 0.89886,
}

for tag in ["SOTAV6_R1_FINAL", "SOTAV6_R1_BEST", "SOTAV6_R1_SAFENET",
            "SFT_r0355_sh0", "SOTAV3_R1_FINAL"]:
    pattern = f"/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/{tag}_sh*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        report["results"][tag] = "NOT FOUND"
        continue
    dfs = [pd.read_csv(f) for f in files]
    df_all = pd.concat(dfs)
    df_all = df_all[df_all['token'] != 'average']
    report["results"][tag] = {
        "N": len(df_all),
        "PDMS": round(df_all['score'].mean(), 5),
    }
    print(f"  {tag}: N={len(df_all)} PDMS={df_all['score'].mean():.5f}")

# Comparison
v6_pdms = report["results"].get("SOTAV6_R1_FINAL", {}).get("PDMS", None)
print("\n=== KEY COMPARISON ===")
for name, pdms in baselines.items():
    print(f"  {name}: {pdms:.5f}")
if v6_pdms:
    print(f"  v6 FINAL: {v6_pdms:.5f}")
    delta_sft = v6_pdms - 0.89008
    delta_v4 = v6_pdms - 0.86094
    print(f"  v6 vs SFT r=0.5: {delta_sft:+.5f}")
    print(f"  v6 vs v4:         {delta_v4:+.5f}")
    if v6_pdms > 0.89008:
        print("  >>> v6 EXCEEDS SFT baseline! ICLR core claim valid.")
    elif v6_pdms > 0.885:
        print("  >>> v6 close to SFT, promising direction.")
    else:
        print("  >>> v6 below SFT. Need per-token credit assignment (ICLR PLAN §2).")

with open("/apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/auto_chain_v6/final_report.json", "w") as f:
    json.dump(report, f, indent=2)

report["baselines"] = baselines
print(f"\n  JSON report: logs/auto_chain_v6/final_report.json")
PYEOF

###############################################################################
# Step 6: 7B nuScenes cross-model eval (best-effort, ~2h, GPU0)
###############################################################################
log "=== Step 6: 7B nuScenes cross-model eval ==="
MODEL_7B="$ROOT/models/ImpromptuVLA_7B/7B_AD_finetune"
SCORER_7B="$ROOT/ckpt/s3_token_scorer_7b"
DATA_7B="$ROOT/code/third_party/ImpromptuVLA/nuscenes_test.json"
OUT_7B="$ROOT/results/impromptu7b"
EVAL_SCRIPT="$ROOT/scripts/run_impromptu7b_nuscenes_eval.py"
EVAL_METRICS="$ROOT/code/third_party/ImpromptuVLA/data_qa_generate/data_engine/datasets/nuscenes/scripts/evaluation_nuscenes.py"
mkdir -p "$OUT_7B"

if [[ ! -f "$MODEL_7B/config.json" ]]; then
  log "SKIP 7B: model not found"
elif [[ ! -f "$SCORER_7B/checkpoint.pt" ]]; then
  log "SKIP 7B: scorer not found"
elif [[ ! -f "$DATA_7B" ]]; then
  log "SKIP 7B: data not found"
else
  log "Launch 7B nuScenes eval (r=0.5, 2000 scenes, GPU0, timeout 14400s)"
  (
    export CUDA_VISIBLE_DEVICES=0
    cd "$ROOT"
    timeout 14400 $PY "$EVAL_SCRIPT" \
      --model-path "$MODEL_7B" \
      --scorer-ckpt "$SCORER_7B" \
      --keep-ratio 0.5 \
      --data-json "$DATA_7B" \
      --max-scenes 2000 \
      --device cuda:0 \
      --output "$OUT_7B/pred_r05_2k.jsonl" \
      > "$LOGDIR/_7b_nuscenes_r05.log" 2>&1
    rc=$?
    n=$(grep -c '"predict"' "$OUT_7B/pred_r05_2k.jsonl" 2>/dev/null || echo 0)
    log "7B r=0.5 done: rc=$rc preds=$n"

    if [[ "$n" -gt 0 && -f "$EVAL_METRICS" ]]; then
      log "Running 7B metrics..."
      cd "$(dirname "$EVAL_METRICS")"
      timeout 600 $PY "$EVAL_METRICS" \
        --jsonl_file "$OUT_7B/pred_r05_2k.jsonl" \
        --output_file "$OUT_7B/eval_r05_2k.json" \
        --mode x-y \
        > "$LOGDIR/_7b_metrics_r05.log" 2>&1 || true
      cd "$ROOT"
    fi
  ) &
  PID_7B=$!
  log "7B eval PID=$PID_7B (background)"
fi

log "=== Chain complete at $(date) ==="
log "Report: logs/auto_chain_v6/final_report.json"
log "CSVs:   results/raw/SOTAV6_R1_*.csv"
log "Logs:   logs/auto_chain_v6/"
