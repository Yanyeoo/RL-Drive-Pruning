#!/usr/bin/env bash
# auto_chain_v4.sh — 无人值守实验链（v4 训练完成后自动执行）
#
# 链结构：
#   wait_train → v4_full_eval (4GPU, 4shard, ~3.6h) → aggregate
#   → ablation_evals (2GPU, shard0 only, ~3.6h) → final_report
#
# 用法：bash scripts/auto_chain_v4.sh
# 前提：v4 训练已启动（PIDs 在 logs/sota_v4_R1.pids）
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"
NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CHAINLOG="$ROOT/logs/auto_chain_v4_$(date +%Y%m%d_%H%M%S).log"
LOGDIR="$ROOT/logs/auto_chain_v4"
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
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$jlog" 2>&1
  local found; found=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    cp -a "$found" "$csv"
    local pdms; pdms=$($PY -c "import pandas as pd;d=pd.read_csv('$found');d=d[d['token']!='average'];print(f'{d[\"score\"].mean():.5f}')" 2>/dev/null)
    log "GPU$gpu DONE  $exp: PDMS=$pdms N=$(wc -l < "$csv")"
  else
    log "GPU$gpu FAIL  $exp: no CSV"
  fi
}

aggregate_csvs(){
  local tag="$1"
  $PY - << PYEOF
import glob, pandas as pd, statistics, json
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
# Step 0: Wait for training
###############################################################################
log "=== Step 0: Waiting for v4 training to complete ==="
TRAIN_PIDFILE="$ROOT/logs/sota_v4_R1.pids"
OUTBASE_FILE="$ROOT/logs/sota_v4_R1.outbase"
if [[ -f "$TRAIN_PIDFILE" ]]; then
  for pid in $(cat "$TRAIN_PIDFILE"); do
    log "Waiting for PID $pid..."
    while kill -0 "$pid" 2>/dev/null; do sleep 120; done
    log "PID $pid done"
  done
fi
log "Training complete at $(date)"

# Find ckpt
if [[ -f "$OUTBASE_FILE" ]]; then
  OUT_BASE=$(cat "$OUTBASE_FILE")
else
  OUT_BASE=$(ls -dt "$ROOT/ckpt/s3_token_scorer_budget_rl_20260804_"*_sh0 2>/dev/null | head -1 | sed 's/_sh0$//')
fi
V4_SH0="${OUT_BASE}_sh0"
V4_BEST="${OUT_BASE}_sh0/ckpt_best"
log "v4 FINAL: $V4_SH0"
if [[ ! -f "$V4_SH0/checkpoint.pt" ]]; then
  # fallback: use latest step
  LATEST=$(ls -d "${OUT_BASE}_sh0/ckpt_step"* 2>/dev/null | sort -V | tail -1)
  if [[ -n "$LATEST" && -f "$LATEST/checkpoint.pt" ]]; then
    V4_SH0="$LATEST"
    log "v4 fallback to latest step: $V4_SH0"
  else
    log "FATAL: no v4 checkpoint found"
    exit 2
  fi
fi

###############################################################################
# Step 1: v4 FINAL full eval (4 GPU, 4 shard, ~3.6h)
###############################################################################
log "=== Step 1: v4 FINAL full eval (4 GPU x 4 shard) ==="
TAG_V4F="SOTAV4_R1_FINAL"
PIDS=""
for SH in 0 1 2 3; do
  eval_shard $SH "$V4_SH0" "$TAG_V4F" $SH &
  PIDS="$PIDS $!"
done
log "v4 eval PIDs: $PIDS"
wait
log "v4 eval done at $(date)"
aggregate_csvs "$TAG_V4F" 2>&1 | tee -a "$CHAINLOG"

###############################################################################
# Step 2: Ablation experiments
#
#   实验设计（4 GPU 充分利用）：
#   2a (GPU0): v4 ckpt_best shard0 eval — 对比 FINAL vs BEST（early stopping 效应）
#   2b (GPU1): v4 FINAL + safety_net shard0 eval — 看 fallback 对 PDMS 的提升
#   2c (GPU2): SFT r=0.355 shard0 eval — 验证 matched-compute baseline
#   2d (GPU3): v3 old ckpt shard0 eval — 直接量化 v3(eff_beta=0.15) vs v4(eff_beta=0.05)
#   2e (CPU):  训练曲线分析
###############################################################################
log "=== Step 2: Ablation experiments (4 GPU full) ==="

# 2a: v4 ckpt_best eval
if [[ -f "$V4_BEST/checkpoint.pt" ]]; then
  log "Launch v4 BEST eval (GPU0)"
  eval_shard 0 "$V4_BEST" "SOTAV4_R1_BEST" 0 &
  PID_BEST=$!
else
  log "SKIP v4 BEST: no ckpt_best"
  PID_BEST=""
fi

# 2b: v4 FINAL + safety_net eval
TAG_SAFE="SOTAV4_R1_SAFENET"
EXP_SAFE="${TAG_SAFE}_sh0"
CSV_SAFE="$OUTDIR/${EXP_SAFE}.csv"
if [[ -f "$CSV_SAFE" ]]; then
  log "SKIP $EXP_SAFE (csv exists)"
else
  log "Launch v4 FINAL + safety_net shard0 eval (GPU1)"
  (
    export CUDA_VISIBLE_DEVICES=1
    cd "$NAVSIM_ROOT"
    $PY navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$EXP_SAFE" \
      train_test_split="${PREFIX}0${SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" \
      +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false \
      +agent.selector=scorer_budget \
      +agent.scorer_ckpt="$V4_SH0" \
      +agent.keep_ratio=0.5 \
      +agent.prune_variant=drop +agent.prune_verbose=true \
      +agent.safety_net=true \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$LOGDIR/_${EXP_SAFE}.log" 2>&1
  found=$(ls -t "$NAVSIM_EXP_ROOT/$EXP_SAFE"/*/*.csv 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    cp -a "$found" "$CSV_SAFE"
    pdms=$($PY -c "import pandas as pd;d=pd.read_csv('$found');d=d[d['token']!='average'];print(f'{d[\"score\"].mean():.5f}')" 2>/dev/null)
    log "GPU1 DONE $EXP_SAFE: PDMS=$pdms"
  fi
) &
PID_SAFE=$!

# 2c: SFT scorer r=0.355 shard0 eval
SFT_CKPT="$ROOT/ckpt/s3_token_scorer"
log "Launch SFT r=0.355 shard0 eval (GPU2)"
eval_shard 2 "$SFT_CKPT" "SFT_r0355_sh0" 0 &
PID_SFT=$!

# 2d: v3 old ckpt (eff_beta=0.15) shard0 eval — 直接对比 v3 vs v4
V3_CKPT="$ROOT/ckpt/s3_token_scorer_budget_rl_20260803_151608_sh0"
if [[ -f "$V3_CKPT/checkpoint.pt" ]]; then
  log "Launch v3 old ckpt shard0 eval (GPU3)"
  eval_shard 3 "$V3_CKPT" "SOTAV3_R1_FINAL" 0 &
  PID_V3=$!
else
  log "SKIP v3: no ckpt"
  PID_V3=""
fi

# 2e: 训练曲线分析（CPU，后台）
log "Launch training curve analysis (CPU)"
(
  $PY - << 'PYEOF' 2>&1
import json, glob, statistics
for sh in range(4):
    pattern = f"/apdcephfs/private_shayladeng/tokenrl_autoVLA/ckpt/s3_token_scorer_budget_rl_20260804_*_sh{sh}/train_log.jsonl"
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
) > "$LOGDIR/analysis_step2.txt" 2>&1 &
PID_ANALYSIS=$!

wait ${PID_BEST:-} ${PID_SAFE:-} ${PID_SFT:-} ${PID_V3:-} ${PID_ANALYSIS:-}
log "Ablation evals done at $(date)"

aggregate_csvs "SOTAV4_R1_BEST" 2>&1 | tee -a "$CHAINLOG"
aggregate_csvs "SOTAV4_R1_SAFENET" 2>&1 | tee -a "$CHAINLOG"
aggregate_csvs "SFT_r0355_sh0" 2>&1 | tee -a "$CHAINLOG"
aggregate_csvs "SOTAV3_R1_FINAL" 2>&1 | tee -a "$CHAINLOG"

###############################################################################
# Step 3: kr distribution analysis from eval logs
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
for f in sorted(glob.glob("/apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/auto_chain_v4/_SOTAV4_R1_FINAL_sh*.log")):
    sh = f.split('_sh')[-1].replace('.log','')
    kr = extract_kr(f)
    all_kr.update(kr)
    vals = list(kr.values())
    if vals:
        print(f"  sh{sh}: N={len(vals)} mean_kr={statistics.mean(vals):.4f} median_kr={statistics.median(vals):.4f} std={statistics.pstdev(vals):.4f}")

vals_all = list(all_kr.values())
if vals_all:
    print(f"\n  OVERALL: N={len(vals_all)} mean_kr={statistics.mean(vals_all):.4f} median_kr={statistics.median(vals_all):.4f}")
    bins = [(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.9)]
    for lo, hi in bins:
        cnt = sum(1 for v in vals_all if lo <= v < hi)
        print(f"    kr [{lo:.1f},{hi:.1f}): {cnt:>5} ({100*cnt/len(vals_all):.1f}%)")
    with open("/apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/auto_chain_v4/kr_distribution.json", "w") as f:
        json.dump({"mean": statistics.mean(vals_all), "median": statistics.median(vals_all), 
                   "std": statistics.pstdev(vals_all), "N": len(vals_all)}, f)
else:
    print("  NO kr data found")
PYEOF

###############################################################################
# Step 4: Denylist reconstruction (offline) + Final report
###############################################################################
log "=== Step 4: Denylist reconstruction + Final report ==="

$PY - << 'PYEOF' 2>&1 | tee -a "$CHAINLOG"
import glob, pandas as pd, json
from datetime import datetime

def load_csvs(tag):
    rows = []
    for f in sorted(glob.glob(f"/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/{tag}_sh*.csv")):
        try:
            d = pd.read_csv(f); d = d[d['token'] != 'average']
            nc = d['num_penalties_collisions'].mean() if 'num_penalties_collisions' in d.columns else float('nan')
            ep = d['ego_progress'].mean() if 'ego_progress' in d.columns else float('nan')
            rows.append({'N': len(d), 'PDMS': d['score'].mean(), 'NC': nc, 'EP': ep, 'df': d})
        except Exception as e: 
            print(f"  WARN: {f}: {e}")
    if rows:
        tn = sum(r['N'] for r in rows)
        wp = sum(r['PDMS']*r['N'] for r in rows)/tn
        return {'N': tn, 'PDMS': round(wp,5), 'n_shards': len(rows), 'rows': rows}
    return None

# Load v4 FINAL raw
v4f = load_csvs("SOTAV4_R1_FINAL")
v4b = load_csvs("SOTAV4_R1_BEST")
v4safe = load_csvs("SOTAV4_R1_SAFENET")
v3old = load_csvs("SOTAV3_R1_FINAL")

# Offline denylist reconstruction for v4 FINAL
# 方法：加载 no-prune baseline (r=1.0) CSV，对 catastrophic tokens 做替换
# 使用已有的 catastrophic_tokens.json
v4_denylist_pdms = None
if v4f:
    try:
        import json as j
        with open("/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/varB_catastrophic_tokens.json") as f:
            catastrophic = j.load(f)
        cat_set = set(catastrophic.keys()) if isinstance(catastrophic, dict) else set(catastrophic)
        
        # Load no-prune baseline
        noprune_dfs = []
        for f in sorted(glob.glob("/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/tokenprune_S3_full/MT_attn_L12_r10_sh*.csv")):
            d = pd.read_csv(f); noprune_dfs.append(d)
        if noprune_dfs:
            noprune_all = pd.concat(noprune_dfs)
            noprune_all = noprune_all[noprune_all['token'] != 'average']
            noprune_map = dict(zip(noprune_all['token'], noprune_all['score']))
            
            # Replace catastrophic tokens with no-prune scores
            all_dfs = [r['df'] for r in v4f['rows']]
            all_v4 = pd.concat(all_dfs)
            all_v4 = all_v4[all_v4['token'] != 'average']
            
            replaced = 0
            scores_after = []
            for _, row in all_v4.iterrows():
                tok = row['token']
                if tok in cat_set and tok in noprune_map:
                    scores_after.append(noprune_map[tok])
                    replaced += 1
                else:
                    scores_after.append(row['score'])
            
            import statistics
            v4_denylist_pdms = round(statistics.mean(scores_after), 5)
            print(f"\n  [Offline Denylist] v4 FINAL + denylist: N={len(scores_after)} PDMS={v4_denylist_pdms} (replaced {replaced} catastrophic tokens)")
    except Exception as e:
        print(f"\n  [Offline Denylist] FAILED: {e}")

# ============================================================
# Final Report
# ============================================================
print("")
print("=" * 70)
print(f"  SOTA v4 Experiment Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)
print(f"{'Method':<40} {'N':>6} {'PDMS':>8} {'Δvs SFT':>10} {'Δvs no-prune':>13}")
print("-" * 70)

baseline_sft = 0.89008
baseline_noprune = 0.89886
baseline_sft355 = 0.85575

def print_row(name, r):
    if r is None: return
    d1 = r['PDMS'] - baseline_sft
    d2 = r['PDMS'] - baseline_noprune
    print(f"{name:<40} {r['N']:>6} {r['PDMS']:>8.5f} {d1:>+10.5f} {d2:>+13.5f}")

print_row("v4 Budget RL FINAL (eff_beta=0.05)", v4f)
print_row("v4 Budget RL BEST (eff_beta=0.05)", v4b)
print_row("v4 Budget RL + safety_net (shard0)", v4safe)
print_row("v3 Budget RL FINAL (eff_beta=0.15)", v3old)

if v4_denylist_pdms:
    d1 = v4_denylist_pdms - baseline_sft
    d2 = v4_denylist_pdms - baseline_noprune
    print(f"{'v4 Budget RL + denylist (offline)':<40} {v4f['N'] if v4f else '?':>6} {v4_denylist_pdms:>8.5f} {d1:>+10.5f} {d2:>+13.5f}")

# Baselines
print(f"{'SFT scorer r=0.5 (fixed)':<40} {11576:>6} {0.89008:>8.5f} {'(baseline)':>10} {0.89008-baseline_noprune:>+13.5f}")
print(f"{'SFT scorer r=0.355 (matched compute)':<40} {11576:>6} {0.85575:>8.5f} {0.85575-baseline_sft:>+10.5f} {0.85575-baseline_noprune:>+13.5f}")
print(f"{'no-prune (r=1.0, upper bound)':<40} {11576:>6} {0.89886:>8.5f} {0.89886-baseline_sft:>+10.5f} {'(upper)':>13}")
print("-" * 70)

# v3 vs v4 comparison
if v4f and v3old:
    print(f"\n  v3 (eff_beta=0.15) vs v4 (eff_beta=0.05): Δ={v4f['PDMS']-v3old['PDMS']:+.5f}")
    print(f"  → efficiency_beta 从 0.15 降到 0.05 的 PDMS 变化")

# Key findings
print()
if v4f:
    v4pdms = v4f['PDMS']
    print(f"  Key Result: v4 FINAL PDMS = {v4pdms:.5f}")
    print(f"  vs SFT r=0.5:  Δ = {v4pdms - baseline_sft:+.5f}")
    print(f"  vs SFT r=0.355: Δ = {v4pdms - baseline_sft355:+.5f}")
    print(f"  vs no-prune:   Δ = {v4pdms - baseline_noprune:+.5f}")
    
    if v4b:
        print(f"\n  v4 BEST vs FINAL: Δ = {v4b['PDMS'] - v4pdms:+.5f}")
    
    if v4_denylist_pdms:
        print(f"  v4 + denylist gain: Δ = {v4_denylist_pdms - v4pdms:+.5f}")
        print(f"  v4 + denylist vs SFT r=0.5: Δ = {v4_denylist_pdms - baseline_sft:+.5f}")
    
    # Performance categorization
    if v4pdms >= baseline_sft:
        print(f"\n  *** v4 SUCCESS: exceeds SFT r=0.5 baseline ***")
    elif v4pdms >= baseline_sft - 0.01:
        print(f"\n  *** v4 MARGINAL: within 0.01 of SFT r=0.5 ***")
    else:
        print(f"\n  *** v4 BELOW SFT r=0.5: need further improvement ***")
        print(f"  Suggested next steps:")
        print(f"    - Lower efficiency_beta further (0.02)")
        print(f"    - Add per-token credit assignment (ICLR PLAN §2)")
        print(f"    - Try 2-epoch training")

print("=" * 70)

# Save JSON report
report = {
    "timestamp": datetime.now().isoformat(),
    "v4_final": {"PDMS": v4f['PDMS'], "N": v4f['N']} if v4f else None,
    "v4_best": {"PDMS": v4b['PDMS'], "N": v4b['N']} if v4b else None,
    "v4_safenet": {"PDMS": v4safe['PDMS'], "N": v4safe['N']} if v4safe else None,
    "v4_denylist": v4_denylist_pdms,
    "baselines": {"sft_r05": 0.89008, "sft_r0355": 0.85575, "no_prune": 0.89886},
    "deltas": {}
}
if v4f:
    report['deltas']['v4_vs_sft_r05'] = round(v4f['PDMS'] - 0.89008, 5)
    report['deltas']['v4_vs_no_prune'] = round(v4f['PDMS'] - 0.89886, 5)
if v4_denylist_pdms:
    report['deltas']['denylist_gain'] = round(v4_denylist_pdms - v4f['PDMS'], 5) if v4f else None

with open("/apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/auto_chain_v4/final_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  JSON report: logs/auto_chain_v4/final_report.json")
PYEOF

log "=== Chain complete at $(date) ==="
log "Report: logs/auto_chain_v4/final_report.json"
log "CSVs:   results/raw/SOTAV4_R1_*.csv"
log "Logs:   logs/auto_chain_v4/"

###############################################################################
# Step 5: 7B nuScenes zero-shot eval (best-effort, ~2h)
#   利用汇总后的空闲 GPU，跑 ImpromptuVLA 7B + 7B scorer on nuScenes
#   r=0.5 only, max 1000 scenes, 超时自动终止
###############################################################################
log "=== Step 5: 7B nuScenes best-effort eval ==="
MODEL_7B="$ROOT/models/ImpromptuVLA_7B/7B_AD_finetune"
SCORER_7B="$ROOT/ckpt/s3_token_scorer_7b"
DATA_7B="$ROOT/code/third_party/ImpromptuVLA/nuscenes_test.json"
OUT_7B="$ROOT/results/impromptu7b"
EVAL_SCRIPT="$ROOT/scripts/run_impromptu7b_nuscenes_eval.py"
EVAL_METRICS="$ROOT/code/third_party/ImpromptuVLA/data_qa_generate/data_engine/datasets/nuscenes/scripts/evaluation_nuscenes.py"
mkdir -p "$OUT_7B"

# Preflight
if [[ ! -f "$MODEL_7B/config.json" ]]; then
  log "SKIP 7B: model not found"
elif [[ ! -f "$SCORER_7B/checkpoint.pt" ]]; then
  log "SKIP 7B: scorer not found"
elif [[ ! -f "$DATA_7B" ]]; then
  log "SKIP 7B: data not found"
else
  log "Launch 7B nuScenes eval (r=0.5, 1000 scenes, GPU0, timeout 7200s)"
  (
    export CUDA_VISIBLE_DEVICES=0
    cd "$ROOT"
    timeout 7200 $PY "$EVAL_SCRIPT" \
      --model-path "$MODEL_7B" \
      --scorer-ckpt "$SCORER_7B" \
      --keep-ratio 0.5 \
      --data-json "$DATA_7B" \
      --max-scenes 1000 \
      --device cuda:0 \
      --output "$OUT_7B/pred_r05_1k.jsonl" \
      > "$LOGDIR/_7b_nuscenes_r05.log" 2>&1
    rc=$?
    n=$(grep -c '"predict"' "$OUT_7B/pred_r05_1k.jsonl" 2>/dev/null || echo 0)
    log "7B r=0.5 done: rc=$rc preds=$n"
    
    # Run metrics if we got predictions
    if [[ "$n" -gt 0 && -f "$EVAL_METRICS" ]]; then
      log "Running 7B metrics..."
      cd "$(dirname "$EVAL_METRICS")"
      timeout 600 $PY "$EVAL_METRICS" \
        --jsonl_file "$OUT_7B/pred_r05_1k.jsonl" \
        --output_file "$OUT_7B/eval_r05_1k.json" \
        --mode x-y \
        > "$LOGDIR/_7b_metrics_r05.log" 2>&1 || true
      cd "$ROOT"
    fi
  ) &
  PID_7B=$!
  log "7B eval PID=$PID_7B"
  
  # Don't wait — let it run in background, will be killed at GPU reclaim
  # But log completion if it finishes before reclaim
  (
    wait $PID_7B 2>/dev/null
    log "7B eval finished at $(date)"
  ) &
fi

log "=== All chains launched at $(date) ==="
log "7B eval may still be running in background"
