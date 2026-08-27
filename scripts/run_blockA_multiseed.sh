#!/usr/bin/env bash
# run_blockA_multiseed.sh — 2026-08-03 周期 Block A
# 目的：回应 AAAI Reviewer #4「统计证据不足」
#
# 事实依据：ckpt/s3_token_scorer_budget_rl_20260722_155943_sh{0..7} 是 8 个独立训练 run
#   - seed = 42 + SH（各不同，见 scripts/run_budget_rl_navtrain.sh:81）
#   - 每个训练在 navtrain 的 1/8 分片上（--num-shards 8 --shard-id SH）
#   - 8 个 ckpt_best/budget_params.pt 的 md5 全部互异（已核验）
#   - 历史所有 eval 只用了 sh0（run_budget_rl_navtrain.sh:56 明确注释）→ 从未报告过 variance
#
# 本 Block：8×H20 并行，GPU i 用 ckpt sh_i，全部在 navtest shard0 (2949 scenes) 上 eval
#           → 得到 8 个独立 PDMS → mean ± std
# 协议：raw（无 entropy fallback、无 denylist），与 SUPP_budgetrl_dynamic_raw 一致，可直接对齐
# 预计：~3.6h（单 GPU 2949 scenes @ ~4.4s/scene）
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh" >/dev/null
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SPLIT="navtest_local_filtered_shard0_20260616_154858"
OUTDIR="$ROOT/results/raw/blockA_multiseed"
LOGDIR="$ROOT/logs/blockA_multiseed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR" "$LOGDIR"
echo "$LOGDIR" > "$ROOT/logs/blockA_logdir.txt"

log(){ echo "[blockA $(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/run.log"; }

worker(){
  local gpu="$1" seedidx="$2"
  local bckpt="$ROOT/ckpt/s3_token_scorer_budget_rl_20260722_155943_sh${seedidx}/ckpt_best"
  local exp="MSEED_budgetrl_s${seedidx}_nt0"
  local csv="$OUTDIR/${exp}.csv"
  local jlog="$LOGDIR/_${exp}.log"
  if [[ -f "$csv" ]]; then log "[GPU$gpu] SKIP $exp (csv exists)"; return; fi
  if [[ ! -d "$bckpt" ]]; then log "[GPU$gpu] FAIL $exp (ckpt missing: $bckpt)"; return; fi
  log "[GPU$gpu] START $exp  ckpt=sh${seedidx} split=$SPLIT"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
    timeout 40000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" \
      train_test_split="$SPLIT" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" \
      +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false \
      +agent.keep_ratio=0.5 +agent.selector=scorer_budget \
      +agent.scorer_ckpt="$bckpt" \
      +agent.prune_variant=drop +agent.prune_verbose=true \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$jlog" 2>&1
  local rc=$?
  local found; found=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    cp -a "$found" "$csv"
    local pdms; pdms=$($PY -c "import pandas as pd;d=pd.read_csv('$csv');d=d[d['token']!='average'];print(f'{d[\"score\"].mean():.5f} N={len(d)}')" 2>/dev/null)
    log "[GPU$gpu] DONE $exp rc=$rc PDMS=$pdms"
  else
    log "[GPU$gpu] FAIL $exp rc=$rc (no csv)"
  fi
}

log "=== Block A: 8 independent Budget RL runs on navtest shard0 (raw protocol) ==="
PIDS=()
for i in 0 1 2 3 4 5 6 7; do
  worker "$i" "$i" &
  PIDS+=($!)
done
printf '%s\n' "${PIDS[@]}" > "$LOGDIR/pids.txt"
log "PIDs: ${PIDS[*]}"
wait "${PIDS[@]}"

log "=== Block A DONE — aggregating ==="
"$PY" - <<EOF 2>&1 | tee -a "$LOGDIR/run.log"
import glob, os, statistics, pandas as pd
rows=[]
for f in sorted(glob.glob("$OUTDIR/MSEED_budgetrl_s*_nt0.csv")):
    d=pd.read_csv(f); d=d[d['token']!='average']
    rows.append((os.path.basename(f), len(d), d['score'].mean(),
                 d['no_at_fault_collisions'].mean(), d['ego_progress'].mean()))
print("\n=== Block A: 8 independent Budget RL runs, navtest shard0 (raw) ===")
for n,N,p,nc,ep in rows:
    print(f"  {n:34s} N={N:5d} PDMS={p:.5f} NC={nc:.4f} EP={ep:.4f}")
if len(rows)>1:
    v=[r[2] for r in rows]
    print(f"\n  n_runs={len(v)}  mean={statistics.mean(v):.5f}  std={statistics.stdev(v):.5f}"
          f"  min={min(v):.5f}  max={max(v):.5f}")
    print(f"  95% CI (t, n-1) approx: {statistics.mean(v):.5f} +/- {2.365*statistics.stdev(v)/len(v)**0.5:.5f}")
EOF
log "=== Logs: $LOGDIR ==="
