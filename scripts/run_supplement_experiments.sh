#!/usr/bin/env bash
# run_supplement_experiments.sh — AAAI 补充材料紧急实验 (8.1 ddl)
# 4×H20 并行，每个GPU串行跑自己的4个shard
# 今晚19:00回收 → 重启后自动续跑 (SKIP已有csv)
#
# GPU0: Budget RL dynamic RAW (no fallback) — 审阅意见#1: 关fallback证明方法本身有效
# GPU1: SFT scorer r=0.355 RAW (matched-compute) — 审阅意见#2: 同平均保留率对照
# GPU2: SparseVLM r=0.5 + SAME fallback — 审阅意见#1: 统一协议
# GPU3: FastV/PruMerge full navtest upgrade sh1/2/3 — 审阅意见#1: baseline全量
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh" >/dev/null
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SHARD_PREFIX="navtest_local_filtered_shard"
SHARD_SUFFIX="_20260616_154858"
OUTDIR="$ROOT/results/raw/tokenprune_S3_full"
LOGDIR="$ROOT/logs/supplement_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR" "$LOGDIR"

SFT_CKPT="$ROOT/ckpt/s3_token_scorer"
BUDGET_RL_CKPT="$ROOT/ckpt/s3_token_scorer_budget_rl_20260722_155943_sh0/ckpt_best"
DENYLIST="$ROOT/results/varB_catastrophic_tokens.json"

log(){ echo "[supp $(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/run.log"; }

# ============================================================
# Per-GPU worker: runs 4 shards sequentially on one GPU
# ============================================================
worker(){
  local gpu="$1"; shift
  local -a jobs=("$@")
  for job in "${jobs[@]}"; do
    IFS='|' read -r sel kr exp sh extra <<< "$job"
    local csv="$OUTDIR/${exp}.csv"
    local jlog="$LOGDIR/_${exp}.log"
    if [[ -f "$csv" ]]; then
      log "[GPU$gpu] SKIP $exp (csv exists)"
      continue
    fi
    log "[GPU$gpu] START $exp (sel=$sel kr=$kr sh=$sh)"
    ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
      timeout 30000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
        experiment_name="$exp" \
        train_test_split="${SHARD_PREFIX}${sh}${SHARD_SUFFIX}" \
        metric_cache_path="$ROOT/data/navtest_metric_cache" \
        +json_data_path="$ROOT/data/navtest_nocot" \
        agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
        +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" \
        +agent.sensor_data_path="$SENSOR" \
        +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false \
        +agent.keep_ratio="$kr" +agent.selector="$sel" \
        +agent.prune_variant=drop +agent.prune_verbose=false \
        $extra \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$jlog" 2>&1
    local rc=$?
    local found; found=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
    if [[ -n "$found" ]]; then
      cp -a "$found" "$csv"
      local rows=$(($(wc -l < "$csv")-1))
      local pdms; pdms=$($PY -c "import pandas as pd; df=pd.read_csv('$csv'); df=df[df['token']!='average']; print(f'{df[\"score\"].mean():.5f}')" 2>/dev/null)
      log "[GPU$gpu] DONE $exp rc=$rc rows=$rows PDMS=$pdms"
    else
      log "[GPU$gpu] FAIL $exp rc=$rc (no csv)"
    fi
  done
}

# ============================================================
# Job definitions per GPU (pipe-delimited: sel|kr|exp_name|shard|extra_hydra)
# ============================================================

GPU0_JOBS=()
GPU1_JOBS=()
GPU2_JOBS=()
GPU3_JOBS=()

# GPU0: Budget RL dynamic RAW (no fallback, no denylist) — 审阅意见#1
for sh in 0 1 2 3; do
  GPU0_JOBS+=("scorer_budget|0.5|SUPP_budgetrl_dynamic_raw_sh${sh}|${sh}|+agent.scorer_ckpt=$BUDGET_RL_CKPT")
done

# GPU1: SFT scorer r=0.355 RAW (matched-compute vs 35.5%) — 审阅意见#2
for sh in 0 1 2 3; do
  GPU1_JOBS+=("scorer|0.355|SUPP_sft_r0355_raw_sh${sh}|${sh}|+agent.scorer_ckpt=$SFT_CKPT")
done

# GPU2: SparseVLM r=0.5 + SAME fallback (协议统一) — 审阅意见#1
for sh in 0 1 2 3; do
  GPU2_JOBS+=("sparsevlm_text|0.5|SUPP_sparsevlm_r05_fallback_sh${sh}|${sh}|+agent.varB_denylist=$DENYLIST +agent.safety_net=true")
done

# GPU3: Baseline full navtest upgrade (FastV/PruMerge sh1/2/3, drop, raw)
for sel in "fastv_l2" "prumerge_cls"; do
  for kr in "0.5" "0.75" "0.25"; do
    rtag="${kr/./}"
    for sh in 1 2 3; do
      GPU3_JOBS+=("${sel}|${kr}|SUPP_${sel}_drop_r${rtag}_sh${sh}|${sh}|")
    done
  done
done

# ============================================================
# Launch all 4 workers in background
# ============================================================
log "=== Launching 4 GPU workers ==="
log "GPU0: ${#GPU0_JOBS[@]} jobs (Budget RL dynamic raw)"
log "GPU1: ${#GPU1_JOBS[@]} jobs (SFT r=0.355 raw, matched-compute)"
log "GPU2: ${#GPU2_JOBS[@]} jobs (SparseVLM r=0.5 + fallback)"
log "GPU3: ${#GPU3_JOBS[@]} jobs (FastV/PruMerge sh1/2/3 upgrade)"

worker 0 "${GPU0_JOBS[@]}" &
PID0=$!
worker 1 "${GPU1_JOBS[@]}" &
PID1=$!
worker 2 "${GPU2_JOBS[@]}" &
PID2=$!
worker 3 "${GPU3_JOBS[@]}" &
PID3=$!

log "PIDs: GPU0=$PID0 GPU1=$PID1 GPU2=$PID2 GPU3=$PID3"
echo "$PID0 $PID1 $PID2 $PID3" > "$LOGDIR/pids.txt"

wait $PID0 $PID1 $PID2 $PID3

# ============================================================
# Final report
# ============================================================
log "=== ALL WORKERS DONE ==="
$PY -c "
import pandas as pd, glob, os
outdir = '$OUTDIR'
print('\n=== Supplement CSVs Summary ===')
for f in sorted(glob.glob(outdir + '/SUPP_*.csv')):
    n = os.path.getsize(f)
    try:
        df = pd.read_csv(f); df = df[df['token']!='average']
        pdms = df['score'].mean()
        nc = df['no_at_fault_collisions'].mean()
        ep = df['ego_progress'].mean()
        print(f'  {os.path.basename(f):55s} N={len(df):5d}  PDMS={pdms:.5f}  NC={nc:.4f}  EP={ep:.4f}')
    except Exception as e:
        print(f'  {os.path.basename(f):55s} (error: {e})')
" 2>/dev/null

log "=== Done. Logs: $LOGDIR/run.log ==="
