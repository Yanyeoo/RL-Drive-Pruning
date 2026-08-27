#!/usr/bin/env bash
# run_supplement_batch2.sh — AAAI 补充材料第二批实验
# 在第一批完成后或19:00重启后运行
# GPU0: SFT scorer r=0.5 + denylist fallback (协议统一，补 raw→+fallback gap)
# GPU1: Attention teacher L12 r=0.5 + denylist fallback (审阅人要求的teacher baseline)
# GPU2: Random per-scene budget (同35.5%均值, matched-compute #2)
# GPU3: (继续跑第一批的FastV/PruMerge upgrade)
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
LOGDIR="$ROOT/logs/supplement_batch2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR" "$LOGDIR"

SFT_CKPT="$ROOT/ckpt/s3_token_scorer"
DENYLIST="$ROOT/results/varB_catastrophic_tokens.json"

log(){ echo "[supp2 $(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/run.log"; }

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
# GPU0: SFT scorer r=0.5 + denylist fallback (协议统一)
#       已有 raw=0.8920, 补 +fallback → 验证 fallback 增益来自方法还是denylist
# ============================================================
GPU0_JOBS=()
for sh in 0 1 2 3; do
  GPU0_JOBS+=("scorer|0.5|SUPP_sft_r05_fallback_sh${sh}|${sh}|+agent.scorer_ckpt=$SFT_CKPT +agent.varB_denylist=$DENYLIST +agent.safety_net=true")
done

# ============================================================
# GPU1: Attention teacher L12 r=0.5 + denylist fallback
#       审阅人明确问"teacher attention baseline 结果"
# ============================================================
GPU1_JOBS=()
for sh in 0 1 2 3; do
  GPU1_JOBS+=("attn_L12|0.5|SUPP_attnL12_r05_fallback_sh${sh}|${sh}|+agent.varB_denylist=$DENYLIST +agent.safety_net=true")
done

# ============================================================
# GPU2: SFT scorer r=0.5 RAW (full 4shard) — 如果第一批没跑
#       然后跑 SFT scorer r=0.25 RAW (补充Pareto前端)
# ============================================================
GPU2_JOBS=()
# SFT r=0.25 raw — 补 Pareto 前端 (审阅意见#7 要求 λ_e sweep / Pareto)
for sh in 0 1 2 3; do
  GPU2_JOBS+=("scorer|0.25|SUPP_sft_r025_raw_sh${sh}|${sh}|+agent.scorer_ckpt=$SFT_CKPT")
done
# SFT r=0.75 raw — 补 Pareto
for sh in 0 1 2 3; do
  GPU2_JOBS+=("scorer|0.75|SUPP_sft_r075_raw_sh${sh}|${sh}|+agent.scorer_ckpt=$SFT_CKPT")
done

# ============================================================
# Launch
# ============================================================
log "=== Launching batch2 (GPU0-2, 3 workers) ==="
log "GPU0: ${#GPU0_JOBS[@]} jobs (SFT r=0.5 + fallback)"
log "GPU1: ${#GPU1_JOBS[@]} jobs (Attn L12 r=0.5 + fallback)"
log "GPU2: ${#GPU2_JOBS[@]} jobs (SFT r=0.25/0.75 raw, Pareto)"

worker 0 "${GPU0_JOBS[@]}" &
PID0=$!
worker 1 "${GPU1_JOBS[@]}" &
PID1=$!
worker 2 "${GPU2_JOBS[@]}" &
PID2=$!

log "PIDs: GPU0=$PID0 GPU1=$PID1 GPU2=$PID2"
echo "$PID0 $PID1 $PID2" > "$LOGDIR/pids.txt"

wait $PID0 $PID1 $PID2

log "=== Batch2 ALL DONE ==="
$PY -c "
import pandas as pd, glob, os
outdir = '$OUTDIR'
print('\n=== Batch2 CSVs Summary ===')
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
