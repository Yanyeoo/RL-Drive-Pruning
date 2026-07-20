#!/usr/bin/env bash
# run_taucut_shard0_quick.sh — τ-cut quick test on shard0 (Route B gate)
#
# Purpose: Test calibrated MSE scorer + global τ threshold adaptive pruning.
# Win condition: τ-cut @ mean_kr≈0.5 PDMS > fixed r=0.5 (0.8920)
#
# 4 τ values × shard0 = 4 jobs, 2 GPUs parallel → ~7h total.
# Launch: nohup bash scripts/run_taucut_shard0_quick.sh > logs/_taucut_shard0.log 2>&1 &
# Stop:   touch STOP_TAUCUT
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
OUT="$ROOT/results/raw/tokenprune_taucut"; mkdir -p "$OUT" logs
STOP="$ROOT/STOP_TAUCUT"
MSE_CKPT="$ROOT/ckpt/s3_token_scorer_mse"
SHARD_PREFIX="navtest_local_filtered_shard"; SHARD_SUFFIX="_20260616_154858"
NG=$(nvidia-smi -L | wc -l); [[ "$NG" -lt 1 ]] && NG=1
log(){ echo "[taucut $(date +%H:%M:%S)] $*"; }

# τ values calibrated for target mean_kr ≈ 0.40/0.50/0.60/0.70
# (from offline calibration on 500 navtest scenes with MSE scorer)
TAUS=("-0.1253" "-0.1487" "-0.1668" "-0.1840")
TAU_TAGS=("kr040" "kr050" "kr060" "kr070")

# Build job list: 4 τ values × shard0 only (quick test)
JOBS=()
for i in "${!TAUS[@]}"; do
    JOBS+=("${TAUS[$i]} ${TAU_TAGS[$i]} 0")
done

run_job(){
  local gpu="$1" tau="$2" tag="$3" sh="$4"
  local exp="TC_mse_tau_${tag}_sh${sh}"
  local csv="$OUT/${exp}.csv"
  [[ -f "$csv" ]] && { log "SKIP $exp (done)"; return; }
  [[ -f "$STOP" ]] && { log "STOP -> skip $exp"; return; }
  log "GPU$gpu start $exp (tau=$tau)"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
    timeout 40000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" train_test_split="${SHARD_PREFIX}${sh}${SHARD_SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false +agent.keep_ratio=0.5 +agent.selector=scorer_taucut \
      +agent.scorer_ckpt="$MSE_CKPT" +agent.tau="$tau" +agent.tau_min_keep=36 \
      +agent.prune_verbose=false worker=single_machine_thread_pool worker.max_workers=1
  ) > "logs/_tc_${exp}.log" 2>&1; local rc=$?
  local c; c=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  [[ -n "$c" ]] && cp -a "$c" "$csv" && log "GPU$gpu done $exp rc=$rc -> csv" || log "GPU$gpu $exp rc=$rc WARN no csv"
}

worker(){ local gpu="$1"; local i="$gpu"
  while [[ "$i" -lt "${#JOBS[@]}" ]]; do
    [[ -f "$STOP" ]] && { log "STOP -> worker$gpu exit"; return; }
    run_job "$gpu" ${JOBS[$i]}
    i=$(( i + NG ))
  done
}

if pgrep -f run_pdm_score_cot >/dev/null; then log "ABORT: pdm_score already running"; exit 1; fi
log "τ-cut shard0 quick test: ${#JOBS[@]} jobs over $NG GPUs. τ=${TAUS[*]}. out=$OUT"
for g in $(seq 0 $((NG-1))); do worker "$g" & done
wait
log "τ-cut dispatch finished. csvs: $(ls "$OUT"/TC_*.csv 2>/dev/null | wc -l)/${#JOBS[@]}"

# Aggregate results
log "Aggregating results..."
"$PY" -c "
import pandas as pd, glob, os, numpy as np
out_dir = '$OUT'
csvs = sorted(glob.glob(f'{out_dir}/TC_mse_tau_*_sh0.csv'))
print(f'\\n=== τ-cut shard0 results ({len(csvs)} arms) ===')
print(f'Win condition: PDMS > 0.8920 (fixed scorer r=0.5 full navtest)')
print(f'Reference: scorer r=0.5 shard0 PDMS ≈ 0.8918 (from MT_scorer_r05_sh0.csv)')
print()
for csv in csvs:
    name = os.path.basename(csv).replace('.csv','')
    df = pd.read_csv(csv)
    df = df[df['token'] != 'average']
    pdms = df['score'].mean()
    n = len(df)
    # Compute actual mean keep ratio from prune stats if available
    print(f'  {name}: N={n}, PDMS={pdms:.6f}')
print()
print('Comparison:')
print('  Fixed scorer r=0.5 (full navtest): PDMS=0.8920')
print('  Fixed scorer r=0.75 (full navtest): PDMS=0.8983')
print('  No-prune r=1.0 (full navtest): PDMS=0.8988')
" 2>&1 | tee "$OUT/_taucut_shard0_summary.txt"
