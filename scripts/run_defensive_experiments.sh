#!/usr/bin/env bash
# run_defensive_experiments.sh — 防守性实验：SFT-based adaptive pruning + baselines
# 分支B：v6 shard0 quick eval PDMS < 0.88 时执行
# 使用全部8卡，按优先级跑：
#   P0: τ-cut kr050/kr070 full 4 shard (补全adaptive curve)
#   P1: MSE scorer r=0.5/0.75 full 4 shard (LambdaRank vs MSE ablation)
#   P2: Matched-compute SFT r=0.355 full 4 shard
#   P3: SparseVLM/PruMerge baseline full (Appendix)
#
# 用法: nohup bash scripts/run_defensive_experiments.sh > logs/defensive_master.log 2>&1 &
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
PREFIX="navtest_local_filtered_shard"; SUFFIX="_20260616_154858"
OUTDIR="$ROOT/results/raw"
LOGDIR="$ROOT/logs/defensive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

log(){ echo "[defensive $(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/master.log"; }

log "=== Defensive experiments started ==="
log "8 GPU, window until ~24:00"

###############################################################################
# eval_shard: run one shard on one GPU, save CSV
###############################################################################
eval_shard(){
  local gpu="$1" exp="$2" sh="$3" selector="$4" kr="$5"
  local extra_flags="${6:-}"
  local csv="$OUTDIR/${exp}.csv"
  local jlog="$LOGDIR/_${exp}.log"
  if [[ -f "$csv" ]]; then
    log "SKIP $exp (csv exists)"
    return
  fi
  log "GPU$gpu START $exp (sel=$selector kr=$kr sh=$sh)"
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
      +agent.keep_ratio="$kr" +agent.selector="$selector" \
      +agent.prune_variant=drop +agent.prune_verbose=false \
      $extra_flags \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$jlog" 2>&1
  local found; found=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    cp -a "$found" "$csv"
    local pdms; pdms=$($PY -c "import pandas as pd;d=pd.read_csv('$found');d=d[d['token']!='average'];print(f'{d[\"score\"].mean():.5f}')" 2>/dev/null)
    log "GPU$gpu DONE  $exp: PDMS=$pdms"
  else
    log "GPU$gpu FAIL  $exp: no CSV"
  fi
}

###############################################################################
# P0: τ-cut kr050/kr070 full 4 shard (补全adaptive curve)
# τ values: kr050=-0.1487, kr070=-0.1840 (verified from journal)
###############################################################################
log "=== P0: τ-cut adaptive full curve ==="

# kr050: sh1,2,3 (sh0 already exists as TC_mse_tau_kr050_sh0.csv)
for sh in 1 2 3; do
  eval_shard "$sh" "DEF_TC_mse_tau_kr050_sh${sh}" "$sh" "scorer_taucut" "0.5" \
    "+agent.scorer_ckpt=$ROOT/ckpt/s3_token_scorer_mse +agent.tau=-0.1487 +agent.tau_min_keep=36" &
done
wait  # wait for GPU1-3 to finish kr050

# kr070: sh1,2,3
for sh in 1 2 3; do
  eval_shard "$sh" "DEF_TC_mse_tau_kr070_sh${sh}" "$sh" "scorer_taucut" "0.5" \
    "+agent.scorer_ckpt=$ROOT/ckpt/s3_token_scorer_mse +agent.tau=-0.1840 +agent.tau_min_keep=36" &
done
wait

log "=== P0 DONE ==="

###############################################################################
# P1: MSE scorer r=0.5/0.75 full 4 shard
###############################################################################
log "=== P1: MSE scorer ablation ==="

# r=0.5 full
for sh in 0 1 2 3; do
  eval_shard "$sh" "DEF_MSE_scorer_r05_sh${sh}" "$sh" "scorer" "0.5" \
    "+agent.scorer_ckpt=$ROOT/ckpt/s3_token_scorer_mse" &
done
wait

# r=0.75 full
for sh in 0 1 2 3; do
  eval_shard "$sh" "DEF_MSE_scorer_r075_sh${sh}" "$sh" "scorer" "0.75" \
    "+agent.scorer_ckpt=$ROOT/ckpt/s3_token_scorer_mse" &
done
wait

log "=== P1 DONE ==="

###############################################################################
# P2: Matched-compute SFT r=0.355 (same FLOPs as Budget RL dynamic)
###############################################################################
log "=== P2: Matched-compute baseline ==="
for sh in 0 1 2 3; do
  eval_shard "$sh" "DEF_SFT_r0355_sh${sh}" "$sh" "scorer" "0.355" \
    "+agent.scorer_ckpt=$ROOT/ckpt/s3_token_scorer" &
done
wait
log "=== P2 DONE ==="

###############################################################################
# P3: SparseVLM/PruMerge baselines full (Appendix, if time permits)
###############################################################################
log "=== P3: Training-free baselines (if time) ==="
for sh in 0 1 2 3; do
  eval_shard "$sh" "DEF_sparsevlm_r05_sh${sh}" "$sh" "sparsevlm_text" "0.5" "" &
done
for sh in 0 1 2 3; do
  eval_shard "$sh" "DEF_prumerge_r05_sh${sh}" "$sh" "prumerge_cls" "0.5" "" &
done
wait
log "=== P3 DONE ==="

###############################################################################
# Aggregate
###############################################################################
log "=== Aggregating all results ==="
$PY - <<'PYEOF'
import glob, os, pandas as pd
from collections import defaultdict

groups = defaultdict(list)
for f in sorted(glob.glob("/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/DEF_*_sh*.csv")):
    key = os.path.basename(f).rsplit('_sh', 1)[0]
    groups[key].append(f)

print("\n=== Defensive Experiment Results ===")
for key in sorted(groups.keys()):
    fs = groups[key]
    dfs = []
    for f in fs:
        df = pd.read_csv(f)
        df = df[df['token'].notna() & (df['token'] != 'average')]
        dfs.append(df)
    all_df = pd.concat(dfs).drop_duplicates('token')
    valid = all_df[all_df['valid']]
    print(f"  {key}: shards={len(fs)} N={len(all_df)} valid={len(valid)} PDMS={valid['score'].mean():.6f}")
PYEOF

log "=== ALL DONE at $(date) ==="
