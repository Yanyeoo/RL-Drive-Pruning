#!/usr/bin/env bash
# run_gpu7_taucut_kr040.sh — GPU7 专属：补齐 τ-cut kr040 sh1-3（完成 4 点 τ-cut 曲线）
# 与 run_21h_master.sh 互斥：本脚本只在 GPU7 跑，不触发 pgrep 双开守护，
# 但每个 job 跑前检查 CSV 已存在则 SKIP（幂等，可安全重跑）。
# 停止：touch STOP_21H
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SHARD_PREFIX="navtest_local_filtered_shard"; SHARD_SUFFIX="_20260616_154858"
LOGDIR="$ROOT/logs/_21h"; mkdir -p "$LOGDIR"
STOP="$ROOT/STOP_21H"
log(){ echo "[gpu7 $(date +%H:%M:%S)] $*"; }

run_one(){
  local gpu="$1" exp="$2" sh="$3" kr="$4" tau="$5" sel="$6" ckpt="$7"
  local csv="$ROOT/results/raw/tokenprune_taucut/${exp}.csv"
  [[ -f "$csv" ]] && { log "SKIP $exp (csv exists)"; return; }
  [[ -f "$STOP" ]] && { log "STOP -> skip $exp"; return; }
  log "GPU$gpu START $exp (kr=$kr sel=$sel tau=$tau)"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
    timeout 40000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" train_test_split="${SHARD_PREFIX}${sh}${SHARD_SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false +agent.keep_ratio="$kr" +agent.selector="$sel" \
      +agent.tau="$tau" +agent.tau_min_keep=36 +agent.scorer_ckpt="$ckpt" \
      +agent.prune_verbose=false worker=single_machine_thread_pool worker.max_workers=1
  ) > "$LOGDIR/_${exp}.log" 2>&1; local rc=$?
  local c; c=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  if [[ -n "$c" ]]; then
    cp -a "$c" "$csv"; log "GPU$gpu DONE $exp rc=$rc -> $csv"
    "$PY" -c "import pandas as pd; df=pd.read_csv('$csv'); print(f'[PDMS] $exp N={len(df)} PDMS={df[\"score\"].mean():.6f}')" 2>/dev/null || true
  else
    log "GPU$gpu $exp rc=$rc WARN no csv (see $LOGDIR/_${exp}.log)"
  fi
}

CKPT_MSE="$ROOT/ckpt/s3_token_scorer_mse"
for sh in 1 2 3; do
  run_one 7 "TC_mse_tau_kr040_sh$sh" "$sh" 0.4 -0.1253 scorer_taucut "$CKPT_MSE"
done
log "GPU7 τ-cut kr040 sh1-3 done. taucut_csvs=$(ls "$ROOT/results/raw/tokenprune_taucut"/TC_*.csv 2>/dev/null | wc -l)"
