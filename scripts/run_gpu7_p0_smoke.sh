#!/usr/bin/env bash
# run_gpu7_p0_smoke.sh — GPU7 专属：P0 Appendix baseline 烟测 (SparseVLM / PruMerge)
# 仅当 GPU7 空闲时由监控循环启动（不与其他 GPU 争用）。每个 selector 跑 sh0 烟测，
# 通过后追加 full(4 sh) 以占满窗口。产物用 cp -a 不覆盖原文件。
# 注意：P0 为 training-free Appendix baseline，无需 scorer_ckpt（scorer/taucut 才需要）。
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
log(){ echo "[gpu7-p0 $(date +%H:%M:%S)] $*"; }

run_one(){
  local gpu="$1" exp="$2" sh="$3" kr="$4" sel="$5"
  local csv
  case "$sel" in
    scorer_taucut) csv="$ROOT/results/raw/tokenprune_taucut/${exp}.csv" ;;
    *) csv="$ROOT/results/raw/tokenprune_S3_full/${exp}.csv" ;;
  esac
  [[ -f "$csv" ]] && { log "SKIP $exp (csv exists)"; return; }
  [[ -f "$STOP" ]] && { log "STOP -> skip $exp"; return; }
  log "GPU$gpu START $exp (kr=$kr sel=$sel)"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
    timeout 40000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" train_test_split="${SHARD_PREFIX}${sh}${SHARD_SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false +agent.keep_ratio="$kr" +agent.selector="$sel" \
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

# smoke: sh0 for both P0 selectors at r=0.5
run_one 7 "MT_sparsevlm_text_r05_sh0" 0 0.5 sparsevlm_text
run_one 7 "MT_prumerge_cls_r05_sh0"   0 0.5 prumerge_cls
# if smoke OK, continue full to fill the window
for sh in 1 2 3; do
  run_one 7 "MT_sparsevlm_text_r05_sh$sh" "$sh" 0.5 sparsevlm_text
  run_one 7 "MT_prumerge_cls_r05_sh$sh"   "$sh" 0.5 prumerge_cls
done
log "GPU7 P0 smoke+full done."
