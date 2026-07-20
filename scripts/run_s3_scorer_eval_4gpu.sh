#!/usr/bin/env bash
# run_s3_scorer_eval_4gpu.sh — evaluate the trained S3 token scorer as a
# pruning selector on the SAME navtest subset as S2, for apples-to-apples vs
# attn_L12 (docs/specs/s3_token_scorer_spec.md §5).
#   GPU0: scorer r=0.25   GPU1: scorer r=0.5   GPU2: scorer r=0.75
# CSVs -> results/raw/tokenprune_S3/.  Stop: touch STOP_S3_EVAL
# Launch: nohup bash scripts/run_s3_scorer_eval_4gpu.sh > logs/_s3_eval.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="${ROOT}/code/third_party/AutoVLA"; NAVSIM_ROOT="${AUTOVLA_ROOT}/navsim"
source "${ROOT}/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="${ROOT}/code:${NAVSIM_ROOT}:${AUTOVLA_ROOT}:${PYTHONPATH:-}"
CKPT="${ROOT}/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
TRAIN_YAML="${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR_DATA="${ROOT}/data/navsim_v2_local"
SCORER="${SCORER:-${ROOT}/ckpt/s3_token_scorer}"
SPLIT="${SPLIT:-navtest_s2sub1500_shard0}"
PREFIX="${PREFIX:-S3sub1500}"
OUTDIR="${ROOT}/results/raw/tokenprune_S3"; mkdir -p "$OUTDIR" logs
STOP="${ROOT}/STOP_S3_EVAL"
log(){ echo "[s3eval $(date +%H:%M:%S)] $*"; }

if pgrep -f run_pdm_score_cot >/dev/null; then log "ABORT: pdm_score already running"; exit 1; fi

run_arm(){
  local gpu="$1" kr="$2" exp="$3"
  [[ -f "$STOP" ]] && { log "STOP -> skip $exp"; return; }
  log "GPU$gpu start $exp (r=$kr)"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
    timeout 30000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" train_test_split="$SPLIT" \
      metric_cache_path="${ROOT}/data/navtest_metric_cache" \
      +json_data_path="${ROOT}/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$TRAIN_YAML" +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="$SENSOR_DATA" \
      +agent.codebook_cache_path="${AUTOVLA_ROOT}/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false \
      +agent.keep_ratio="$kr" +agent.selector=scorer +agent.scorer_ckpt="$SCORER" \
      +agent.prune_verbose=false \
      worker=single_machine_thread_pool worker.max_workers=1
  ); local rc=$?
  log "GPU$gpu done $exp rc=$rc"
  local c; c=$(ls -t "${NAVSIM_EXP_ROOT}/${exp}"/*/*.csv 2>/dev/null | head -1)
  [[ -n "$c" ]] && cp -a "$c" "${OUTDIR}/${exp}.csv" && log "csv -> ${OUTDIR}/${exp}.csv" || log "WARN no csv $exp"
}

[[ -f "${SCORER}/checkpoint.pt" ]] || { log "FATAL: scorer ckpt not found at $SCORER"; exit 2; }
log "S3 scorer eval. split=$SPLIT scorer=$SCORER"
run_arm 0 0.25 "${PREFIX}_scorer_r025" &
run_arm 1 0.5  "${PREFIX}_scorer_r050" &
run_arm 2 0.75 "${PREFIX}_scorer_r075" &
wait
log "S3 scorer eval: all arms finished. csvs:"; ls -la "${OUTDIR}/${PREFIX}"_*.csv 2>/dev/null
