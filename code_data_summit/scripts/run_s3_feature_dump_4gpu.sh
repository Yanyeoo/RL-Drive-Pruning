#!/usr/bin/env bash
# run_s3_feature_dump_4gpu.sh — S3 navtrain per-token feature dump, 4x H20.
# Each GPU takes shard i of (shard_stride=4), capped at PER_SHARD tokens.
# Total dumped ~= 4 * PER_SHARD scenes.  Stop: touch STOP_S3_FEAT
# Launch: nohup bash scripts/run_s3_feature_dump_4gpu.sh > logs/_s3_featdump.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="${ROOT}/code/third_party/AutoVLA"; NAVSIM_ROOT="${AUTOVLA_ROOT}/navsim"
source "${ROOT}/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="${ROOT}/code:${NAVSIM_ROOT}:${AUTOVLA_ROOT}:${PYTHONPATH:-}"
SAVE="${ROOT}/data/s3_scorer/features"; mkdir -p "$SAVE" logs
JSON="${ROOT}/data/navtrain_nocot"
PER_SHARD="${PER_SHARD:-1000}"
STOP="${ROOT}/STOP_S3_FEAT"
log(){ echo "[s3feat $(date +%H:%M:%S)] $*"; }

if pgrep -f "run_feature_dump" >/dev/null; then log "ABORT: run_feature_dump already running"; exit 1; fi
if pgrep -f "run_pdm_score_cot" >/dev/null; then log "ABORT: pdm_score running"; exit 1; fi

worker(){
  local g="$1"
  [[ -f "$STOP" ]] && { log "STOP -> gpu$g abort"; return; }
  log "GPU$g start shard $g/4 per_shard=$PER_SHARD"
  cd "$NAVSIM_ROOT"
  CUDA_VISIBLE_DEVICES="$g" timeout 25000 "$PY" -m rldrive.scoring.run_feature_dump \
    --save-dir "$SAVE" --gpu "$g" --json-dir "$JSON" \
    --shard-stride 4 --shard-index "$g" --max-scenes "$PER_SHARD" --skip-done \
    > "${ROOT}/logs/_s3_featdump_g${g}.log" 2>&1
  log "GPU$g done rc=$?"
}

log "S3 feature dump start. 4 GPUs, per_shard=$PER_SHARD -> ~$((4*PER_SHARD)) scenes. save=$SAVE"
for g in 0 1 2 3; do worker "$g" & done
wait
log "S3 feature dump: all GPUs finished. n_pt=$(ls "$SAVE"/*.pt 2>/dev/null | wc -l)"
