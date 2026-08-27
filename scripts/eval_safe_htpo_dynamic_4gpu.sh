#!/usr/bin/env bash
# Full NAVSIM evaluation for one trained Safe-HTPO dynamic policy.
# Runs physical Variant-B token drop on four disjoint navtest shards.
set -Eeuo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"
NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh" >/dev/null
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

: "${SAFE_HTPO_CHECKPOINT:?Set SAFE_HTPO_CHECKPOINT to the completed Safe-HTPO checkpoint directory}"
: "${SAFE_HTPO_RUN_ID:?Set SAFE_HTPO_RUN_ID to an immutable run identifier}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SHARD_PREFIX="navtest_local_filtered_shard"
SHARD_SUFFIX="_20260616_154858"
RESULT_DIR="$ROOT/results/raw/safe_htpo_${SAFE_HTPO_RUN_ID}"
LOG_DIR="$ROOT/logs/safe_htpo_${SAFE_HTPO_RUN_ID}/eval"
TIMEOUT_SEC="${SAFE_EVAL_TIMEOUT_SEC:-18000}"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

log() { echo "[safe-htpo-eval $(date '+%F %T')] $*"; }
fail() { log "FATAL: $*"; exit 1; }
[[ -f "$SAFE_HTPO_CHECKPOINT/checkpoint.pt" ]] || fail "checkpoint.pt not found: $SAFE_HTPO_CHECKPOINT"
[[ -f "$SAFE_HTPO_CHECKPOINT/feature_norm.pt" ]] || fail "feature_norm.pt not found: $SAFE_HTPO_CHECKPOINT"
[[ -f "$SAFE_HTPO_CHECKPOINT/budget_params.pt" ]] || fail "budget_params.pt not found: $SAFE_HTPO_CHECKPOINT"

cat > "$RESULT_DIR/protocol.json" <<EOF
{
  "run_id": "${SAFE_HTPO_RUN_ID}",
  "checkpoint": "${SAFE_HTPO_CHECKPOINT}",
  "selector": "scorer_budget",
  "prune_variant": "drop",
  "denylist": null,
  "note": "agent.keep_ratio is an ignored compatibility default for scorer_budget; each scene samples its own budget head ratio"
}
EOF

PIDS=()
for SH in 0 1 2 3; do
  EXP="MT_safehtpo_${SAFE_HTPO_RUN_ID}_dynamic_sh${SH}"
  CSV="$RESULT_DIR/${EXP}.csv"
  LOG="$LOG_DIR/_${EXP}.log"
  if [[ -f "$CSV" ]]; then
    log "SKIP shard ${SH}: existing result $CSV"
    continue
  fi
  log "START GPU${SH} shard${SH} dynamic random-budget Variant-B evaluation"
  (
    cd "$NAVSIM_ROOT"
    export CUDA_VISIBLE_DEVICES="$SH"
    timeout --signal=TERM --kill-after=10m "$TIMEOUT_SEC" "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$EXP" \
      train_test_split="${SHARD_PREFIX}${SH}${SHARD_SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" \
      +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" \
      +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false \
      +agent.selector=scorer_budget \
      +agent.scorer_ckpt="$SAFE_HTPO_CHECKPOINT" \
      +agent.keep_ratio=1.0 \
      +agent.prune_variant=drop \
      +agent.prune_verbose=true \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$LOG" 2>&1 &
  PIDS+=("$!")
done

RC=0
for PID in "${PIDS[@]:-}"; do
  if ! wait "$PID"; then
    RC=1
  fi
done
[[ "$RC" -eq 0 ]] || fail "At least one NAVSIM shard failed; inspect $LOG_DIR"

for SH in 0 1 2 3; do
  EXP="MT_safehtpo_${SAFE_HTPO_RUN_ID}_dynamic_sh${SH}"
  CSV="$RESULT_DIR/${EXP}.csv"
  if [[ ! -f "$CSV" ]]; then
    FOUND=$(find "$NAVSIM_EXP_ROOT/$EXP" -type f -name '*.csv' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    [[ -n "$FOUND" ]] || fail "No CSV found for $EXP"
    cp -a "$FOUND" "$CSV"
  fi
  [[ -s "$CSV" ]] || fail "Empty CSV: $CSV"
done

"$PY" "$ROOT/scripts/aggregate_safe_htpo_eval.py" \
  --result-dir "$RESULT_DIR" \
  --log-dir "$LOG_DIR" \
  --run-id "$SAFE_HTPO_RUN_ID" \
  --expected-scenes 11576 | tee "$RESULT_DIR/aggregate_stdout.json"

touch "$RESULT_DIR/EVAL_DONE"
log "DONE: $RESULT_DIR/safe_htpo_summary.md"
