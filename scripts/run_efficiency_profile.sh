#!/usr/bin/env bash
# run_efficiency_profile.sh — TokenRL EFFICIENCY evidence (contribution ①).
#
# Two measurements:
#  (1) Token-saving %% : parse per-scene "[token_budget] ... N=.. kr=.. -> prune X tokens"
#      lines from the Budget RL DYNAMIC eval logs (stage B, which used prune_verbose=true).
#      No GPU needed.
#  (2) Wall-clock speedup : re-run navtest shard0 with prune_variant=drop (real FLOPs saving)
#      for scorer@0.5 vs attn_L12@1.0 (no-prune reference), timed with /usr/bin/time -v.
#      This is the number D (which uses attn_mask by default) does NOT produce.
#
# Output: results/eff_profile/{_time_*.log, MT_eff_*.csv, efficiency_summary.md}
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"; SCORER="$ROOT/ckpt/s3_token_scorer"
OUT="$ROOT/results/eff_profile"; mkdir -p "$OUT" logs
SHARD_PREFIX="navtest_local_filtered_shard"; SHARD_SUFFIX="_20260616_154858"
log(){ echo "[eff $(date +%H:%M:%S)] $*" | tee -a "$ROOT/logs/unattended_journal.log"; }

# ---------- (1) token-saving %% from B dynamic logs ----------
log "=== (1) token-saving from Budget RL dynamic eval logs ==="
FILES=$(grep -rl "\[token_budget\]" logs/ 2>/dev/null)
if [[ -z "$FILES" ]]; then
  log "WARN: no [token_budget] logs found (stage B not done yet?) -> skip token-saving parse"
else
  awk '
    /\[token_budget\]/ {
      for(i=1;i<=NF;i++){
        if($i=="N="){n=$(i+1)}
        if($i=="kr="){kr=$(i+1)}
        if($i=="prune"){p=$(i+1)}
      }
      if(n>0 && p>0){ TN+=n; TP+=p; SCENES++ }
      if(kr!=""){ KRS+=kr; KRN++ }
    }
    END{
      if(TN>0){
        printf "scenes=%d total_vision_tokens=%d pruned=%d saving_pct=%.2f\n", SCENES, TN, TP, 100.0*TP/TN
      }
      if(KRN>0){
        printf "mean_dynamic_keep_ratio=%.3f\n", KRS/KRN
      }
    }' $FILES | while read -r l; do log "  $l"; done
  # stash for summary
  awk '/\[token_budget\]/{for(i=1;i<=NF;i++){if($i=="N=")n=$(i+1);if($i=="prune")p=$(i+1);if($i=="kr=")kr=$(i+1)} if(n>0&&p>0){TN+=n;TP+=p;S++} if(kr!=""){K+=kr;KN++}} END{printf "saving_pct=%.2f mean_kr=%.3f scenes=%d\n",100.0*TP/TN,K/KN,S}' $FILES > "$OUT/_token_saving.txt"
fi

# ---------- (2) wall-clock speedup (timed, drop variant) ----------
run_timed(){
  local gpu="$1" sel="$2" kr="$3" exp="$4"
  local csv="$OUT/${exp}.csv"
  [[ -f "$csv" ]] && { log "SKIP $exp (done)"; return; }
  log "GPU$gpu TIMED $exp (sel=$sel kr=$kr, drop)"
  (
    cd "$NAVSIM_ROOT"
    export CUDA_VISIBLE_DEVICES="$gpu"
    /usr/bin/time -v "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" \
      train_test_split="${SHARD_PREFIX}0${SHARD_SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" \
      +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false +agent.keep_ratio="$kr" +agent.selector="$sel" \
      +agent.prune_variant=drop +agent.prune_verbose=true \
      +agent.scorer_ckpt="$SCORER" \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$OUT/_run_${exp}.log" 2> "$OUT/_time_${exp}.log"
  local c; c=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  [[ -n "$c" ]] && cp -a "$c" "$csv" && log "DONE $exp -> csv"
}
elapsed_sec(){ grep -E "Elapsed \(wall clock\) time" "$OUT/_time_$1.log" 2>/dev/null | grep -oE "[0-9]+:[0-9]{2}:[0-9]{2}|[0-9.]+" | head -1; }

log "=== (2) timed speedup: ref(attn_L12@1.0) vs pruned(scorer@0.5, drop) on shard0 ==="
# run both in parallel on gpu0 / gpu1
run_timed 0 "attn_L12" "1.0" "MT_eff_ref_sh0" &
run_timed 1 "scorer"   "0.5" "MT_eff_pruned_sh0" &
wait

REF=$(elapsed_sec MT_eff_ref_sh0)
PRU=$(elapsed_sec MT_eff_pruned_sh0)
if [[ -n "$REF" && -n "$PRU" && "$PRU" != "0.00" ]]; then
  SPEEDUP=$(awk "BEGIN{printf \"%.3f\", $REF/$PRU}")
  log "ref_elapsed=$REF  pruned_elapsed=$PRU  speedup=${SPEEDUP}x"
  echo "speedup=$SPEEDUP ref=$REF pruned=$PRU" > "$OUT/_speedup.txt"
else
  log "WARN: could not parse elapsed times (ref=$REF pruned=$PRU)"
fi

# ---------- summary ----------
{
  echo "# TokenRL Efficiency Profile (generated $(date '+%Y-%m-%d %H:%M'))"
  echo
  echo "## (1) Token saving (Budget RL dynamic budget, parsed from stage-B logs)"
  [[ -f "$OUT/_token_saving.txt" ]] && cat "$OUT/_token_saving.txt" | sed 's/^/- /'
  echo
  echo "## (2) Wall-clock speedup (prune_variant=drop, shard0 timed)"
  [[ -f "$OUT/_speedup.txt" ]] && cat "$OUT/_speedup.txt" | sed 's/^/- /'
  echo
  echo "Note: stage D used attn_mask (Variant A, no real FLOPs saving); this script uses"
  echo "drop (Variant B) to measure TRUE speedup at the cost of real token removal."
} > "$OUT/efficiency_summary.md"
log "summary -> $OUT/efficiency_summary.md"
log "EFFICIENCY PROFILE DONE"
