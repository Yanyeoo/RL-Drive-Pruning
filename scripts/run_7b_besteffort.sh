#!/usr/bin/env bash
# run_7b_besteffort.sh — Best-effort 7B ImpromptuVLA + nuScenes zero-shot eval.
#
# NON-BLOCKING and PROTOCOL-SAFE: this tries to obtain FastDriveVLA-comparable
# zero-shot numbers using the already-trained 7B scorer. On ANY failure it
# writes a diagnostic report and exits 0 (never blocks the main table). It does
# NOT modify any model/protocol; the 7B preprocessor_config already contains
# image_processor_type (verified 2026-07-28), so the earlier loader blocker
# should be resolved.
#
# Step 1: prune+infer -> pred_r<rt>.jsonl (this repo's script)
# Step 2: evaluation_nuscenes.py -> L2 / collision metrics
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
MODEL="$ROOT/models/ImpromptuVLA_7B/7B_AD_finetune"
SCORER="$ROOT/ckpt/s3_token_scorer_7b"
DATA="$ROOT/code/third_party/ImpromptuVLA/nuscenes_test.json"
EVAL2="$ROOT/code/third_party/ImpromptuVLA/data_qa_generate/data_engine/datasets/nuscenes/scripts/evaluation_nuscenes.py"
OUT="$ROOT/results/impromptu7b"; RPT="$OUT/BESTEFFORT_REPORT.md"
LOGD="$ROOT/logs/maintable_overnight"; mkdir -p "$OUT" "$LOGD"
DEADLINE_EPOCH=$(date -d "2026-07-29 13:45:00" +%s)
tl(){ echo $(( DEADLINE_EPOCH - $(date +%s) )); }
say(){ echo "[7b $(date '+%F %T')] $*"; }
report(){ echo "$*" >> "$RPT"; }

: > "$RPT"
report "# 7B ImpromptuVLA / nuScenes best-effort report"
report ""
report "- started: $(date -Iseconds)"
report "- model: $MODEL"
report "- scorer: $SCORER"
report "- data: $DATA (ShareGPT, 6019 samples)"
report ""

# --- preflight ---
if ! grep -q image_processor_type "$MODEL/preprocessor_config.json" 2>/dev/null; then
  say "preprocessor_config missing image_processor_type -> abort best-effort"
  report "RESULT: SKIPPED — model preprocessor_config.json lacks image_processor_type (loader blocker)."
  exit 0
fi
[[ -f "$DATA" ]] || { report "RESULT: SKIPPED — data json not found."; exit 0; }
[[ -f "$SCORER/checkpoint.pt" ]] || { report "RESULT: SKIPPED — 7B scorer ckpt not found."; exit 0; }

# --- smoke: 8 scenes r=0.5 on GPU0 ---
say "smoke (8 scenes r=0.5)"
CUDA_VISIBLE_DEVICES=0 timeout 1800 "$PY" scripts/run_impromptu7b_nuscenes_eval.py \
  --model-path "$MODEL" --scorer-ckpt "$SCORER" --keep-ratio 0.5 \
  --data-json "$DATA" --max-scenes 8 --device cuda:0 \
  --output "$OUT/smoke_r05.jsonl" > "$LOGD/_7b_smoke.log" 2>&1
srac=$?
n_ok=$(grep -c '"predict": "[^"]' "$OUT/smoke_r05.jsonl" 2>/dev/null || echo 0)
say "smoke rc=$srac n_ok=$n_ok"
if [[ "$srac" -ne 0 || "${n_ok:-0}" -lt 1 ]]; then
  report "RESULT: SMOKE FAILED (rc=$srac, non-empty preds=$n_ok)."
  report ""
  report "Last 25 lines of smoke log:"
  report '```'
  tail -25 "$LOGD/_7b_smoke.log" >> "$RPT" 2>/dev/null
  report '```'
  say "smoke failed -> report and exit 0 (non-blocking)"
  exit 0
fi
report "SMOKE: PASS (non-empty preds=$n_ok). Proceeding to full parallel runs."
report ""

# --- full runs: 4 keep-ratios in parallel, one per GPU ---
declare -A RG=( [1.0]=0 [0.75]=1 [0.5]=2 [0.25]=3 )
declare -A RNAME=( [1.0]=r10 [0.75]=r075 [0.5]=r05 [0.25]=r025 )
pids=()
for r in 1.0 0.75 0.5 0.25; do
  gpu=${RG[$r]}; nm=${RNAME[$r]}; pred="$OUT/pred_${nm}.jsonl"
  if [[ -f "$pred" ]] && [[ $(grep -c '"predict"' "$pred" 2>/dev/null || echo 0) -ge 6019 ]]; then
    say "SKIP r=$r (pred exists complete)"; continue
  fi
  to=$(( $(tl) - 600 )); [[ $to -gt 32000 ]] && to=32000
  if [[ $to -lt 1200 ]]; then say "deadline -> skip r=$r"; continue; fi
  say "START full r=$r on GPU$gpu (timeout=${to}s)"
  ( CUDA_VISIBLE_DEVICES=$gpu timeout "$to" "$PY" scripts/run_impromptu7b_nuscenes_eval.py \
      --model-path "$MODEL" --scorer-ckpt "$SCORER" --keep-ratio "$r" \
      --data-json "$DATA" --device cuda:0 \
      --output "$pred" > "$LOGD/_7b_full_${nm}.log" 2>&1 ) &
  pids+=($!)
done
[[ ${#pids[@]} -gt 0 ]] && wait "${pids[@]}"
say "full runs finished"

# --- step 2: metrics per ratio ---
report "## Zero-shot results (nuScenes, prune+infer)"
report ""
report "| keep_ratio | pred lines | eval metrics file |"
report "|---|---:|---|"
for r in 1.0 0.75 0.5 0.25; do
  nm=${RNAME[$r]}; pred="$OUT/pred_${nm}.jsonl"; evj="$OUT/eval_${nm}.json"
  [[ -f "$pred" ]] || { report "| $r | MISSING | — |"; continue; }
  n=$(grep -c '"predict"' "$pred" 2>/dev/null || echo 0)
  ( cd "$(dirname "$EVAL2")" && timeout 1200 "$PY" "$EVAL2" \
      --jsonl_file "$pred" --output_file "$evj" --mode x-y \
      > "$LOGD/_7b_eval_${nm}.log" 2>&1 ) || true
  report "| $r | $n | $(basename "$evj") |"
done
report ""
report "- finished: $(date -Iseconds)"
say "done -> $RPT"
exit 0
