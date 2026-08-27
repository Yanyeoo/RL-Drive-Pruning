#!/usr/bin/env bash
# run_budget_rl_eval_8gpu.sh — 8-GPU Budget RL eval over FULL navtest (4 shards).
#
# Design (uses all 8 GPUs, produces robust full-navtest numbers):
#   GPU0-3 : DYNAMIC per-scene budget (selector=scorer_budget) on navtest shard 0-3
#   GPU4-7 : FIXED  r=0.5        (selector=scorer)        on navtest shard 0-3  (apples-to-apples)
#
# Resumable: SKIP_DONE per experiment CSV. Copies result CSV back from NAVSIM_EXP_ROOT.
# Does NOT chain 7B here — the unattended orchestrator owns stage chaining.
# Launch: bash scripts/run_budget_rl_eval_8gpu.sh
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SHARD_PREFIX="navtest_local_filtered_shard"
SHARD_SUFFIX="_20260616_154858"
OUTDIR="$ROOT/results/raw/tokenprune_S3_full"
LOGDIR="$ROOT/logs/_budget_rl_eval_8gpu"
mkdir -p "$OUTDIR" "$LOGDIR"
log(){ echo "[budget-eval-8gpu $(date +%H:%M:%S)] $*"; }

# Checkpoint selection (carries budget_net head). Use shard0's ckpt (matches eval design).
# IMPORTANT: the FINAL model is written to the SHARD ROOT as checkpoint.pt at the end of
# training (tag='final' -> save_dir = out_dir); there is NO ckpt_final/ subdir.
# ckpt_best is selected by a single-group noisy mean_reward and gets frozen at an early
# lucky spike (observed stuck at step~50), i.e. essentially the SFT warm-start. We therefore
# PREFER the true final (shard-root checkpoint.pt), fallback to latest ckpt_stepN, then ckpt_best.
SH0DIR=$(ls -dt $ROOT/ckpt/s3_token_scorer_budget_rl_*_sh0 2>/dev/null | head -1)
BUDGET_CKPT=""
if [[ -n "$SH0DIR" ]]; then
    if [[ -f "$SH0DIR/checkpoint.pt" ]]; then
        BUDGET_CKPT="$SH0DIR"
    else
        # latest step checkpoint by numeric step
        LATEST_STEP=$(ls -d "$SH0DIR"/ckpt_step* 2>/dev/null | sed -E 's/.*ckpt_step([0-9]+)/\1/' | sort -n | tail -1)
        if [[ -n "$LATEST_STEP" && -f "$SH0DIR/ckpt_step${LATEST_STEP}/checkpoint.pt" ]]; then
            BUDGET_CKPT="$SH0DIR/ckpt_step${LATEST_STEP}"
        elif [[ -f "$SH0DIR/ckpt_best/checkpoint.pt" ]]; then
            BUDGET_CKPT="$SH0DIR/ckpt_best"
        else
            BUDGET_CKPT="$SH0DIR"
        fi
    fi
fi
if [[ -z "$BUDGET_CKPT" ]]; then
    log "FATAL: No budget RL ckpt found"; exit 2
fi
log "Using budget RL ckpt: $BUDGET_CKPT"

# SFT scorer (plain TokenImportanceScorer, no budget head) used for the FIXED r=0.5 baseline.
# NOTE: we cannot reuse $BUDGET_CKPT for the 'scorer' selector because ScorerRunner.load_state_dict
# is strict and the budget checkpoint carries extra/mismatched 'token_net.*'+'budget_net.*' keys.
# The budget RL token_net was warm-started from this SFT scorer, so it is the faithful fixed-ratio baseline.
SFT_CKPT=$(ls -dt $ROOT/ckpt/s3_token_scorer 2>/dev/null | head -1)
if [[ -z "$SFT_CKPT" || ! -f "$SFT_CKPT/checkpoint.pt" ]]; then
    log "WARN: SFT scorer ckpt not found at $ROOT/ckpt/s3_token_scorer; fixed-r baseline will fall back to BUDGET_CKPT (may error)"
fi
log "Using SFT scorer for fixed-r baseline: $SFT_CKPT"

run_one(){
    local gpu="$1" sel="$2" kr="$3" sh="$4" verbose="$5" tag="$6" sckpt="${7:-$BUDGET_CKPT}"
    local exp="MT_budget_rl_${tag}_sh${sh}"
    local csv="$OUTDIR/${exp}.csv"
    [[ -f "$csv" ]] && { log "SKIP $exp (done)"; return; }
    log "GPU$gpu START $exp (sel=$sel kr=$kr scorer=$sckpt)"
    ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
      timeout 50000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
        experiment_name="$exp" \
        train_test_split="${SHARD_PREFIX}${sh}${SHARD_SUFFIX}" \
        metric_cache_path="$ROOT/data/navtest_metric_cache" \
        +json_data_path="$ROOT/data/navtest_nocot" \
        agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
        +agent.config_path="$YAML" \
        +agent.checkpoint_path="$CKPT" \
        +agent.sensor_data_path="$SENSOR" \
        +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false \
        +agent.selector="$sel" \
        +agent.scorer_ckpt="$sckpt" \
        +agent.keep_ratio="$kr" \
        +agent.prune_variant=drop \
        +agent.prune_verbose="$verbose" \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${exp}.log" 2>&1 &
    echo $!
}

PIDS=""
# DYNAMIC per-scene budget: GPU0-3 x shard 0-3
for SH in 0 1 2 3; do
    P=$(run_one "$SH" "scorer_budget" "0.5" "$SH" "true" "dynamic"); [[ -n "$P" ]] && PIDS="$PIDS $P"
done
# FIXED r=0.5 comparison: GPU4-7 x shard 0-3 (uses SFT scorer - faithful fixed-ratio baseline)
for SH in 0 1 2 3; do
    GPU=$((SH+4))
    P=$(run_one "$GPU" "scorer" "0.5" "$SH" "false" "r050" "$SFT_CKPT"); [[ -n "$P" ]] && PIDS="$PIDS $P"
done

[[ -n "$PIDS" ]] && { log "Waiting on PIDs:$PIDS"; wait $PIDS 2>/dev/null; }

# Copy CSVs back from NAVSIM exp root
for EXP in MT_budget_rl_dynamic_sh0 MT_budget_rl_dynamic_sh1 MT_budget_rl_dynamic_sh2 MT_budget_rl_dynamic_sh3 \
          MT_budget_rl_r050_sh0    MT_budget_rl_r050_sh1    MT_budget_rl_r050_sh2    MT_budget_rl_r050_sh3; do
    CSV="$OUTDIR/${EXP}.csv"
    if [[ ! -f "$CSV" ]]; then
        FOUND=$(ls -t "$NAVSIM_EXP_ROOT/$EXP"/*/*.csv 2>/dev/null | head -1)
        [[ -n "$FOUND" ]] && cp -a "$FOUND" "$CSV" && log "DONE $EXP"
    fi
done

# Aggregate + report
$PY -c "
import pandas as pd, glob
from pathlib import Path
print('=== Budget RL Eval (FULL navtest, 8-GPU) ===')
for tag, pat in [('dynamic(scorer_budget)','MT_budget_rl_dynamic_sh[0-3].csv'),
                 ('fixed r=0.5 (scorer)','MT_budget_rl_r050_sh[0-3].csv')]:
    fs = sorted(glob.glob('$OUTDIR/'+pat))
    if fs:
        df = pd.concat([pd.read_csv(f) for f in fs]); df = df[df['token']!='average']
        print(f'  {tag:24s}: N={len(df)}, PDMS={df[\"score\"].mean():.4f}')
"
# Summarize per-scene keep_ratio spread from dynamic logs
KR=$(grep -hoE "kr=[0-9.]+" "$LOGDIR"/_MT_budget_rl_dynamic_sh*.log 2>/dev/null | sed 's/kr=//' | \
     awk '{s+=$1;s2+=$1*$1;n++} END{if(n>0) printf "mean=%.3f std=%.3f n=%d", s/n, sqrt(s2/n-(s/n)^2), n}')
[[ -n "$KR" ]] && log "Dynamic budget keep_ratio spread: $KR"
log "=== Budget RL 8-GPU eval done ==="
