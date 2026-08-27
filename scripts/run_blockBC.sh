#!/usr/bin/env bash
# run_blockBC.sh — 2026-08-03 周期 Block B / Block C
# 用法: bash run_blockBC.sh B    |    bash run_blockBC.sh C
#
# Block B — matched-compute 三元组（回应 Reviewer #2）
#   GPU0-3: MC_shuffled  = Budget RL scorer ranking + 置换后的 per-scene 预算
#   GPU4-7: MC_const     = Budget RL scorer ranking + 常数预算 0.354125
#   与已有 SUPP_budgetrl_dynamic_raw (learned) 构成三元组，三者平均 FLOPs 严格相同。
#     Δ(shuffled - const)   = 「预算有方差」的贡献
#     Δ(learned  - shuffled)= 「预算与场景匹配」的贡献  ← 论文核心 claim
#
# Block C — 同 compute 基线（回应 Reviewer #1）
#   GPU0-3: SparseVLM r=0.355 raw 全 navtest
#   GPU4-7: FastV     r=0.355 raw 全 navtest
#   现有 supplement 用 SparseVLM r=0.5 (保留 50%) 对比 Budget RL (保留 35.4%)，
#   保留率不同 → 不是同 compute。本 Block 补齐 r=0.355 的基线。
#
# 协议：全部 raw（无 safety_net、无 denylist、prune_variant=drop）
# 预计：每 Block ~3.6h（4 shard 并行，每 shard 2894-2963 scenes）
set -uo pipefail
BLOCK="${1:?usage: run_blockBC.sh B|C}"
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh" >/dev/null
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
PREFIX="navtest_local_filtered_shard"; SUFFIX="_20260616_154858"
BRL="$ROOT/ckpt/s3_token_scorer_budget_rl_20260722_155943_sh0/ckpt_best"
SHUF="$ROOT/results/kr_maps/kr_map_shuffled.json"
CONST="$ROOT/results/kr_maps/kr_map_const.json"
OUTDIR="$ROOT/results/raw/block${BLOCK}"
LOGDIR="$ROOT/logs/block${BLOCK}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR" "$LOGDIR"
echo "$LOGDIR" > "$ROOT/logs/block${BLOCK}_logdir.txt"

log(){ echo "[block$BLOCK $(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/run.log"; }

worker(){
  local gpu="$1"; shift
  local -a jobs=("$@")
  for job in "${jobs[@]}"; do
    IFS='|' read -r sel kr exp sh extra <<< "$job"
    local csv="$OUTDIR/${exp}.csv"
    local jlog="$LOGDIR/_${exp}.log"
    if [[ -f "$csv" ]]; then log "[GPU$gpu] SKIP $exp (csv exists)"; continue; fi
    log "[GPU$gpu] START $exp (sel=$sel kr=$kr shard=$sh)"
    ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
      timeout 40000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
        experiment_name="$exp" \
        train_test_split="${PREFIX}${sh}${SUFFIX}" \
        metric_cache_path="$ROOT/data/navtest_metric_cache" \
        +json_data_path="$ROOT/data/navtest_nocot" \
        agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
        +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" \
        +agent.sensor_data_path="$SENSOR" \
        +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false \
        +agent.keep_ratio="$kr" +agent.selector="$sel" \
        +agent.prune_variant=drop +agent.prune_verbose=true \
        $extra \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$jlog" 2>&1
    local rc=$?
    local found; found=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
    if [[ -n "$found" ]]; then
      cp -a "$found" "$csv"
      local pdms; pdms=$($PY -c "import pandas as pd;d=pd.read_csv('$csv');d=d[d['token']!='average'];print(f'{d[\"score\"].mean():.5f} N={len(d)}')" 2>/dev/null)
      log "[GPU$gpu] DONE $exp rc=$rc PDMS=$pdms"
    else
      log "[GPU$gpu] FAIL $exp rc=$rc (no csv)"
    fi
  done
}

declare -a J0 J1 J2 J3 J4 J5 J6 J7
if [[ "$BLOCK" == "B" ]]; then
  for sh in 0 1 2 3; do
    eval "J${sh}+=(\"scorer_budget|0.5|MC_shuffled_sh${sh}|${sh}|+agent.scorer_ckpt=$BRL +agent.kr_override_map=$SHUF\")"
    eval "J$((sh+4))+=(\"scorer_budget|0.5|MC_const_sh${sh}|${sh}|+agent.scorer_ckpt=$BRL +agent.kr_override_map=$CONST\")"
  done
  log "Block B: matched-compute triplet (shuffled | const) vs existing learned"
elif [[ "$BLOCK" == "C" ]]; then
  for sh in 0 1 2 3; do
    eval "J${sh}+=(\"sparsevlm_text|0.355|FAIR_sparsevlm_r0355_raw_sh${sh}|${sh}|\")"
    eval "J$((sh+4))+=(\"fastv_l2|0.355|FAIR_fastv_r0355_raw_sh${sh}|${sh}|\")"
  done
  log "Block C: same-compute baselines at r=0.355 raw (SparseVLM | FastV)"
else
  echo "unknown block '$BLOCK' (expect B or C)"; exit 2
fi

PIDS=()
for g in 0 1 2 3 4 5 6 7; do
  eval "jobs=(\"\${J${g}[@]}\")"
  [[ ${#jobs[@]} -eq 0 ]] && continue
  worker "$g" "${jobs[@]}" &
  PIDS+=($!)
done
printf '%s\n' "${PIDS[@]}" > "$LOGDIR/pids.txt"
log "PIDs: ${PIDS[*]}"
wait "${PIDS[@]}"

log "=== Block $BLOCK DONE — aggregating ==="
"$PY" - <<EOF 2>&1 | tee -a "$LOGDIR/run.log"
import glob, os, re, collections, pandas as pd
groups=collections.defaultdict(list)
for f in sorted(glob.glob("$OUTDIR/*.csv")):
    groups[re.sub(r'_sh\d+\.csv$','',os.path.basename(f))].append(f)
print("\n=== Block $BLOCK results (full navtest) ===")
for g,fs in sorted(groups.items()):
    d=pd.concat([pd.read_csv(f) for f in fs]);d=d[d['token']!='average'].drop_duplicates('token')
    print(f"  {g:34s} N={len(d):6d} PDMS={d['score'].mean():.5f} "
          f"NC={d['no_at_fault_collisions'].mean():.4f} EP={d['ego_progress'].mean():.4f} shards={len(fs)}")
EOF
log "=== Logs: $LOGDIR ==="
