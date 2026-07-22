#!/usr/bin/env bash
# run_baseline_pareto_4gpu.sh — 补跑 baseline 方法的 r=0.25/0.5/0.75 (GPU4-7 并行, 仅 shard0)
# shard0 = navtest_local_filtered_shard0_20260616_154858 (N≈2949) 够报数字
# 与 run_baseline_pareto_gpu4.sh 同 job 列表, 但 4 卡并行 (~4-5h vs 串行的 ~16h)
# 注意: 本脚本故意不检查 pdm_score 是否在跑, 因为 GPU0-3 正跑 Budget RL 验证, 互不冲突
# Launch: nohup bash scripts/run_baseline_pareto_4gpu.sh > logs/baseline_pareto_4gpu.log 2>&1 &
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
LOGDIR="$ROOT/logs/_baseline_pareto_4gpu"
mkdir -p "$OUTDIR" "$LOGDIR"
DENYLIST="$ROOT/results/varB_catastrophic_tokens.json"
GPUS=(4 5 6 7); NG=${#GPUS[@]}
log(){ echo "[pareto4 $(date +%H:%M:%S)] $*"; }

run_eval(){
    local gpu="$1" sel="$2" kr="$3" exp="$4"
    local CSV="$OUTDIR/${exp}.csv"
    if [[ -f "$CSV" ]]; then
        log "SKIP $exp (exists)"
        return
    fi
    log "START $exp on GPU$gpu (sel=$sel kr=$kr)"
    local extra_args=""
    if [[ "$sel" == "scorer" ]]; then
        extra_args="+agent.scorer_ckpt=$ROOT/ckpt/s3_token_scorer +agent.varB_denylist=$DENYLIST +agent.safety_net=true"
    elif [[ "$sel" == "scorer_taucut" ]]; then
        extra_args="+agent.scorer_ckpt=$ROOT/ckpt/s3_token_scorer +agent.varB_denylist=$DENYLIST +agent.safety_net=true +agent.tau=-0.1668 +agent.tau_min_keep=36"
    fi
    ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
      timeout 20000 $PY navsim/planning/script/run_pdm_score_cot.py \
        experiment_name="$exp" \
        train_test_split="${SHARD_PREFIX}0${SHARD_SUFFIX}" \
        metric_cache_path="$ROOT/data/navtest_metric_cache" \
        +json_data_path="$ROOT/data/navtest_nocot" \
        agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
        +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" +agent.sensor_data_path="$SENSOR" \
        +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
        +agent.lora_conf.use_lora=false +agent.keep_ratio="$kr" +agent.selector="$sel" \
        +agent.prune_variant=drop $extra_args +agent.prune_verbose=false \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$LOGDIR/_${exp}.log" 2>&1
    local FOUND=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
    if [[ -n "$FOUND" ]]; then
        cp -a "$FOUND" "$CSV"; log "DONE $exp"
        $PY -c "import pandas as pd; df=pd.read_csv('$CSV'); df=df[df['token']!='average']; print(f'  PDMS={df[\"score\"].mean():.6f} N={len(df)}')" 2>/dev/null || true
    else
        log "WARN $exp no csv (see $LOGDIR/_${exp}.log)"
    fi
}

# Job list: "selector keep_ratio experiment_name"
JOBS=(
  "fastv_l2 0.25 MT_fastv_l2_drop_r025_sh0"
  "random 0.25 MT_random_drop_r025_sh0"
  "prumerge_cls 0.25 MT_prumerge_cls_drop_r025_sh0"
  "sparsevlm_text 0.25 MT_sparsevlm_text_drop_r025_sh0"
  "fastv_l2 0.5 MT_fastv_l2_drop_r05_sh0"
  "random 0.5 MT_random_drop_r05_sh0"
  "prumerge_cls 0.5 MT_prumerge_cls_drop_r05_sh0"
  "sparsevlm_text 0.5 MT_sparsevlm_text_drop_r05_sh0"
  "fastv_l2 0.75 MT_fastv_l2_drop_r075_sh0"
  "random 0.75 MT_random_drop_r075_sh0"
  "prumerge_cls 0.75 MT_prumerge_cls_drop_r075_sh0"
  "sparsevlm_text 0.75 MT_sparsevlm_text_drop_r075_sh0"
  "scorer 0.25 MT_sft_varB_drop_r025_sh0"
  "scorer 0.75 MT_sft_varB_drop_r075_sh0"
  "scorer_taucut 0.5 MT_sft_taucut_drop_kr060_sh0"
  "scorer 0.5 MT_varB_safetynet_r05_sh0"
)

worker(){
  local gpu="$1" idx="$2" i="$idx"
  while [[ "$i" -lt "${#JOBS[@]}" ]]; do
    local -a j=(${JOBS[$i]})
    run_eval "$gpu" "${j[0]}" "${j[1]}" "${j[2]}"
    i=$(( i + NG ))
  done
}

log "=== Baseline Pareto补跑 (shard0 only, GPU4-7 并行, ${#JOBS[@]} jobs) ==="
pids=()
for idx in "${!GPUS[@]}"; do
  worker "${GPUS[$idx]}" "$idx" &
  pids+=($!)
done
wait "${pids[@]}"
log "=== ALL GPU4-7 TASKS COMPLETE ==="
# 汇总已产出的 shard0 数字
$PY -c "
import pandas as pd
from pathlib import Path
out=Path('$OUTDIR')
for name,exp in [('FastV r=0.25','MT_fastv_l2_drop_r025_sh0'),('Random r=0.25','MT_random_drop_r025_sh0'),('PruMerge r=0.25','MT_prumerge_cls_drop_r025_sh0'),('SparseVLM r=0.25','MT_sparsevlm_text_drop_r025_sh0'),('FastV r=0.5','MT_fastv_l2_drop_r05_sh0'),('Random r=0.5','MT_random_drop_r05_sh0'),('PruMerge r=0.5','MT_prumerge_cls_drop_r05_sh0'),('SparseVLM r=0.5','MT_sparsevlm_text_drop_r05_sh0'),('FastV r=0.75','MT_fastv_l2_drop_r075_sh0'),('Random r=0.75','MT_random_drop_r075_sh0'),('PruMerge r=0.75','MT_prumerge_cls_drop_r075_sh0'),('SparseVLM r=0.75','MT_sparsevlm_text_drop_r075_sh0'),('SFT VarB r=0.25','MT_sft_varB_drop_r025_sh0'),('SFT VarB r=0.75','MT_sft_varB_drop_r075_sh0'),('SFT tau-cut','MT_sft_taucut_drop_kr060_sh0'),('SFT safety-net','MT_varB_safetynet_r05_sh0')]:
    p=out/(exp+'.csv')
    if p.exists():
        df=pd.read_csv(p); df=df[df['token']!='average']
        print(f'  {name}: PDMS={df[\"score\"].mean():.4f} N={len(df)}')
    else:
        print(f'  {name}: MISSING')
" 2>/dev/null
