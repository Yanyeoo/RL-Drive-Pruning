#!/usr/bin/env bash
set -uo pipefail
ROOT=/apdcephfs/private_shayladeng/tokenrl_autoVLA
cd "$ROOT"
PY=/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"
NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh" >/dev/null
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
CID=${CYCLE_ID:-$(cat "$ROOT/logs/rl_sota_cycle_latest")}
CYCLE_DIR="$ROOT/logs/rl_sota_cycle_${CID}"
RUN_ROOT="$ROOT/ckpt/rl_sota_cycle_${CID}"
VLA_CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
VLA_CFG="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
mkdir -p "$CYCLE_DIR/full_candidate_eval" "$ROOT/results/raw"
exec >> "$CYCLE_DIR/full_candidate_eval/queue.log" 2>&1

CANDIDATES=(delta_eff005_std2 delta_eff02 delta_eff01 delta_eff0)
for name in "${CANDIDATES[@]}"; do
  ckpt="$RUN_ROOT/$name/ckpt_best"
  [[ -f "$ckpt/checkpoint.pt" ]] || { echo "[$(date)] SKIP $name missing $ckpt"; continue; }
  echo "[$(date)] START full-navtest candidate=$name ckpt=$ckpt"
  pids=()
  for gpu in {0..7}; do
    exp="RL8_${CID}_FULLARM_${name}_g${gpu}"
    logfile="$CYCLE_DIR/full_candidate_eval/${name}_g${gpu}.log"
    (cd "$NAVSIM_ROOT" && CUDA_VISIBLE_DEVICES=$gpu timeout 10800 "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" train_test_split="rlcycle_${CID}_g${gpu}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$VLA_CFG" +agent.checkpoint_path="$VLA_CKPT" +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" +agent.lora_conf.use_lora=false \
      +agent.selector=scorer_budget +agent.scorer_ckpt="$ckpt" +agent.keep_ratio=0.5 \
      +agent.prune_variant=drop +agent.prune_verbose=true worker=single_machine_thread_pool worker.max_workers=1 \
      > "$logfile" 2>&1) & pids+=("$!")
  done
  printf '%s\n' "${pids[@]}" > "$CYCLE_DIR/full_candidate_eval/current.pids"
  for p in "${pids[@]}"; do wait "$p" || true; done
  for gpu in {0..7}; do
    exp="RL8_${CID}_FULLARM_${name}_g${gpu}"
    found=$(find "$NAVSIM_EXP_ROOT/$exp" -name '*.csv' -type f 2>/dev/null | head -1)
    [[ -n "$found" ]] && cp -a "$found" "$ROOT/results/raw/${exp}.csv"
  done
  "$PY" - "$ROOT" "$CID" "$name" <<'PY'
import glob,json,re,sys,pandas as pd
root,cid,name=sys.argv[1:]
fs=glob.glob(f'{root}/results/raw/RL8_{cid}_FULLARM_{name}_g*.csv')
result={'name':name,'N':0,'pdms':None,'mean_kr':None}
if fs:
 d=pd.concat([pd.read_csv(f) for f in fs]); d=d[d.token.astype(str)!='average'].drop_duplicates('token')
 logs=glob.glob(f'{root}/logs/rl_sota_cycle_{cid}/full_candidate_eval/{name}_g*.log')
 k=[]
 for f in logs:
  k += [float(x) for x in re.findall(r'\[token_budget\].*?\bkr=([0-9.]+)',open(f,errors='ignore').read())]
 result={'name':name,'N':len(d),'pdms':float(d.score.mean()),'mean_kr':sum(k)/len(k) if k else None}
out=f'{root}/logs/rl_sota_cycle_{cid}/full_candidate_eval/{name}_result.json'
json.dump(result,open(out,'w'),indent=2); print(result)
PY
  echo "[$(date)] DONE candidate=$name"
done
"$PY" - "$CYCLE_DIR/full_candidate_eval" <<'PY'
import glob,json,os,sys
out=sys.argv[1]; rows=[]
for f in glob.glob(out+'/*_result.json'):
 try: rows.append(json.load(open(f)))
 except Exception: pass
rows.sort(key=lambda x:x['pdms'] if x['pdms'] is not None else -1,reverse=True)
json.dump(rows,open(out+'/summary.json','w'),indent=2); print(rows)
PY
echo "[$(date)] candidate queue complete" | tee "$CYCLE_DIR/full_candidate_eval/COMPLETE"
