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
CYCLE_ID=${CYCLE_ID:-$(date +%Y%m%d_%H%M%S)}
CYCLE_DIR="$ROOT/logs/rl_sota_cycle_${CYCLE_ID}"
RUN_ROOT="$ROOT/ckpt/rl_sota_cycle_${CYCLE_ID}"
mkdir -p "$CYCLE_DIR" "$RUN_ROOT" "$ROOT/results/raw"
echo "$CYCLE_ID" > "$ROOT/logs/rl_sota_cycle_latest"
exec > >(tee -a "$CYCLE_DIR/orchestrator.log") 2>&1

SCORER="$ROOT/ckpt/s3_token_scorer"
VLA_CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
VLA_CFG="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
JSON_DIR="$ROOT/data/navtrain_nocot"
METRIC="$ROOT/data/navtrain_metric_cache"
BASE="$ROOT/results/baseline_sub_scores.json"
SWEEP_SCENES=${SWEEP_SCENES:-512}
FINAL_SCENES=${FINAL_SCENES:-8192}
GROUP=${GROUP:-8}

# Keep the SFT token ranking nearly frozen and tune the existing Gaussian budget RL.
NAMES=(abs_eff0 delta_eff0 delta_eff002 delta_eff005 delta_eff01 delta_eff02 delta_eff005_lr3 delta_eff005_std2)
EFF=(0 0 0.002 0.005 0.01 0.02 0.005 0.005)
DELTA=(0 1 1 1 1 1 1 1)
BLR=(0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0003 0.0001)
BSTD=(-1 -1 -1 -1 -1 -1 -1 -2)

train_arm() {
  local gpu=$1 i=$2 out="$RUN_ROOT/${NAMES[$i]}"
  local delta_arg=()
  [[ ${DELTA[$i]} == 1 ]] && delta_arg=(--delta-reward)
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_scorer_budget_rl.py \
    --scorer-ckpt "$SCORER" --out-dir "$out" --json-dir "$JSON_DIR" \
    --metric-cache "$METRIC" --sensor-data-path "$SENSOR" \
    --autovla-config "$VLA_CFG" --autovla-ckpt "$VLA_CKPT" \
    --num-epochs 1 --max-scenes "$SWEEP_SCENES" --group-size "$GROUP" \
    --lr 1e-7 --budget-lr "${BLR[$i]}" --kl-beta 1.0 --budget-kl-beta 0.0 \
    --selection-pg-weight 0.0 --efficiency-beta "${EFF[$i]}" --driving-scale 3.0 \
    --safety-beta 0.5 --safety-margin 0.0 --min-keep-ratio 0.35 --max-keep-ratio 0.70 \
    --budget-log-std-init "${BSTD[$i]}" --seed 3407 --prune-variant attn_mask \
    --baseline-scores "$BASE" --counterfactual-k 0 --save-every 8 --log-every 1 \
    "${delta_arg[@]}" > "$CYCLE_DIR/train_${NAMES[$i]}.log" 2>&1
}

if [[ ${RESUME_FINAL_ONLY:-0} != 1 ]]; then
echo "[$(date)] phase1: eight-arm budget-RL sweep"
pids=()
for i in {0..7}; do train_arm "$i" "$i" & pids+=("$!"); done
printf '%s\n' "${pids[@]}" > "$CYCLE_DIR/phase1.pids"
for p in "${pids[@]}"; do wait "$p" || true; done

# Fast, identical held-out gate: 200 fixed navtest scenes, one model per GPU.
eval_arm() {
  local gpu=$1 i=$2 ckpt="$RUN_ROOT/${NAMES[$i]}/ckpt_best"
  [[ -f "$ckpt/checkpoint.pt" ]] || ckpt="$RUN_ROOT/${NAMES[$i]}"
  local exp="RL8_${CYCLE_ID}_${NAMES[$i]}_s200"
  (cd "$NAVSIM_ROOT" && CUDA_VISIBLE_DEVICES=$gpu timeout 10800 "$PY" navsim/planning/script/run_pdm_score_cot.py \
    experiment_name="$exp" train_test_split=navtest_s2sub200_shard0 \
    metric_cache_path="$ROOT/data/navtest_metric_cache" +json_data_path="$ROOT/data/navtest_nocot" \
    agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
    +agent.config_path="$VLA_CFG" +agent.checkpoint_path="$VLA_CKPT" +agent.sensor_data_path="$SENSOR" \
    +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" +agent.lora_conf.use_lora=false \
    +agent.selector=scorer_budget +agent.scorer_ckpt="$ckpt" +agent.keep_ratio=0.5 \
    +agent.prune_variant=drop +agent.prune_verbose=true worker=single_machine_thread_pool worker.max_workers=1 \
    > "$CYCLE_DIR/eval_${NAMES[$i]}.log" 2>&1)
  local found
  found=$(find "$NAVSIM_EXP_ROOT/$exp" -name '*.csv' -type f 2>/dev/null | head -1)
  [[ -n "$found" ]] && cp -a "$found" "$ROOT/results/raw/${exp}.csv"
}

echo "[$(date)] phase2: held-out evaluation"
pids=()
for i in {0..7}; do eval_arm "$i" "$i" & pids+=("$!"); done
printf '%s\n' "${pids[@]}" > "$CYCLE_DIR/phase2.pids"
for p in "${pids[@]}"; do wait "$p" || true; done

"$PY" - "$CYCLE_ID" "$CYCLE_DIR" <<'PY'
import glob,json,os,re,sys
import pandas as pd
cid, out = sys.argv[1:]
rows=[]
for f in glob.glob(f"/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/RL8_{cid}_*_s200.csv"):
 d=pd.read_csv(f); d=d[d.token.astype(str)!='average']
 name=os.path.basename(f)[len(f"RL8_{cid}_"):-len("_s200.csv")]
 log=os.path.join(out,f'eval_{name}.log')
 text=open(log,errors='ignore').read() if os.path.exists(log) else ''
 krs=[float(x) for x in re.findall(r'\[token_budget\].*?\bkr=([0-9.]+)',text)]
 rows.append({'name':name,'pdms':float(d.score.mean()),'mean_kr':sum(krs)/len(krs) if krs else None,'N':len(d)})
rows.sort(key=lambda x:x['pdms'],reverse=True)
if rows:
 near_best=[r for r in rows if rows[0]['pdms']-r['pdms'] <= 0.0005 and r['mean_kr'] is not None]
 winner=min(near_best,key=lambda r:r['mean_kr'])['name'] if near_best else rows[0]['name']
else: winner='delta_eff005'
json.dump(rows,open(out+'/sweep_results.json','w'),indent=2)
print(rows,'winner',winner)
open(out+'/winner.txt','w').write(winner+'\n')
PY
fi
WINNER=${WINNER_OVERRIDE:-$(cat "$CYCLE_DIR/winner.txt")}
case "$WINNER" in
  abs_eff0) W_EFF=0; W_DELTA=0; W_BLR=.0001; W_STD=-1;;
  delta_eff0) W_EFF=0; W_DELTA=1; W_BLR=.0001; W_STD=-1;;
  delta_eff002) W_EFF=.002; W_DELTA=1; W_BLR=.0001; W_STD=-1;;
  delta_eff005) W_EFF=.005; W_DELTA=1; W_BLR=.0001; W_STD=-1;;
  delta_eff01) W_EFF=.01; W_DELTA=1; W_BLR=.0001; W_STD=-1;;
  delta_eff02) W_EFF=.02; W_DELTA=1; W_BLR=.0001; W_STD=-1;;
  delta_eff005_lr3) W_EFF=.005; W_DELTA=1; W_BLR=.0003; W_STD=-1;;
  delta_eff005_std2) W_EFF=.005; W_DELTA=1; W_BLR=.0001; W_STD=-2;;
  *) W_EFF=.005; W_DELTA=1; W_BLR=.0001; W_STD=-1;;
esac
DARG=(); [[ $W_DELTA == 1 ]] && DARG=(--delta-reward)
FINAL_OUT="$RUN_ROOT/final_${WINNER}"
echo "$FINAL_OUT" > "$CYCLE_DIR/final_out.txt"
if [[ ${EVAL_ONLY:-0} != 1 ]]; then
echo "[$(date)] phase3: 8-GPU synchronized final train winner=$WINNER"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 28800 "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_scorer_budget_rl.py --distributed --scorer-ckpt "$SCORER" --out-dir "$FINAL_OUT" \
  --json-dir "$JSON_DIR" --metric-cache "$METRIC" --sensor-data-path "$SENSOR" \
  --autovla-config "$VLA_CFG" --autovla-ckpt "$VLA_CKPT" --num-epochs 1 --max-scenes "$FINAL_SCENES" \
  --group-size "$GROUP" --lr 1e-7 --budget-lr "$W_BLR" --kl-beta 1.0 --selection-pg-weight 0.0 \
  --efficiency-beta "$W_EFF" --driving-scale 3.0 --safety-beta .5 --safety-margin 0 \
  --min-keep-ratio .35 --max-keep-ratio .70 --budget-log-std-init "$W_STD" --seed 4407 \
  --prune-variant attn_mask --baseline-scores "$BASE" --counterfactual-k 0 --save-every 16 --log-every 1 \
  "${DARG[@]}" > "$CYCLE_DIR/final_train.log" 2>&1 || true
fi

# Full navtest is split into eight disjoint token lists so every GPU contributes useful evidence.
echo "[$(date)] phase4: final checkpoint full-navtest 8-way evaluation"
FINAL_CKPT="$FINAL_OUT/ckpt_best"; [[ -f "$FINAL_CKPT/checkpoint.pt" ]] || FINAL_CKPT="$FINAL_OUT"
FILTER_DIR="$NAVSIM_ROOT/navsim/planning/script/config/common/train_test_split/scene_filter"
"$PY" - "$FILTER_DIR" "$CYCLE_ID" <<'PY'
import copy,glob,os,sys,yaml
fd,cid=sys.argv[1:]
paths=sorted(glob.glob(os.path.join(fd,'navtest_local_filtered_shard*_20260616_154858.yaml')))
configs=[yaml.safe_load(open(p)) for p in paths]
tokens=[]
for c in configs: tokens.extend(c.get('tokens') or [])
tokens=list(dict.fromkeys(tokens))
logs=[]
for c in configs: logs.extend(c.get('log_names') or [])
logs=list(dict.fromkeys(logs))
for i in range(8):
 name=f'rlcycle_{cid}_g{i}'
 c=copy.deepcopy(configs[0]); c['tokens']=tokens[i::8]; c['log_names']=logs
 with open(os.path.join(fd,name+'.yaml'),'w') as f: yaml.safe_dump(c,f,sort_keys=False)
 wrapper=os.path.join(os.path.dirname(fd),name+'.yaml')
 with open(wrapper,'w') as f: f.write(f'data_split: test\ndefaults:\n- scene_filter: {name}\n')
print('full-navtest tokens',len(tokens),'split sizes',[len(tokens[i::8]) for i in range(8)])
PY
pids=()
for gpu in {0..7}; do
  (cd "$NAVSIM_ROOT" && CUDA_VISIBLE_DEVICES=$gpu timeout 14400 "$PY" navsim/planning/script/run_pdm_score_cot.py \
    experiment_name="RL8_${CYCLE_ID}_FINAL_g${gpu}" train_test_split="rlcycle_${CYCLE_ID}_g${gpu}" \
    metric_cache_path="$ROOT/data/navtest_metric_cache" +json_data_path="$ROOT/data/navtest_nocot" \
    agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
    +agent.config_path="$VLA_CFG" +agent.checkpoint_path="$VLA_CKPT" +agent.sensor_data_path="$SENSOR" \
    +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" +agent.lora_conf.use_lora=false \
    +agent.selector=scorer_budget +agent.scorer_ckpt="$FINAL_CKPT" +agent.keep_ratio=0.5 \
    +agent.prune_variant=drop +agent.prune_verbose=true worker=single_machine_thread_pool worker.max_workers=1 \
    > "$CYCLE_DIR/final_eval_g${gpu}.log" 2>&1) & pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$CYCLE_DIR/phase4.pids"
for p in "${pids[@]}"; do wait "$p" || true; done
for gpu in {0..7}; do
  exp="RL8_${CYCLE_ID}_FINAL_g${gpu}"
  found=$(find "$NAVSIM_EXP_ROOT/$exp" -name '*.csv' -type f 2>/dev/null | head -1)
  [[ -n "$found" ]] && cp -a "$found" "$ROOT/results/raw/${exp}.csv"
done
"$PY" - "$CYCLE_ID" "$CYCLE_DIR" <<'PY'
import glob,json,sys,pandas as pd
cid,out=sys.argv[1:]
fs=glob.glob(f'/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/RL8_{cid}_FINAL_g*.csv')
if fs:
 d=pd.concat([pd.read_csv(f) for f in fs]); d=d[d.token.astype(str)!='average'].drop_duplicates('token')
 r={'N':len(d),'pdms':float(d.score.mean())}
else: r={'N':0,'pdms':None}
json.dump(r,open(out+'/final_result.json','w'),indent=2); print(r)
PY

echo "[$(date)] cycle complete" | tee "$CYCLE_DIR/CYCLE_COMPLETE"
