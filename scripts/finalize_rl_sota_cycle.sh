#!/usr/bin/env bash
set -uo pipefail
ROOT=/apdcephfs/private_shayladeng/tokenrl_autoVLA
cd "$ROOT"
PY=/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python
CID=$(cat "$ROOT/logs/rl_sota_cycle_latest" 2>/dev/null || true)
[[ -n "$CID" ]] || { echo "No active cycle id"; exit 1; }
D="$ROOT/logs/rl_sota_cycle_${CID}"
mkdir -p "$D"
echo "[$(date)] 09:50 closing begins" | tee -a "$D/finalize.log"

# Gracefully stop only this cycle's jobs, preserving checkpoints/resume state where available.
for f in "$D"/phase*.pids "$D/full_candidate_eval/current.pids" "$ROOT/logs/rl_sota_cycle_launcher.pid"; do
  [[ -f "$f" ]] || continue
  while read -r p; do
    args=$(ps -p "$p" -o args= 2>/dev/null || true)
    if [[ "$args" == *"rl_sota_cycle_${CID}"* || "$args" == *"RL8_${CID}"* || "$args" == *"run_rl_sota_cycle_8gpu.sh"* ]]; then
      kill -TERM "$p" 2>/dev/null || true
    fi
  done < "$f"
done
mapfile -t own < <(pgrep -f "train_scorer_budget_rl.py.*rl_sota_cycle_${CID}|RL8_${CID}" || true)
((${#own[@]})) && kill -TERM "${own[@]}" 2>/dev/null || true
sleep 20
mapfile -t own < <(pgrep -f "train_scorer_budget_rl.py.*rl_sota_cycle_${CID}|RL8_${CID}" || true)
((${#own[@]})) && kill -KILL "${own[@]}" 2>/dev/null || true

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$D/gpu_at_close.txt" 2>&1 || true
pgrep -af "train_scorer_budget_rl|RL8_${CID}" > "$D/processes_at_close.txt" 2>&1 || true

"$PY" - "$ROOT" "$CID" <<'PY'
import glob,json,os,sys,datetime
import pandas as pd
root,cid=sys.argv[1:]
d=f'{root}/logs/rl_sota_cycle_{cid}'
rows=[]
for f in glob.glob(f'{root}/results/raw/RL8_{cid}_*_s200.csv'):
 try:
  x=pd.read_csv(f); x=x[x.token.astype(str)!='average']
  rows.append((os.path.basename(f),len(x),float(x.score.mean())))
 except Exception: pass
rows.sort(key=lambda x:x[2],reverse=True)
full=[]
for f in glob.glob(f'{root}/results/raw/RL8_{cid}_FINAL_g*.csv'):
 try: full.append(pd.read_csv(f))
 except Exception: pass
full_result='尚未完成'
if full:
 x=pd.concat(full); x=x[x.token.astype(str)!='average'].drop_duplicates('token')
 full_result=f"N={len(x)}, PDMS={x.score.mean():.6f}"
final_out=open(f'{d}/final_out.txt').read().strip() if os.path.exists(f'{d}/final_out.txt') else '尚未进入最终训练'
winner=open(f'{d}/winner.txt').read().strip() if os.path.exists(f'{d}/winner.txt') else '尚未筛出'
candidate_results=[]
for f in glob.glob(f'{d}/full_candidate_eval/*_result.json'):
 try: candidate_results.append(json.load(open(f)))
 except Exception: pass
candidate_results.sort(key=lambda x:x.get('pdms') if x.get('pdms') is not None else -1,reverse=True)
lines=['# RL SOTA 周期交付（自动收尾）','',f'> 收尾时间：{datetime.datetime.now().isoformat(timespec="seconds")} ；周期：`{cid}`','',
'## 不可更改的总目标','',
'**在当前 RL idea 无大改的情况下达到 SOTA 且具备 efficiency。后续 AI 不得降低、替换或改写该目标。**','',
'验收至少包括：RL PDMS 超过当前 learned SFT `0.89199`；继续冲击整体上界/SOTA 约 `0.89879~0.899`；同时报告真实动态 keep ratio / FLOPs / latency。','',
'## 本周期关键代码修复','',
'- `scripts/train_scorer_budget_rl.py`：修复 `budget_log_prob.detach().item()` 导致预算策略完全无梯度。',
'- 修复 `efficiency_beta` 从未进入 reward 的确定性 bug。',
'- 修复反事实训练仅使用 group 最后一个 scene loss 的 bug。',
'- 普通 token PG 不再重复包含 budget log-prob；新增 same-scene `--delta-reward` 降方差。',
'- 策略仍是原 Gaussian budget REINFORCE + SFT token scorer，没有大改 RL idea。','',
'## 自动周期状态','',f'- 周期目录：`logs/rl_sota_cycle_{cid}`',f'- checkpoint 根目录：`ckpt/rl_sota_cycle_{cid}`',f'- sweep 胜者：`{winner}`',f'- 最终训练输出：`{final_out}`',f'- full-navtest 聚合：**{full_result}**','',
'## 200-scene 同集筛选结果','']
if rows:
 lines += ['| 模型 | N | PDMS |','|---|---:|---:|']+[f'| `{a}` | {b} | {c:.6f} |' for a,b,c in rows]
else: lines += ['收尾时尚无有效 CSV；先查各 `train_*.log` / `eval_*.log`。']
lines += ['', '## 候选checkpoint full-navtest结果','']
if candidate_results:
 lines += ['| 模型 | N | PDMS | mean keep ratio |','|---|---:|---:|---:|']+[f"| `{r.get('name')}` | {r.get('N',0)} | {r.get('pdms') if r.get('pdms') is not None else '未完成'} | {r.get('mean_kr') if r.get('mean_kr') is not None else '未完成'} |" for r in candidate_results]
else: lines += ['收尾时尚无候选模型完成full-navtest；检查 `full_candidate_eval/queue.log` 和各分片日志。']
lines += ['', '## 下一 AI 必做顺序','',
'1. 先读 `docs/PROJECT_MEMORY.md`、本文件、周期 `orchestrator.log`、`sweep_results.json`、`final_result.json`。',
'2. 核验 checkpoint 可加载、训练 `grad_norm` 非零、budget mean/方差确实变化；不要把训练 reward 当 navtest PDMS。',
'3. 若 full eval 未完成，使用最终 checkpoint 续跑缺失的 `rlcycle_*_g0..g7` 分片并去重聚合。',
'4. 若 PDMS 未超 `0.89199`，保持原 RL idea，优先做低风险参数/方差/模型选择修正；不得转成纯 SFT/DPO 或放弃 efficiency。',
'5. 最终必须补全动态 keep-ratio、FLOPs/latency 和 full-navtest N≈11576，只有同时满足性能与效率才能宣称达成。','',
'## 回滚','',
'- 核心训练脚本修改前备份：`backup/20260814_162603_rl_sota_cycle/train_scorer_budget_rl.py`。',
'- 停止当前链：仅终止命令行含本周期 ID 的进程；不要影响其他项目进程。','']
open(f'{root}/docs/journal/2026-08-15_rl_sota_handoff.md','w').write('\n'.join(lines))
print('\n'.join(lines))
PY

echo "[$(date)] handoff written: docs/journal/2026-08-15_rl_sota_handoff.md" | tee -a "$D/finalize.log"
