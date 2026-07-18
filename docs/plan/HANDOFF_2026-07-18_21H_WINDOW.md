# HANDOFF — RL-Drive-Pruning / AutoVLA token pruning（给无记忆 AI）

**写入时间**：2026-07-18 17:50 CST  
**项目根**：`/apdcephfs/private_shayladeng/tokenrl_autoVLA`  
**Python**：`/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python`  
**环境变量**：`source scripts/setup_navsim_env_vars.sh`  
**GPU**：8× H20；18:00 回收，本交接面向下一次 **21h 窗口**。

---

## 0. 接手后必须先做

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
source scripts/setup_navsim_env_vars.sh
pgrep -af 'run_pdm_score_cot.py|train_scorer_grpo.py|python' || true
nvidia-smi
```

规则：
1. 每个决策先事实核验。
2. 起进程前 `pgrep` 查残留，避免双开。
3. 改关键 artifact 前 `cp -a` 备份。
4. 偏离本 handoff/design 时立刻写 `docs/journal/YYYY-MM-DD.md`，附理由和 reverse 指令。
5. 实时更新 todo。
6. 企业微信每 15 分钟发进度、当前思考、下一步计划。

---

## 1. 当前结论（一句话）

**不要继续把 RL 或 Variant B 当主线赌。**  
当前最稳论文主线是：

> **Driving-task-aligned adaptive vision token pruning for AD-VLA**：SFT scorer + calibrated τ-cut adaptive，显著优于 random/FastV，`r=0.75` 近似 free lunch，`τ-cut kr060` full-navtest 有 adaptive 点。

RL 目前作为 negative/探索 ablation；Variant B 目前作为 engineering risk/future deployment path。

---

## 2. 关键结果表与入口

### 2.1 Full navtest 主表（权威 CSV 入口）

目录：`results/raw/tokenprune_S3_full/`

| 方法 | CSV pattern | N | PDMS | 结论 |
|---|---|---:|---:|---|
| no-prune r=1.0 | `MT_attn_L12_r10_sh*.csv` | ≈11575 | 0.8988 | baseline |
| scorer SFT r=0.75 | `MT_scorer_r075_sh*.csv` | 11572 | **0.898305** | free lunch, −0.05pt |
| scorer SFT r=0.5 | `MT_scorer_r05_sh*.csv` | 11571 | **0.891990** | 主操作点 |
| scorer SFT r=0.25 | `MT_scorer_r025_sh*.csv` | 11573 | 0.850820 | aggressive cliff |
| attn_L12 r=0.5 | `MT_attn_L12_r05_sh*.csv` | ≈11577 | 0.8901 | teacher baseline |
| random r=0.5 | `MT_random_r05_sh*.csv` | ≈11576 | 0.8635 | weak baseline |
| FastV L2 r=0.75 | `MT_fastv_l2_r075_sh*.csv` | 11575 | 0.882258 | generic baseline 很差 |
| FastV L2 r=0.5 | `MT_fastv_l2_r05_sh*.csv` | 11573 | 0.832961 | generic baseline 崩 |
| Variant B safe r=0.5 | `MT_varBsafe_scorer_r05_sh*.csv` | 11576 | 0.872533 | 失败，不主打 |

聚合命令：

```bash
/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python - <<'PY'
import glob, os, pandas as pd
for pat,name in [('results/raw/tokenprune_S3_full/MT_scorer_r*.csv','scorer'),('results/raw/tokenprune_S3_full/MT_fastv_l2_r*.csv','fastv')]:
 print('\n['+name+']')
 groups={}
 for f in glob.glob(pat):
  key=os.path.basename(f).rsplit('_sh',1)[0]
  groups.setdefault(key,[]).append(f)
 for k,fs in sorted(groups.items()):
  m=pd.concat([pd.read_csv(f).query("token!='average'") for f in fs]).drop_duplicates('token')
  v=m[m.valid]
  print(k, len(fs), len(v), v.score.mean())
PY
```

### 2.2 τ-cut adaptive

目录：`results/raw/tokenprune_taucut/`

| τ-cut tag | CSV pattern | status | N | PDMS |
|---|---|---|---:|---:|
| kr040 | `TC_mse_tau_kr040_sh0.csv` | shard0 only | 2949 | 0.880680 |
| kr050 | `TC_mse_tau_kr050_sh0.csv` | shard0 only | 2949 | 0.890452 |
| kr060 | `TC_mse_tau_kr060_sh*.csv` | **full** | 11575 | **0.893951** |
| kr070 | `TC_mse_tau_kr070_sh0.csv` | shard0 only | 2949 | 0.893662 |

21h 窗口建议补：`kr050 sh1-3`、`kr070 sh1-3`，可选 `kr040 sh1-3`。

脚本入口：`scripts/run_taucut_fullnavtest.sh`，但原脚本会 4 shard 全跑且 skip done。示例：

```bash
nohup bash scripts/run_taucut_fullnavtest.sh <tau> kr050 > logs/_taucut_kr050_full.log 2>&1 &
```

τ 值需要从旧日志/quick test 查；不要猜，先 `grep -R "kr050\|kr070" logs docs results -n`。

### 2.3 RL / GRPO 结果

代码：`scripts/train_scorer_grpo.py`  
训练产物：

- `ckpt/s3_token_scorer_rl_20260717_230129_sh0/`
- `ckpt/s3_token_scorer_rl_20260717_230129_sh1/`
- `ckpt/s3_token_scorer_rl_20260717_230129_sh2/`
- `ckpt/s3_token_scorer_rl_20260717_230129_sh3/`

训练是 4-way scene-shard 并行，不是 DDP 同步梯度；已在 journal 记录偏离。

Q1500 quick eval 入口：`results/raw/tokenprune_S3_quick/Q1500_*.csv`

| rank | candidate | N | PDMS |
|---:|---|---:|---:|
| 1 | `Q1500_rlsh3best_r05.csv` | 1495 | **0.889109** |
| 2 | `Q1500_rlsh0best_r05.csv` | 1495 | 0.883768 |
| 3 | `Q1500_rlsh1best_r05.csv` | 1495 | 0.882968 |
| 4 | `Q1500_rlsh2best_r05.csv` | 1495 | 0.882016 |
| 5 | `Q1500_rlsh0final_r05.csv` | 1495 | 0.873399 |
| 6 | `Q1500_rlsh2final_r05.csv` | 1495 | 0.873136 |
| 7 | `Q1500_rlsh3final_r05.csv` | 1494 | 0.864597 |
| 8 | `Q1500_rlsh1final_r05.csv` | 1494 | 0.861224 |

同 split 的 SFT scorer r=0.5：`0.895262`（见 `docs/results/key_results.md §10.2` / `results/raw/tokenprune_S3/S3sub1500_scorer_r050.csv`）。

**结论**：RL best 仍低 SFT `~0.615pt`；final 全部更差，说明 RL 过训/漂移。不要把 RL 作为主 claim，除非后续 objective 改造后重新验证。

### 2.4 Variant B

代码入口：

- `code/rldrive/agents/token_prune_patch_varB.py`
- `code/rldrive/agents/autovla_with_token_prune.py`

结果入口：

- 原 shard0：`exp/MT_varB_scorer_r05_sh0/2026.07.15.15.42.07/2026.07.15.19.13.52.csv`
- catastrophic token list：`results/varB_catastrophic_tokens.json`
- safe fallback full：`results/raw/tokenprune_S3_full/MT_varBsafe_scorer_r05_sh*.csv`

结果：

- 原 Variant B shard0 `PDMS=0.8758`，排除 66 catastrophic 后 B > A。
- safe fallback full `PDMS=0.872533`，更明确失败。

**结论**：bug 不只是短 decode；疑似 M-RoPE / `position_ids` / `attention_mask` / `cache_position` / sequence surgery 对齐问题。21h 窗口不要优先修它，除非主表已锁。

---

## 3. 21h 窗口推荐计划

### P0：补齐 2 个 generic baseline（最重要）

目标：`SparseVLM` + `ToMe` 或 `LLaVA-PruMerge/CLS-attn prune`。

现有 related-work 指引：`docs/related_work_survey.md §2.2/2.3/2.6`。

建议实现为 `AutoVLAWithTokenPruneAgent` 的新 selector：

- `sparsevlm_text`: text-guided attention score（训练无关）
- `prumerge_cls` 或 `tome_merge`: 若 ToMe merge 工程风险大，优先做 CLS/text-attn prune 版本保证出结果

必须跑：

- `r=0.5` full 4 shards
- `r=0.75` full 4 shards

时间预算：实现/烟测 2-4h；full eval 两轮约 7-8h。

### P1：补齐 τ-cut curve

补 full：

- `kr050 sh1-3`
- `kr070 sh1-3`
- 可选 `kr040 sh1-3`

目标：有完整 adaptive Pareto，不只一个 `kr060` 点。

### P2：补 MSE fixed-r ablation

MSE ckpt：`ckpt/s3_token_scorer_mse/`

建议至少跑：

- MSE scorer fixed `r=0.5` full
- 有空跑 `r=0.75` full

用途：证明 LambdaRank/SFT scorer 训练选择优于 MSE fixed-r；同时 τ-cut 用的是 MSE calibrated scorer。

### P3：整理论文表和 journal

必须更新：

- `docs/results/key_results.md`
- `docs/journal/2026-07-18.md` 或新日期 journal
- 如果有 paper 表格，更新 `paper/aaai2027/main.tex`

### P4：Variant B 深修（仅 P0-P3 完成后）

只做 66 catastrophic token 最小复现，不再 full 盲跑。

---

## 4. 无人值守 / 企业微信 15 分钟进度

现有脚本：`scripts/wecom_progress_heartbeat.sh`

已修复：原 `urllib` 证书错误，已改用 `curl`。

### 启动/重启 15 分钟 heartbeat

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
chmod +x scripts/wecom_progress_heartbeat.sh
if [ -f logs/wecom_progress_heartbeat.pid ]; then kill "$(cat logs/wecom_progress_heartbeat.pid)" 2>/dev/null || true; fi
INTERVAL_SECONDS=900 nohup scripts/wecom_progress_heartbeat.sh >> logs/wecom_progress_heartbeat.log 2>&1 &
echo $! > logs/wecom_progress_heartbeat.pid
```

检查：

```bash
pid=$(cat logs/wecom_progress_heartbeat.pid)
ps -p "$pid" -o pid,etime,cmd
```

手动发一次企业微信：

```bash
curl -sS -X POST 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82' \
  -H 'Content-Type: application/json' \
  -d '{"msgtype":"text","text":{"content":"RL-Drive-Pruning heartbeat test: 接手成功，15分钟进度汇报已启动。"}}'
```

### 15 分钟汇报内容要求

每次企业微信必须包含：

1. 当前运行进程 / GPU 状态。
2. 最近 15 分钟 artifact / CSV / checkpoint。
3. 当前思考：结果是否符合预期；是否要停/切换/补实验。
4. 下一步计划：下一 15-60 分钟做什么。
5. 若偏离计划，说明理由和 reverse 指令已写 journal。

---

## 5. 常用启动模板

所有启动前先：

```bash
pgrep -af 'run_pdm_score_cot.py|train_scorer_grpo.py' || true
nvidia-smi
```

### 4-shard eval 模板

参考 `scripts/run_fastv_baseline.sh` 和 `scripts/run_taucut_fullnavtest.sh`。核心 hydra 参数：

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
source scripts/setup_navsim_env_vars.sh
export PYTHONPATH="$PWD/code:$PWD/code/third_party/AutoVLA/navsim:$PWD/code/third_party/AutoVLA:${PYTHONPATH:-}"
```

`run_pdm_score_cot.py` 关键参数：

- `train_test_split=navtest_local_filtered_shard${sh}_20260616_154858`
- `metric_cache_path=$ROOT/data/navtest_metric_cache`
- `+json_data_path=$ROOT/data/navtest_nocot`
- `agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent`
- `+agent.keep_ratio=0.5/0.75`
- `+agent.selector=<selector>`
- `+agent.scorer_ckpt=<ckpt>` if selector uses scorer

---

## 6. 当前文件改动/备份

重要备份：

- `backups/start_20h_20260717_230020/`
- `backups/varB_safe_20260717_230201/`

已改文件：

- `scripts/train_scorer_grpo.py`：增加 `--num-shards/--shard-id`。
- `code/rldrive/agents/autovla_with_token_prune.py`：加入 Variant B 短 trajectory fallback（但结果证明不够）。
- `scripts/wecom_progress_heartbeat.sh`：改用 `curl`，建议 15 分钟 interval。

如需回滚，优先看上述 backup 和 `docs/journal/2026-07-17.md`、`docs/journal/2026-07-18.md`。

---

## 7. 最后建议

21h 窗口不要再用大量 GPU 跑 RL full eval；当前证据表明 RL 不赢 SFT。  
先把 **SparseVLM + ToMe/PruMerge + τ-cut full curve + MSE ablation** 跑完，论文就能闭环。RL 和 Variant B 都可作为 appendix/negative/future work。
