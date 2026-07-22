# 2026-07-21 Session 完整梳理

**窗口**: 5× H20, 2026-07-21 17:45 → 2026-07-22 14:00 (~20h)
**记录时间**: 20:46

---

## 一、项目整体进度

### 论文核心数据 (3B AutoVLA + NAVSIM)

| 实验 | PDMS | 状态 | 论文位置 |
|------|------|------|---------|
| No prune baseline (r=1.0) | 0.8988 | ✅ 完成 | 主表 |
| **Variant B 真剪枝 + fallback** | **0.9045 (+0.57pt!)** | ✅ 数据已有，等 clean 重跑验证 | **主表主推** |
| SFT scorer r=0.5 (Variant A mask) | 0.8920 (−0.68pt) | ✅ 完成 | 主表对照 |
| SFT scorer r=0.75 | 0.8983 (p=0.58 不显著) | ✅ 完成 | Pareto |
| SFT scorer r=0.25 | 0.8508 | ✅ 完成 | Pareto |
| Attention L12 (teacher) r=0.5 | 0.8901 | ✅ 完成 | 主表 |
| MSE scorer r=0.5 | 0.8894 | ✅ 完成 | 主表 |
| Random r=0.5 | 0.8635 | ✅ 完成 | 主表 |
| FastV r=0.5 | 0.8330 | ✅ 完成 | 主表 |
| PruMerge r=0.5 | 0.8085 | ✅ 完成 | 主表 |
| SparseVLM r=0.5 | 0.8899 | ✅ 完成 | 主表 |
| SparseVLM r=0.75 | ? | 🔄 GPU4 跑中 (shard0: 2313/2949) | Pareto |
| τ-cut (kr060, 动态) | 0.8940 | ✅ 完成 | Pareto + 动态证据 |
| **RL shaped reward r=0.5** | ? | 🔄 GPU0-3 训练中 (step ~195/1086) | **主表 RL 行** |

### 动态性证据

| 指标 | 值 | 意义 |
|------|-----|------|
| τ-cut keep ratio std | 0.085 | ✅ >0.05 证明动态 |
| keep ratio range | [0.30, 0.92] | 不同场景保留 30%-92% tokens |
| histogram | ✅ 已生成 | 论文 figure |

### 效率数据

| 指标 | 值 |
|------|-----|
| FLOPs saving (r=0.5) | 33.6% |
| Wall-clock speedup (Variant B) | 15% (1.15×) |
| 序列压缩 (Variant B) | 38.3% |
| Fallback scenes | 6.6% (768/11576) |

### 7B Cross-Model (补充实验)

| 数据 | 状态 |
|------|------|
| 7B scorer (emb=3584, LambdaRank) | ✅ 已训好, pairwise acc=0.856 |
| 7B offline 分析 (attention 集中度) | ✅ top-25% tokens 占 95.9% attention |
| ImpromptuVLA 7B 模型 | ✅ 已下载 (7B_AD_finetune, driving fine-tuned) |
| nuScenes eval 数据 | ✅ 已就绪 (data/nuscenes_impromptu_val/) |
| ImpromptuVLA 7B eval | ⏳ 等 RL eval 完后自动启动 |

---

## 二、讨论后确定的关键决策

### 决策 1: 论文主表报 Variant B (真剪枝)

- **不报 Variant A (mask)**：mask 不省实际计算，只是 attention mask
- **报 Variant B + fallback**: 0.9045, 超越 no-prune baseline, 有真实加速
- **fallback 机制**: 运行时检测 degenerate output → 回退无剪枝，6.6% overhead

### 决策 2: 7B 实验用 ImpromptuVLA + nuScenes

- **不用 base Qwen2.5-VL-7B** (无 driving 能力，PDMS 无意义)
- **用 ImpromptuVLA 7B** (driving fine-tuned, 可出 L2/Collision 指标)
- **scorer 即插即用**: 7B scorer (base 模型训) → zero-shot 用在 fine-tuned ImpromptuVLA
- **对标 FastDriveVLA Table 1** (同模型、同 eval 口径)

### 决策 3: RL 和 SFT 关系

- **3B 主线**: RL 能超 SFT (0.8920) → 论文主贡献; 不能超 → negative insight + ablation
- **7B**: 只用 SFT scorer (emb 维度不同，RL scorer 不能跨)
- **两者独立报**: 3B 论文核心数据, 7B 是泛化性补充

### 决策 4: hidden_size 与"即插即用"的正确含义

- **方法论即插即用** (训练方法通用)，不是权重即插即用
- 给新模型只需: feature dump (~2h) + attention probe (~2h) + train scorer (~30s)
- 但 ImpromptuVLA 和 base Qwen2.5-VL-7B 同架构同维度 → 同一个 scorer 权重可以跨用

---

## 三、完整待办事项清单

### 正在执行 (GPU 占用中)

| # | 任务 | GPU | 进度 | 预计完成 |
|---|------|-----|------|---------|
| 1 | RL shaped reward 训练 (4 shard) | 0-3 | step ~195/1086 (18%) | ~23:50 |
| 2 | SparseVLM r=0.75 (shard0) | 4 | 2313/2949 (78%) | ~21:00 |
| 2b | SparseVLM r=0.75 (shard1-3) | 4 | 等 shard0 完 | ~02:00 |
| 2c | Variant B r=0.75 (4 shards) | 4 | 接 SparseVLM 后 | ~04:00 |

### 自动链式执行 (脚本已就绪)

| # | 任务 | 触发条件 | GPU | 预计时间 |
|---|------|---------|-----|---------|
| 3 | RL eval (full navtest, r=0.5) | RL 训练完 | 0-3 | ~00:00-03:00 |
| 4 | ImpromptuVLA 7B nuScenes eval | RL eval 完 | 0-3 | ~03:00-07:00 |

### 待人工确认/调度

| # | 任务 | 依赖 | 说明 |
|---|------|------|------|
| 5 | Variant B 完整 denylist (768 tokens) 重跑 | GPU 空闲 | denylist 已生成，验证 0.9045 |
| 6 | RL Pareto 补点 (r=0.25, r=0.75) | RL eval 结果好 | 如果 RL 赢了补曲线 |
| 7 | 论文写作 (Abstract + Intro) | 所有数据 | 28 号交稿 |

---

## 四、脚本清单与逻辑

### 已编写完成的脚本

| 脚本 | 功能 | 状态 |
|------|------|------|
| `scripts/run_rl_shaped_4gpu.sh` | RL 训练 4 卡 4 shard 并行 | ✅ 运行中 |
| `scripts/run_sparsevlm_r075_gpu4.sh` | SparseVLM r=0.75 + Variant B r=0.75 串行 | ✅ 运行中 |
| `scripts/run_rl_eval_4gpu.sh` | RL eval → 自动链到 7B eval | ✅ 就绪 |
| `scripts/run_7b_eval_dual.sh` | ImpromptuVLA 7B + nuScenes (4卡并行) | ✅ 就绪 |
| `scripts/wecom_heartbeat_v2.sh` | 进度监控 + 企业微信 + 自动决策 | ✅ 运行中 |

### GPU 持续满载 20h 的完整链式逻辑

```
时间轴 (17:45 → 14:00 次日):

GPU 0-3:
├─ [17:45 - 23:50] RL shaped reward 训练 (4 shard 并行)
├─ [23:50 - 03:00] RL eval on navtest (4 shard 并行) ← 自动触发
├─ [03:00 - 07:00] ImpromptuVLA 7B nuScenes eval (4 ratio 并行) ← 自动触发
└─ [07:00 - 14:00] 备用: Variant B denylist 重跑 / RL Pareto 补点

GPU 4:
├─ [17:52 - 21:00] SparseVLM r=0.75 shard0
├─ [21:00 - 01:00] SparseVLM r=0.75 shard1-3
└─ [01:00 - 07:00] Variant B r=0.75 (4 shards 串行)
```

### 稳定性保障

| 保障 | 实现 |
|------|------|
| 遗留进程检测 | 每个脚本开头 `pgrep -f` 检查，避免双开 |
| 自动链式触发 | `wecom_heartbeat_v2.sh` 每 30min 检测完成状态 → 启动下一步 |
| 企业微信通知 | 每 60min 推送 GPU 状态 + 进度 + 决策 |
| 超时保护 | 每个 eval 用 `timeout 40000` 防止卡死 |
| CSV 跳过 | `[[ -f "$CSV" ]] && skip`，已完成的不重跑 |
| Checkpoint 保存 | RL 每 50 step 保存 + best model |
| 备份 | SFT ckpt 已备份 (`ckpt/s3_token_scorer_backup_before_rl_20260721`) |
| Denylist 备份 | 旧版 66-token 备份到 `varB_catastrophic_tokens_66_backup.json` |

### 自动决策逻辑 (wecom_heartbeat_v2.sh)

```bash
每 30 分钟:
  1. nvidia-smi 检查 GPU 状态
  2. grep RL 日志 → 判断是否训完 (4 shard 都有 "DONE")
  3. 如果 RL 训完 + eval 没跑 → 自动 nohup run_rl_eval_4gpu.sh
  4. run_rl_eval_4gpu.sh 末尾自动调 run_7b_eval_dual.sh
  5. 每 60min 推企业微信汇报
```

---

## 五、关键文件路径索引

| 用途 | 路径 |
|------|------|
| RL 训练输出 | `ckpt/s3_token_scorer_rl_shaped_20260721_174549_sh{0-3}/` |
| SFT scorer (主) | `ckpt/s3_token_scorer/` |
| 7B scorer | `ckpt/s3_token_scorer_7b/` |
| Baseline sub-scores | `results/baseline_sub_scores.json` (11574 scenes) |
| 完整 denylist | `results/varB_catastrophic_tokens.json` (768 tokens) |
| τ-cut 分析结果 | `results/analysis/taucut_dynamic_stats.json` |
| ImpromptuVLA 模型 | `models/ImpromptuVLA_7B/7B_AD_finetune/` |
| nuScenes 数据 | `data/nuscenes_impromptu_val/unpacked/` |
| 企业微信心跳 log | `logs/wecom_heartbeat_v2.log` |
| RL 训练 log | `logs/rl_shaped_sh{0-3}.log` |
| 监控决策 log | `logs/monitor_decisions.log` |

---

*记录时间: 2026-07-21 20:46*
