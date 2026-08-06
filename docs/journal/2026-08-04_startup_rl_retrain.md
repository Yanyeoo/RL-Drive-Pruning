# 2026-08-04 本周期启动：重新训练 Budget RL 投向 ICLR

> 时间窗口：H20×4，明天下午 18:00 回收（~24h）
> 前置状态：8×H20 全部空闲，无遗留进程，无遗留 python 进程

## 0. 事实核验

### 0.1 GPU 状态
```
0, NVIDIA H20, 0 MiB, 97871 MiB, 0 %
1, NVIDIA H20, 0 MiB, 97871 MiB, 0 %
2, NVIDIA H20, 0 MiB, 97871 MiB, 0 %
3, NVIDIA H20, 0 MiB, 97871 MiB, 0 %
```
**只有 4 张 H20 可用**（用户明确 H20×4）。

### 0.2 上一周期（8.3）产物

| 产物 | 状态 | 说明 |
|---|---|---|
| Block A 多 seed eval | ✅ 完成 | 8 seed，mean PDMS=0.86106，CSV 在 `results/raw/blockA_multiseed/` |
| sota_v3 R1 训练 | ✅ 完成 | 8 shard 各 151 steps，1 epoch，ckpt 在 `ckpt/s3_token_scorer_budget_rl_20260803_151608_sh{0..7}/` |
| sota_v3 R1 eval | ❌ 失败 | Hydra 报错，未产出 CSV。**根因**：`train()` 函数 stdout 日志混入 `$(train)` 返回值，导致 `BCKPT` 包含多行日志 → eval 的 `ckpt` 参数异常 |
| sota_v3 R2 训练 | ❌ 失败 | 同根因，`INIT_R2="$BCKPT"` 把日志前缀当作路径 → FileNotFoundError |
| sota_v3 R2 eval | ❌ 失败 | R2 训练未产出 ckpt（空目录） |

### 0.3 已训 ckpt 质量评估（R1 shard0 train_log）

| 指标 | 值 |
|---|---|
| Steps | 151 (1 epoch, 2404 scenes) |
| Mean reward | ~0.76 |
| Best reward | 0.7801 (step ~25, 存为 ckpt_best) |
| Final reward | ~0.76 |
| Mean kr | ~0.49 (初始 ~0.55 → 最终 ~0.34) |
| kr 趋势 | **持续下降**：efficiency_beta=0.15 的惩罚推得太狠 |

**关键观察**：
- kr 从 0.55 一路降到 0.34，远低于 target_kr=0.355
- driving_reward_proxy 均值 ~0.71，但最后几步 kr 太低导致 reward 下降
- 这是 adaptive_efficiency 惩罚（target-centric）过度压缩的结果

### 0.4 基线 PDMS（供对比）

| 方法 | PDMS | 备注 |
|---|---|---|
| SFT scorer r=0.5 raw | 0.89199 | 当前最强 baseline |
| Budget RL (旧, raw) | 0.87066 | AAAI 投稿版本 |
| Budget RL + denylist | 0.89125 | 同协议重建 |
| SparseVLM r=0.5 + fallback | 0.89063 | 同协议对比 |
| no-prune (full) | 0.89879 | 理论上界 |
| sota_v3 R1 (未 eval) | ？ | **本周期需补 eval** |

---

## 1. 决策：本周期策略

### 1.1 目标
重新训练 Budget RL，PDMS 达到 **> 0.892**（超过 SFT scorer baseline），为 ICLR 提供更强的 RL 结果。

### 1.2 根因分析

sota_v3 的问题：
1. **efficiency_beta=0.15 太高**：kr 从 0.55 降到 0.34，过度剪枝损害 PDMS
2. **adaptive_efficiency 的 target_kr=0.355 在训练早期就作为 hard penalty**：模型还没学会高质量 pruning 就被迫压缩
3. **ckpt_best 停在 step ~25**（early luck），不是真正的 best

### 1.3 策略：两阶段 RL

**Phase 1（今晚，4 GPU）**：温和 Budget RL
- 从 SFT scorer 初始化
- efficiency_beta=0.05（降低 3×），driving_scale=3.0
- 去掉 adaptive_efficiency，恢复线性 efficiency bonus
- safety_beta=0.05（降低安全惩罚）
- 1 epoch，~150 steps/shard
- 目标：kr 稳定在 ~0.45-0.50，PDMS > 0.88

**Phase 2（明天白天，如 Phase 1 结果好）**：Fine-tune + eval
- 从 Phase 1 best ckpt 继续 1 epoch
- 或者直接用 Phase 1 best ckpt 做全量 eval

### 1.4 时间线

```
18:30-19:00  修复脚本 + 对已有 sota_v3 R1 ckpt 做 quick eval（验证 eval 通路）
19:00-01:00  Phase 1 训练（4 GPU, 1 epoch, ~6h）
01:00-05:00  全量 navtest eval（4 shard, ~3.6h）
05:00-12:00  Phase 2 训练（如 Phase 1 PDMS > 0.88）
12:00-16:00  Phase 2 eval
16:00-18:00  汇总报告 + 备份
```

### 1.5 与 ICLR PLAN 的关系

原 ICLR PLAN 的目标 0.9107 不成立（factcheck 已确认）。新目标：
- **短期**（本周期）：Budget RL 达到 > 0.892 raw，证明 RL 能超过 SFT baseline
- **中期**（后续周期）：Per-token counterfactual REINFORCE（ICLR PLAN §二），目标 > 0.895

---

## 2. Reverse 指令

如果本周期决策被判定错误，回滚：
```bash
# 删除本周期新产物
rm -rf ckpt/s3_token_scorer_budget_rl_20260804_*/
rm -rf logs/sota_v4_*/
rm -rf exp/SOTAV4_*/
rm -rf results/raw/SOTAV4_*/
```

---

## 3. 环境状态

- 4× H20 空闲（0 MiB / 0%），无遗留进程
- Python: `/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python` (3.9.23)
- 训练数据: `data/navtrain_nocot/` (19225 scenes)
- 测试数据: `data/navtest_nocot/` (11596 scenes, 4 shard)
- SFT 初始化: `ckpt/s3_token_scorer/`
- VLA checkpoint: `models/AutoVLA/AutoVLA_PDMS_89.ckpt`
