# 2026-07-21 GPU Window 执行日志

**资源**: 5× H20 (97GB), 回收时间 2026-07-22 14:00
**启动时间**: 17:45

---

## 运行规则

1. 每个决策先做事实核验，不再"自信但没核"
2. 起任何进程前先看是不是有遗留进程，避免双开
3. 关键 artifact 操作前先备份（cp -a，不动原文件）
4. 任何偏离 PROMPT / design doc 的决策都当场写 journal，附理由 + reverse 指令
5. 实时更新 todo，每完成一项 immediate mark
6. **GPU 拉满**：5 卡全用上，不空闲
7. 每 30 分钟轮询进度 + 做决策；每小时汇报

---

## 启动时间线

| 时间 | 事件 | GPU | PID |
|------|------|-----|-----|
| 17:45 | RL shaped 4-shard 训练启动 | 0-3 | 7083-7086 |
| 17:45 | τ-cut 动态分析 (CPU) 启动 | — | 6545 |
| 17:52 | SparseVLM r=0.75 eval 启动 | 4 | 12379 |
| 17:52 | Progress monitor 启动 | — | 12380 |

---

## 已完成

### τ-cut 动态分析 ✅ (17:50 完成核心计算)

**结果**:
- N=4000 scenes, τ=-0.1668 (kr060)
- **std_keep_ratio = 0.085 > 0.05 ✅ (动态性证明)**
- range = [0.30, 0.92] (不同场景保留 30%~92% token)
- mean = 0.520, median = 0.507
- p10=0.426, p25=0.465, p75=0.560, p90=0.625
- histogram 已保存: `results/analysis/taucut_dynamic_histogram.png`
- stats JSON: `results/analysis/taucut_dynamic_stats.json`
- scatter plot: 待补（feature scenes 是 navtrain，baseline PDMS 是 navtest，不重叠）

**Note**: 缺少 difficulty correlation，因为 feature dump (navtrain 4000) 和 baseline PDMS (navtest 11574) 是不同的 scene set。如果需要 scatter，需用 navtest features 重跑，或者用 τ-cut eval 结果中的 per-scene PDMS 做相关性。

---

## 正在进行

### RL Shaped Reward 训练 (GPU 0-3)

- Config: LR=3e-5, KL=0.01, group=8, epochs=3, shaped=True
- Init: `ckpt/s3_token_scorer` (LambdaRank SFT)
- Baseline: `results/baseline_sub_scores.json` (11574 scenes)
- Output: `ckpt/s3_token_scorer_rl_shaped_20260721_174549_sh{0-3}`
- 状态: 模型加载中 (AutoVLA 3B ~2min)

### SparseVLM r=0.75 (GPU 4)

- Selector: sparsevlm_text, keep_ratio=0.75
- 4 shards 串行 on GPU4
- Output: `results/raw/tokenprune_S3_full/MT_sparsevlm_text_r075_sh{0-3}.csv`
- 后续接 Variant B r=0.75

---

## 路径变更记录

| 时间 | 变更 | 原因 |
|------|------|------|
| 17:45 | 新建 `ckpt/s3_token_scorer_rl_shaped_20260721_174549_sh{0-3}/` | RL 训练输出 |
| 17:45 | 备份 `ckpt/s3_token_scorer` → `ckpt/s3_token_scorer_backup_before_rl_20260721` | RL 启动前备份 |

---

## 后续计划

1. RL 训完 (~22:00) → 自动启动 RL eval (4卡, ~3h)
2. RL eval 完 (~01:00) → 如果 GPU 空出，启动 7B eval
3. SparseVLM + VarB r=0.75 (~20:00-22:00) → 补全论文数据

---

*记录时间: 2026-07-21 17:55*
