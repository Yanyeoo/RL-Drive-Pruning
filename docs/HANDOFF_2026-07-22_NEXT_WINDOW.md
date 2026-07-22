# 2026-07-22 窗口结束总结

**窗口**: 2026-07-21 17:45 → 2026-07-22 14:00 (约 20h)
**资源**: 5× H20 GPU
**总结时间**: 13:41

---

## 一、实验完成情况

### ✅ 已完成

| 实验 | 结果 | 备注 |
|------|------|------|
| **RL shaped reward 训练** (4 shard) | Best reward: sh3=0.3567, sh0=0.3337 | 全部完成 |
| **RL eval r=0.5** (full navtest) | **PDMS = 0.8909** | ❌ 未超 SFT (0.8920) |
| **RL eval r=0.25** (sh0) | PDMS = 0.8139 | — |
| **RL eval r=0.75** (sh0) | PDMS = 0.8855 | — |
| **RL + τ-cut** (sh0) | CSV 已生成 | — |
| **SparseVLM r=0.75** (4 shard) | 全部完成 | PDMS=0.9032 (sh3) |
| **Variant B r=0.75** (sh0 完成) | PDMS = 0.8850 | sh1 进行中 |
| **τ-cut 动态分析** | std=0.085, range=[0.30, 0.92] | ✅ 动态性证明 |
| **完整 denylist 生成** | 768 tokens | ✅ |
| **论文重写** (Abstract/Intro/Table/Discussion/Conclusion) | 已完成核心框架 | 数字待填 |

### 🔄 正在跑 (窗口结束前仍在执行)

| 实验 | GPU | 进度 | 预计完成 |
|------|-----|------|---------|
| **Budget RL 训练** (4 shard) | 0-3 | step 656, kr=0.42±0.07 | ~14:00-15:00 |
| **Variant B r=0.75** (sh1-3) | 4 | sh1 进行中 | ~16:00 |

### ❌ 未开始 (时间不够)

| 实验 | 原因 |
|------|------|
| Budget RL eval | Budget RL 还没训完 |
| ImpromptuVLA 7B nuScenes eval | 依赖 budget RL 完成 |
| Baseline pareto drop 版 (12个) | GPU4 还在跑 VarB r=0.75 |
| Safety-net 验证 | 排在 baseline pareto 后面 |

---

## 二、关键数字汇总

| 方法 | r=0.25 | r=0.5 | r=0.75 | 说明 |
|------|--------|-------|--------|------|
| No Prune | — | — | — | 0.8988 |
| **SFT Scorer VarB+fallback** | — | **0.9045** ✅ | — | 超越 baseline! |
| RL Scorer (shaped) | 0.8139 | 0.8909 | 0.8855 | ❌ 未超 SFT |
| SparseVLM (mask) | — | 0.8899 | 0.9032 | r=0.75 很强 |
| Budget RL | — | — | — | 训练中 (kr~0.42) |

### RL 未超 SFT 的分析

RL shaped (0.8909) vs SFT (0.8920)：差 0.11pt。可能原因：
1. RL eval 用的是 mask (Variant A)，不是 drop — 需要确认
2. shaped reward 的权重可能还需要调
3. 训练 epoch 不够 (3 epochs on 1/4 数据 per shard)

**但这不影响论文**：SFT VarB+fallback 已经是 0.9045。RL 可以放 ablation。

---

## 三、Budget RL 观察

- step 656, 平均 keep_ratio = 0.42 (自动学到了保留 ~42% 的 tokens)
- kr std = 0.07 (不同 scene 确实保留不同数量)
- 训练还在继续 (~1086 steps/epoch × 3 = ~3258 total)
- **如果窗口延长，需要等它训完 + eval 才有数字**

---

## 四、文件产出清单

### 新 ckpt
```
ckpt/s3_token_scorer_rl_shaped_20260721_174549_sh{0-3}/  — RL shaped (完成)
ckpt/s3_token_scorer_budget_rl_20260722_072423_sh{0-3}/  — Budget RL (进行中)
```

### 新 CSV 结果
```
results/raw/tokenprune_S3_full/MT_rl_shaped_r05_sh{0-3}.csv  — RL eval full
results/raw/tokenprune_S3_full/MT_rl_shaped_r025_sh0.csv     — RL r=0.25
results/raw/tokenprune_S3_full/MT_rl_shaped_r075_sh0.csv     — RL r=0.75
results/raw/tokenprune_S3_full/MT_rl_taucut_kr060_sh0.csv    — RL τ-cut
results/raw/tokenprune_S3_full/MT_sparsevlm_text_r075_sh{0-3}.csv  — SparseVLM r=0.75
results/raw/tokenprune_S3_full/MT_varBsafe_scorer_r075_sh0.csv     — VarB r=0.75
results/analysis/taucut_dynamic_stats.json                   — τ-cut 动态分析
results/analysis/taucut_dynamic_histogram.png
```

### 新代码
```
code/rldrive/scoring/token_scorer_budget.py    — Budget scorer 模型
scripts/train_scorer_budget_rl.py              — Budget RL 训练
scripts/run_rl_shaped_4gpu.sh                  — RL 训练脚本
scripts/run_rl_eval_4gpu.sh                    — RL eval + chain
scripts/run_budget_rl_4gpu.sh                  — Budget RL 训练脚本
scripts/run_budget_rl_eval.sh                  — Budget RL eval + chain 7B
scripts/run_7b_eval_dual.sh                    — ImpromptuVLA 7B eval
scripts/run_sparsevlm_r075_gpu4.sh             — SparseVLM + VarB (GPU4)
scripts/run_baseline_pareto_gpu4.sh            — Baseline drop 版补跑
scripts/wecom_heartbeat_v2.sh                  — 监控+心跳
```

### 论文
```
paper/aaai2027/main.tex                        — 重写版 (Abstract/Intro/Table/Disc/Conc)
paper/aaai2027/main_backup_20260721_2244.tex   — 备份
```

### 日志/文档
```
docs/journal/2026-07-21_decisions.md           — 决策记录
docs/journal/2026-07-21_session_summary.md     — Session 总结
docs/journal/2026-07-21_main_table_plan.md     — 主表计划
docs/journal/2026-07-21_gpu_window_execution.md — 执行日志
```

---

## 五、下次窗口 TODO

### P0 (必须)

1. **Budget RL eval** — 训练应该已完成/即将完成，直接 eval
2. **Baseline pareto drop 版** (12 个实验) — GPU4 排队
3. **SFT VarB r=0.25/0.75** — 补全主表 Ours 行
4. **ImpromptuVLA 7B eval** — 跨模型泛化数据
5. **Safety-net 验证** — 确认 entropy 检测有效

### P1 (优先)

6. **确认 RL eval 是否用了 drop** — 当前 RL 0.8909 可能是 mask 版本
7. **论文 Method section 加 Budget RL** — 写 §3.5 Budget Head
8. **填主表数字** — 等实验数据
9. **生成论文 figures** (Pareto curve, dynamic histogram, architecture fig)

### 止损

- 如果 Budget RL 也没超 SFT → 论文回退到 "SFT + τ-cut + VarB = 0.9045" 为主 contribution
- RL/Budget RL 放 ablation 讲 reward design insight
- 论文核心卖点变为: "learned scorer + τ-cut + VarB = surpass unpruned baseline"

---

## 六、环境/路径备忘

```
Python: /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python
项目根: /apdcephfs/private_shayladeng/tokenrl_autoVLA
环境变量: source scripts/setup_navsim_env_vars.sh
PYTHONPATH: $ROOT/code:$ROOT/code/third_party/AutoVLA/navsim:$ROOT/code/third_party/AutoVLA
```

---

*记录时间: 2026-07-22 13:41*
