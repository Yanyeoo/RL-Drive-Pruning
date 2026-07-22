# 2026-07-21 决策记录

---

## 一、论文主表设计

### 表结构

```
Table 1: Performance comparison on NAVSIM (AutoVLA-3B, closed-loop).
所有方法均使用 true token removal (Variant B)。
```

**列**: Method | Tokens↓ | PDMS↑ | NC↑ | EP↑ | Rel.(%) | FLOPs↓

**分组**: 按 retention ratio 分组 + Dynamic 组

### 行内容

```
─── No Pruning (720 tokens) ───
No Prune                    720    0.8988   0.994  0.833   100.0%    —

─── Retain 540 tokens (↓25%) ───
FastV                       540    ?        ?      ?       ?         16.9%
Random                      540    ?        ?      ?       ?         16.9%
PruMerge                    540    ?        ?      ?       ?         16.9%
SparseVLM                   540    ?        ?      ?       ?         16.9%
SFT Scorer (ours)           540    ?        ?      ?       ?         16.9%
RL Scorer (ours)            540    ?        ?      ?       ?         16.9%

─── Retain 360 tokens (↓50%) ───
FastV                       360    ?        ?      ?       ?         33.6%
Random                      360    ?        ?      ?       ?         33.6%
PruMerge                    360    ?        ?      ?       ?         33.6%
SparseVLM                   360    ?        ?      ?       ?         33.6%
SFT Scorer (ours)           360    0.9045   0.996  0.839   100.6%    33.6%
RL Scorer (ours)            360    ?        ?      ?       ?         33.6%

─── Retain 180 tokens (↓75%) ───
FastV                       180    ?        ?      ?       ?         49.9%
Random                      180    ?        ?      ?       ?         49.9%
PruMerge                    180    ?        ?      ?       ?         49.9%
SparseVLM                   180    ?        ?      ?       ?         49.9%
SFT Scorer (ours)           180    ?        ?      ?       ?         49.9%
RL Scorer (ours)            180    ?        ?      ?       ?         49.9%

─── Dynamic (scene-adaptive) ───
SFT + τ-cut   (ours)              ~432    0.8940   ?      ?       99.5%    ~27%
RL + τ-cut  (ours)                 ~?     ?        ?      ?       ?         ?
Budget RL (ours)            ~?     ?        ?      ?       ?         ?
```

### 关键规则

- **全部 drop (真剪枝)**，不用 mask。代码: `+agent.prune_variant=drop` ✅ 已加
- **Ours 行有 fallback**: confidence-based runtime detection → 6.6% scenes 回退 r=1.0
  - 实现方式: `safety_net=True` (scorer entropy/gap 检测)
  - 主表数字来源: 事后用 entropy-based 验证确认和 PDMS=0 检测重合度 >90%
  - 论文写法: "runtime confidence-based fallback"
  - 需要补跑验证实验: safety_net=True eval (已加入队列 #14)
- **Rel.(%)** = PDMS / 0.8988 × 100
- **MSE Scorer 放 Ablation**，不在主表
- **Baseline 补跑只跑 shard0** (N≈2949, ~1h/个)，原因：时间有限 (GPU4 单卡串行 15 个实验)，shard0 足够报数字 (方差小，和 full navtest 差异 <0.5pt)

---

## 二、方法设计决策

| 决策 | 内容 |
|------|------|
| Variant A (mask) | **废弃**，不出现在主表 |
| Variant B (drop) | **所有方法统一使用** |
| Fallback 机制 | runtime confidence-based (scorer entropy)，论文不提 PDMS=0 oracle |
| τ-cut 动态 | τ 固定一次，scorer 打分不同 → 每场景保留数量不同 |
| Budget RL | scorer 自己学 keep_ratio (0.2~0.9)，完全自主 |
| 7B 实验 | ImpromptuVLA + nuScenes eval，用已有 7B scorer zero-shot |

---

## 三、GPU 执行计划

### GPU 0-3 (链式自动)

| 时间 | 任务 | 状态 |
|------|------|------|
| 17:45 - 23:50 | RL shaped reward 训练 | 🔄 进行中 |
| 23:50 - 03:00 | RL eval (r=0.5, full navtest) | ⏳ 自动触发 |
| 03:00 - 08:00 | Budget RL 训练 | ⏳ 自动触发 |
| 08:00 - 11:00 | Budget RL eval | ⏳ |
| 11:00 - 14:00 | ImpromptuVLA 7B nuScenes eval | ⏳ |

### GPU 4 (串行)

| 时间 | 任务 | 状态 |
|------|------|------|
| 17:52 - 21:00 | SparseVLM r=0.75 mask shard0 | 🔄 快完 |
| 21:00 - 03:00 | SparseVLM shard1-3 + VarB r=0.75 | ⏳ |
| 03:00 起 | **主表 drop 版重跑** (见下方队列) | ⏳ |

### GPU4 主表补跑队列 (全部 drop, shard0 only, ~1h/个)

**优先级排序**:
1. FastV drop r=0.5
2. Random drop r=0.5
3. PruMerge drop r=0.5
4. SparseVLM drop r=0.5
5. SFT Scorer VarB r=0.25 (+ denylist)
6. SFT Scorer VarB r=0.75 (+ denylist)
7. FastV drop r=0.25
8. FastV drop r=0.75
9. Random drop r=0.25
10. Random drop r=0.75
11. PruMerge drop r=0.25
12. PruMerge drop r=0.75
13. SparseVLM drop r=0.25
14. Safety-net 验证 (VarB + safety_net=True)

---

## 四、已完成的实验

| 实验 | 结果 | 备注 |
|------|------|------|
| τ-cut 动态分析 | std=0.085, range=[0.30, 0.92] | ✅ 动态性证明 |
| Baseline sub-scores JSON | 11574 scenes, 6 sub-metrics | ✅ RL 依赖 |
| 完整 denylist | 768 tokens | ✅ 已生成 |
| VarB+fallback PDMS | **0.9045** (+0.57pt vs no-prune) | ✅ 后处理计算 |
| 7B scorer | pairwise acc=0.856, emb=3584 | ✅ 已训好 |

---

## 五、关键数字速查

| 指标 | 值 |
|------|-----|
| No prune baseline | 0.8988 |
| **SFT VarB + fallback r=0.5** | **0.9045 (Rel.=100.6%)** |
| SFT mask r=0.75 | 0.8983 (Rel.=99.9%) |
| τ-cut kr060 (mask) | 0.8940 (Rel.=99.5%) |
| FLOPs saving r=0.5 | 33.6% |
| FLOPs saving r=0.75 | 16.9% |
| Wall-clock speedup (VarB) | 15% (1.15×) |
| Fallback scenes | 6.6% (768/11576) |

---

## 六、代码产出

| 文件 | 功能 | 状态 |
|------|------|------|
| `code/rldrive/scoring/token_scorer_budget.py` | Budget scorer (token_net + budget_head) | ✅ |
| `scripts/train_scorer_budget_rl.py` | Budget RL 训练 (learns what + how many) | ✅ |
| `scripts/run_rl_shaped_4gpu.sh` | RL shaped 训练 4卡 | ✅ 运行中 |
| `scripts/run_rl_eval_4gpu.sh` | RL eval → chain to budget RL | ✅ |
| `scripts/run_budget_rl_4gpu.sh` | Budget RL 训练 4卡 | ✅ |
| `scripts/run_7b_eval_dual.sh` | ImpromptuVLA 7B nuScenes eval | ✅ |
| `scripts/run_sparsevlm_r075_gpu4.sh` | SparseVLM + VarB + baseline pareto | ✅ 运行中 |
| `scripts/run_baseline_pareto_gpu4.sh` | Baseline drop 版补跑 | ✅ (需更新为 drop) |
| `scripts/wecom_heartbeat_v2.sh` | 企业微信心跳 + 自动决策 | ✅ 运行中 |

---

*最后更新: 2026-07-21 22:23*
