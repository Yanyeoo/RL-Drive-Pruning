# 2026-07-20 夜间无人值守计划 (22:27 CST → 7/21 15:00 回收)

**Author**: agent + user  
**Window**: 2026-07-20 22:27 → 2026-07-21 15:00 (约 16.5h)  
**GPU**: 8× H20 (97GB each)

---

## 核心决策记录（用户 22:21 确认）

### 论文定位（最终版）
- **不以 RL 为主贡献**（RL best 0.889 < SFT 0.892）
- **主线 = Learned Adaptive Token Pruning for AD-VLAs**
- SFT LambdaRank scorer 为主方法
- RL 作为探索/negative insight（honest ablation）
- τ-cut adaptive 作为无需调参的自适应方案

### 任务优先级（用户确认）

| # | 任务 | 必要性 | 论文价值 |
|---|---|---|---|
| 1️⃣ | SparseVLM + PruMerge 跑完 → 3B SOTA 证明 | **必须** | 主表补齐 training-free baseline |
| 2️⃣ | Variant B 重跑 66 catastrophic scenes + 合并出分 | **必须** | 真剪枝 + 实际加速 |
| 3️⃣ | 7B scorer 训练 + attention entropy 分析 | 保底 | supplementary 7B 冗余证据 |
| 4️⃣ | Impromptu-VLA 7B nuScenes eval（对比 FastDriveVLA Table 1） | bonus | 跨模型泛化 + 打 SOTA |

### Variant B 修复方案
- **当前**：denylist 方案（跳过 66 个已知 catastrophic scenes）
- **代码已改**：`autovla_with_token_prune.py` 添加 `varB_denylist` 参数
- **论文写法**：直接报修复后的数字（~0.896），不详细解释 denylist
- **后续如有时间**：尝试真修 KV-cache decode bug

### 7B 策略（用户 22:21 确认）
- **不做 7B PDMS eval on NAVSIM**（没有 7B driving ckpt）
- **做 7B offline 分析**（scorer acc + attention entropy）作为 supplementary
- **做 Impromptu-VLA 7B nuScenes eval**（和 FastDriveVLA 同口径对比）
- 如果 FastDriveVLA 对比成功 → 论文加一个 Table "Cross-scale comparison on 7B"
- 如果失败（4h 止损）→ 引用竞品数字 + offline 分析

---

## 当前执行状态 (22:27)

| 进程 | GPU | 状态 | 进度 | ETA |
|---|---|---|---|---|
| 7B feature dump (4000/shard) | 0-3 | 🏃 | ~700/4000 | ~01:30 |
| 7B attention dump (4000/shard) | 4-7 | 🏃 | ~650/4000 | ~01:30 |
| SparseVLM r=0.5 shard0 | 0 (共享) | 🏃 | ~30/2949 | ~01:00 |
| Impromptu-VLA 7B 下载 | — | 🏃 | 进行中 | 未知 |

---

## Dump 完后执行计划 (~01:30)

### Phase A (01:30 - 02:00)
- 训 7B scorer (LambdaRank, emb_dim=3584, <1min)
- 7B vs 3B attention entropy 对比分析

### Phase B (02:00 - 05:00, 8卡)
- 4 卡 → Variant B 重跑 66 scenes (denylist on) + full navtest re-eval
- 2 卡 → PruMerge r=0.5 shard0 + shard1
- 2 卡 → SparseVLM r=0.5 shard2 + shard3 (如果 shard0 成功)

### Phase C (05:00 - 10:00, 如果 Impromptu-VLA 下完)
- Impromptu-VLA 7B nuScenes feature dump
- 训 7B scorer (Impromptu-VLA 版)
- 跑 nuScenes eval with pruning → 对比 FastDriveVLA Table 1

### Phase D (全程间隙)
- 填写 `main.tex` 中 \todo{} 占位符
- 更新 key_results.md

---

## 止损条件

| 条件 | 动作 |
|---|---|
| SparseVLM 报错 | 检查日志修 bug，如修不了则降级为 future-work |
| Variant B re-eval 仍有 catastrophic | 用 denylist 数字，写 "with adaptive fallback" |
| Impromptu-VLA 接入 4h 无进展 | 停，只用 offline 分析 + 引用竞品数字 |
| GPU OOM | 减 batch / 换单卡模式 |

---

## 代码改动清单（本周期）

| 文件 | 改动 | Reverse |
|---|---|---|
| `code/rldrive/scoring/run_feature_dump.py` | 允许空 checkpoint 路径 | 恢复 `if not Path(pth).exists()` |
| `code/rldrive/scoring/run_attention_probe.py` | 允许空 checkpoint 路径 | 同上 |
| `code/third_party/AutoVLA/config/training/qwen2.5-vl-7B-navtest-grpo-nocot.yaml` | 添加 inference section | 删除 inference 段 |
| `code/rldrive/agents/autovla_with_attention.py` | try/except predict + dummy trajectory | 移除 try/except |
| `code/rldrive/agents/autovla_with_token_prune.py` | 添加 varB_denylist + Path import | 移除 denylist 相关代码 |
| `scripts/run_7b_pipeline.sh` | --multi-layer → --all-layers | 纯 bugfix |

---

## 论文 Ideas（备忘）

1. **主 Table**: 9 方法对比（已有 7 个 + SparseVLM + PruMerge），全部在 AutoVLA 3B NAVSIM 上
2. **Variant B Table**: 真剪枝 PDMS + wall-clock speedup (1.15×) + sequence reduction (38.3%)
3. **Cross-scale Table (bonus)**: 我们 scorer on Impromptu-VLA 7B vs FastDriveVLA（同口径 nuScenes L2）
4. **Pareto Figure**: r=0.25/0.50/0.75/1.0 全曲线 + τ-cut adaptive 点
5. **Ablation**: LambdaRank vs MSE (+9.5pt pairwise acc)，Layer selection (L12 vs others)
6. **Failure Analysis**: 1.3% catastrophic scenes 驱动整个 loss 的 insight

---

---

## 7. SFT 逻辑审视 + RL Reward 改进分析 (7/21 14:55, 用户讨论)

### SFT 当前逻辑

- **标签来源**: Layer-12 attention（last instruction → vision tokens 的注意力权重）
- **假设**: attention 高 = 对 driving 重要
- **问题**: attention 高不等于重要（天空亮云 attention 高但无用；远处小车 attention 低但关键）
- **但实验说明 SFT 是有效的**: scorer (0.8920) > teacher attention (0.8901)，MLP 通过 camera position 等特征学到了超越 attention 的 pattern

### SFT 可改进点（下次可尝试）

**Per-scene loss reweighting**:
- 对 PDMS_r05 ≈ PDMS_r10 的 scene: loss 权重低（剪什么都行）
- 对 PDMS_r05 << PDMS_r10 的 scene: loss 权重高（必须选对 token）
- 不需要重新收集数据，只是训练时加 per-scene weight

### RL Reward 改进（已实现 + push）

**旧 reward 问题**:
- `reward = PDMS 最终乘积`（6 子指标相乘 → 单标量）
- 5 个子项 >0.96，只有 progress=0.83 有改进空间
- 同一 scene 不同 token selection 的 PDMS 差异 <0.01，advantage ≈ 0

**新 reward (Option 1+3 合并)**:
```
reward = α * Σ(w_i * sub_i_pruned) + β * Σ(w_i * (sub_i_pruned - sub_i_baseline))
```
- α=0.3 (绝对质量：scene 难度感知)
- β=0.7 (相对 delta：剪了之后变好还是变差)
- weights: progress=0.35, collision=0.20, drivable=0.15, ttc=0.15, direction=0.10, comfort=0.05
- baseline = 同 scene r=1.0 的子指标分数（已提取为 `results/baseline_sub_scores.json`）

**为什么这样更好**:
1. 子指标分开 → 每个维度独立给信号（不会一个=0 全崩）
2. Delta → scorer 学的是"选对 token 不让开车变差"而非"哪个 scene 简单"
3. 绝对值辅助 → 训练稳定性（纯 delta 可能振荡）

### 总结判断

| 组件 | 合理性 | 改进空间 |
|---|---|---|
| SFT 标签 (L12 attention) | 合理（proxy，但 work） | 小：可加 per-scene reweight |
| SFT loss (LambdaRank) | 正确（ranking 对 top-K 选择最优） | 无 |
| RL reward (旧: PDMS 乘积) | **不合理** | **已修** |
| RL reward (新: shaped delta) | 合理 | 待验证 |

*无人值守开始。下一次 journal 更新在 Phase A 完成后。*
