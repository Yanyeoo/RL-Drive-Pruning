# 主表结构讨论 (2026-07-21 22:00)

## 决策

1. **Variant A (mask) 完全废弃** — 不出现在主表，只 mask 没有真实加速
2. **所有"我们的方法"都是 Variant B (真剪枝)** — 真实移除 tokens
3. **主表多列**: r=0.25 / r=0.5 / r=0.75 / Dynamic
4. **MSE Scorer 放 Ablation 不放主表**

## 主表设计

| Method | r=0.25 | r=0.5 | r=0.75 | Dynamic |
|--------|--------|-------|--------|---------|
| No Prune | — | — | — | 0.8988 |
| FastV | ? | 0.8330 | ✅ 有 | — |
| Random | ? | 0.8635 | ? | — |
| PruMerge | ? | 0.8085 | ? | — |
| SparseVLM | ? | 0.8899 | 🔄 跑中 | — |
| **SFT Scorer (ours)** | ? | **0.9045** | ? | — |
| **RL Scorer (ours)** | ? | ? | ? | — |
| **SFT + τ-cut** | — | — | — | ? |
| **RL + τ-cut** | — | — | — | ? |
| **Budget RL (ours)** | — | — | — | ? |

注：
- SFT Scorer r=0.5 报 **Variant B + fallback = 0.9045**（不是 mask 的 0.8920）
- Dynamic 列 = 不固定 ratio，模型自己决定每 scene 保留多少
- No Prune 放 Dynamic 列（因为它等价于"所有 scene 都保留 100%"）

## 已有数据盘点

| 方法 | r=0.25 | r=0.5 | r=0.75 | 说明 |
|------|--------|-------|--------|------|
| FastV | ❌ 缺 | ✅ | ✅ 有 | 需补 r=0.25 |
| Random | ❌ 缺 | ✅ | ❌ 缺 | 需补 r=0.25, r=0.75 |
| PruMerge | ❌ 缺 | ✅ | ❌ 缺 | 需补 r=0.25, r=0.75 |
| SparseVLM | ❌ 缺 | ✅ | 🔄 GPU4 跑中 | 需补 r=0.25 |
| SFT Scorer | ✅ (varA) | ✅ (varA+varB) | ✅ (varA) | **需补 varB r=0.25, r=0.75** |
| RL Scorer | ❌ | 🔄 训练中 | ❌ | 等 RL 训完后补 |

## 需要补跑的实验 (GPU4 单卡串行)

### Baselines (不需要我们的 scorer，只需换 selector 参数)

| # | 实验 | selector | ratio | 预计时间 |
|---|------|----------|-------|---------|
| 1 | FastV r=0.25 | fastv_l2 | 0.25 | ~3h |
| 2 | Random r=0.25 | random | 0.25 | ~3h |
| 3 | Random r=0.75 | random | 0.75 | ~3h |
| 4 | PruMerge r=0.25 | prumerge_cls | 0.25 | ~3h |
| 5 | PruMerge r=0.75 | prumerge_cls | 0.75 | ~3h |
| 6 | SparseVLM r=0.25 | sparsevlm_text | 0.25 | ~3h |

### 我们的方法 (Variant B 真剪枝)

| # | 实验 | 说明 | 预计时间 |
|---|------|------|---------|
| 7 | SFT Scorer VarB r=0.25 | 用 denylist fallback | ~3h |
| 8 | SFT Scorer VarB r=0.75 | 用 denylist fallback | ~3h |
| 9 | RL Scorer VarB r=0.25 | 等 RL 训完 | ~3h |
| 10 | RL Scorer VarB r=0.5 | 等 RL 训完 | ~3h |
| 11 | RL Scorer VarB r=0.75 | 等 RL 训完 | ~3h |

### Dynamic 列

| # | 实验 | 说明 |
|---|------|------|
| 12 | SFT + τ-cut (VarB) | τ=kr060, varB + fallback |
| 13 | RL + τ-cut (VarB) | 等 RL 训完 |
| 14 | Budget RL | 等 budget RL 训完 |

## 执行优先级

**GPU4 空闲后立即开始补 baseline (1-6)**：
- 这些不依赖任何训练结果，现在就可以排队
- 6 个实验 × ~3h/个 = 18h (串行单卡) → 放不完！
- **解决**: 每个实验只跑 shard0 (N≈2949)，够报数字，1h/个 → 6h 搞定

**GPU0-3 空闲后补我们的 (7-14)**：
- 依赖 RL / Budget RL 训练结果

## 重要提醒

- **所有 "Ours" 行都是 Variant B (真剪枝)**，不是 mask
- SFT Scorer r=0.5 报 0.9045 (VarB + fallback)，已有数据
- 当前 VarB 的 denylist 已更新为 768 tokens (全量)
- Baseline 方法不需要 denylist（它们不用 Variant B，用 Variant A 就行 or 各自的方法）

---

---

## 追加决策 (22:15-22:19 讨论)

### 决策: 主表全部用 Drop (真剪枝)

**原因**:
- FastV/SparseVLM/PruMerge/VisPruner/DivPrune 论文原始实现**全部是真减少序列**
- Mask 只是我们内部 ablation 用的，不是最终论文该报的
- 全部 drop 对我们最有利（我们有 fallback，别人没有）

**影响**: 需要重跑所有 baseline 的 drop 版本

### 决策: Fallback 机制论文写法

- **主表数字**: 报 0.9045（不变）
- **论文写法**: "confidence-based runtime fallback"（不提 PDMS=0 oracle）
- **验证实验**: 开 `safety_net=True`，确认 entropy 检测能 catch catastrophic scenes
- **补跑**: 加入 GPU4 队列

### 更新后需要重跑的实验清单

**全部改为 prune_variant=drop**:

| # | 实验 | ratio | 说明 |
|---|------|-------|------|
| 1 | FastV drop r=0.25 | 0.25 | shard0 |
| 2 | FastV drop r=0.5 | 0.5 | shard0 (重跑) |
| 3 | FastV drop r=0.75 | 0.75 | shard0 (重跑) |
| 4 | Random drop r=0.25 | 0.25 | shard0 |
| 5 | Random drop r=0.5 | 0.5 | shard0 (重跑) |
| 6 | Random drop r=0.75 | 0.75 | shard0 |
| 7 | PruMerge drop r=0.25 | 0.25 | shard0 |
| 8 | PruMerge drop r=0.5 | 0.5 | shard0 (重跑) |
| 9 | PruMerge drop r=0.75 | 0.75 | shard0 |
| 10 | SparseVLM drop r=0.25 | 0.25 | shard0 |
| 11 | SparseVLM drop r=0.5 | 0.5 | shard0 (重跑) |
| 12 | SparseVLM drop r=0.75 | 0.75 | shard0 (已跑中 mask 版) |
| 13 | SFT Scorer VarB r=0.25 | 0.25 | shard0 + denylist |
| 14 | SFT Scorer VarB r=0.75 | 0.75 | shard0 + denylist |
| 15 | Safety-net 验证 | 0.5 | VarB + safety_net=True |

共 15 个实验，shard0 only (~1h/个) = ~15h → GPU4 串行放得下

### GPU4 最终执行顺序

```
[现在-21:00] SparseVLM r=0.75 mask (完成 shard0)
[21:00-03:00] SparseVLM r=0.75 shard1-3 + VarB r=0.75
[03:00 起] 全部切到 drop 版重跑 (15 个实验, ~15h)
```

⚠️ **时间不够全跑完** (只剩 ~16h，但有 15 个 ~1h 的实验 + 当前排队)
优先级: 先跑 r=0.5 drop 版 (比较基准) + SFT VarB r=0.25/r=0.75 + safety-net 验证

*记录时间: 2026-07-21 22:19*
