# ICLR 2027 转投计划

> 写于 2026-07-30，更新于 2026-07-31（补充材料实验完成）。
> 基于 AAAI 2027 投稿的 DriveToken 论文，规划 ICLR 2027 转投的方法创新和实验方案。
>
> **AAAI 补充材料状态**：Fallback Audit (Section H) 和 Matched-Compute (Section I) 数据已产出并填入 supplement.tex。核心发现：Budget RL raw PDMS=0.8707，同协议下 SparseVLM+fallback=0.8906 vs DriveToken+fallback=0.9107 (+0.0201)。Matched-compute Δ=+0.0149 (动态 vs 固定同预算)。

---

## 零、对 AAAI 审阅意见的回应 — ICLR 版本的修复

AAAI 审阅人提出了 9 条意见（Rating 4/10）。以下是对每条意见在 ICLR 版本中的结构性修复：

| # | AAAI 审阅意见 | 8.1 补充材料结果 | ICLR 版本修复 |
|---|---|---|---|
| 1 | fallback 协议不公平 | ✅ 已补: Budget RL raw=0.8707, SparseVLM+fallback=0.8906, DriveToken+fallback=0.9107 (+0.0201 margin) | ICLR 版本主表统一协议：所有方法应用相同 fallback，或全部 raw |
| 2 | 缺少 matched-compute 对照 | ✅ 已补: SFT r=0.355 raw=0.8558 vs Budget RL raw=0.8707, Δ=+0.0149 | 新增 matched-compute 完整矩阵（random budget, entropy heuristic, frozen scorer） |
| 3 | selection surrogate 不是合法 PG | 文字澄清已加入 supplement | **核心方法改动**：用 Per-Token Counterfactual REINFORCE 替换 surrogate（见 §二） |
| 4 | 统计证据不足 | 标注"multi-seed pending" | 5 seeds + paired bootstrap + 均值±std 报告 |
| 5 | 跨模型泛化主张过度 | 降级为 "attention-ranking transfer" | 7B 完整规划指标必须跑通后才恢复 |
| 6 | FLOPs vs 延迟差距大 | 加注 "theoretical upper bound" | 报告 7B 实测 P50/P95/P99 延迟 |
| 7 | 缺少因子消融 | 标注"planned ablation matrix" | 完整 7 项消融矩阵 |
| 8 | 预算自适应证据不足 | 标注"scene-group analysis pending" | 按场景难度分组分析 |
| 9 | Rebuttal 策略 | AAAI rebuttal 已准备好 | ICLR 版本天然解决：方法是新的，实验是完整的 |

**关键洞察**：AAAI 审阅人的 #3（surrogate 问题）是结构性的，无法仅靠加实验修复。ICLR 版本必须从方法层面替换 surrogate。这正是 §二 的 Per-Token Counterfactual REINFORCE 的核心动机。

---

## 一、现状分析

### 1.1 AAAI→ICLR 的核心差距

| 维度 | AAAI 现状 | ICLR 要求 | 差距 |
|---|---|---|---|
| **Novelty** | LambdaRank + GroupNorm PG 已不新 | 需要方法层面有新贡献 | **大** |
| **Experiments** | Baseline 是 shard0-only, nuScenes 没跑通, Budget RL 只训了 1 epoch | 全量、多数据集、多 backbone | **大** |
| **Cross-model transfer** | 只在 NAVSIM 上 3B→7B zero-shot | 需要不同架构、不同数据集的迁移证据 | **中** |
| **Scaling law** | 只有 attention concentration 的数字 | 需要 PDMS vs model size 的 scaling curve | **中** |
| **RL部分** | REINFORCE + Gaussian policy | 需要 token-level credit assignment 等更精细的 RL | **大** |

### 1.2 保留 AAAI 投稿中的强项

以下内容可以直接搬进 ICLR paper，无需重做：
- Stage-1 LambdaRank scorer 架构和训练方法（0.59M，超越 attention teacher）
- 物理 token removal 的工程实现
- Confidence-based fallback 协议
- NAVSIM 上的 PDMS sub-metric breakdown
- Attention concentration 分析（3B vs 7B）

---

## 二、方法创新：Per-Token Credit Assignment for Visual Token Pruning

### 2.1 Motivation

AAAI 版本的 Stage-2 RL 使用 group-normalized REINFORCE 来优化一个 joint (budget, token_set) 决策。核心局限：
1. **粗粒度 reward**：所有 selected tokens 共享同一个 trajectory-level PDMS reward，无法区分"哪些 token 对 PDMS 贡献更大"
2. **Top-K 选择不可微**：用 advantage-weighted softmax surrogate 近似梯度，本质是 heuristic
3. **budget 和 token ranking 耦合**：一个 reward 同时更新 budget head 和 scorer，两者的 credit 分配是隐式的

### 2.2 方法设计

**核心 idea**：将 pruning 决策从 one-shot Top-K 改为 sequential token-level decision process，每个 token 有独立的 counterfactual reward。

#### Stage 2'：Token-Level Counterfactual RL

对于场景 $s$，不直接输出 Top-K 选择，而是：

1. **Sampling**：从 scorer $f_\theta$ 输出中采样 $M$ 个候选 pruning mask（每个 mask 是 binary vector $\mathbf{m} \in \{0,1\}^{720}$），用 budget head $g_\phi$ 控制每个 mask 的稀疏度
2. **Counterfactual evaluation**：对每对 mask $(\mathbf{m}_a, \mathbf{m}_b)$，如果它们只在 token $i$ 上不同（一个 keep 一个 drop），则：
   $$ \Delta_i = \text{PDMS}(\mathbf{m}_a) - \text{PDMS}(\mathbf{m}_b) $$
   这是 token $i$ 的 counterfactual advantage
3. **Credit assignment**：对每个被 kept 的 token，其 advantage 来自所有包含/排除该 token 的 mask pair 的加权平均
4. **Policy update**：per-token REINFORCE with counterfactual baseline:
   $$ \nabla_\theta \mathcal{L} = -\mathbb{E}_{i \sim \text{kept}} \left[ \Delta_i \cdot \nabla_\theta \log p_\theta(\text{keep token } i \mid \text{scene}) \right] $$
5. **Budget update**：budget head 仍然用 group-normalized PG，但现在 reward 是 "per-token advantage 的总和 + efficiency bonus"

#### 与 AAAI 版本的对比

| 维度 | AAAI 版本 | ICLR 版本 |
|---|---|---|
| Decision granularity | Scene-level (one Top-K) | Token-level (sequential) |
| Reward signal | Trajectory PDMS (sparse) | Per-token counterfactual PDMS delta |
| Gradient | Advantage-weighted softmax surrogate | REINFORCE with counterfactual baseline |
| Budget + token coupling | Implicit (shared reward) | Explicit (budget head gets aggregate advantage) |
| Cross-model transfer | Scorer logit level | Per-token advantage level (scalar, invariant to hidden dim) |

### 2.3 为什么 per-token advantage 对跨架构迁移更好

当前 AAAI 版本的 scorer 输出 logit $\in \mathbb{R}$，其分布依赖于 ViT→LLM projection 的 hidden dimension (2048)。当迁移到不同 hidden dim 的 backbone 时，需要重新 projection。

Per-token advantage 本身是 **scalar**（PDMS 差分），不依赖 hidden dim。迁移时：
- Budget head ($k_s$ 标量) → zero-shot transfer
- Per-token scorer → 只需要 **最后 1 层 MLP head** 在新 backbone 上微调（frozen backbone feature 用 projection 对齐到 256 维）

这提供了一个清晰的"泛化性证明"故事：
> "DriveToken's per-token credit assignment produces scalar advantages that are invariant to the backbone's hidden dimension. Only the scorer's input projection layer needs adaptation, while the core ranking and budget policies transfer zero-shot."

### 2.4 实现细节

**Sampling 策略**：
- 每个 scene，sample $M=16$ 个 budget 值（从 Gaussian policy）
- 对每个 budget，做 $T=5$ 次 token swap（随机选 1 个 kept→drop + 1 个 dropped→keep）
- 总共 $M \times (1 + 2T) = 16 \times 11 = 176$ 次 VLA forward per scene per training step

**计算效率**：
- 相比 AAAI 版本（$G \times K = 8 \times 4 = 32$ forward/step），ICLR 版本需要 $176$ forward/step
- 但可以用 **importance sampling** 缩减：只对 "最不确定" 的 token 做 counterfactual（scorer logit 在 median ± 1 std 内的 token）
- 实际缩减后：$\sim 60$ forward/step，约 $2\times$ AAAI 版本的成本
- 总训练时间：$\sim 12$ 天 on 8×H20（vs AAAI 版本的 $\sim 5$ 天）

---

## 三、实验计划

### 3.1 P0 实验（ICLR 必须）

| # | 实验 | 预算 | 说明 |
|---|---|---|---|
| I1 | **Per-token counterfactual RL 训练 + eval** | ~400 GPU-h (8×H20, 12天) | 新的 Stage-2 方法，目标 PDMS > AAAI 0.9107 |
| I2 | **3B 异构 VLA 迁移** | ~150 GPU-h (4×H20, 10天) | 找不同 backbone/架构的 3B VLA（候选：LMDrive/ORION/Senna），训 scorer + budget head，验证 zero-shot transfer |
| I3 | **3B→7B nuScenes transfer** | ~80 GPU-h (4×H20, 3天) | 完整 nuScenes 6,019 sample，修复 image mapping |
| I4 | **Scaling law: PDMS vs model size** | ~150 GPU-h (4×H20, 7天) | 2B/3B/7B 三个规模的 pruning PDMS 曲线 + attention concentration |

### 3.2 P1 实验（锦上添花）

| # | 实验 | 预算 | 说明 |
|---|---|---|---|
| I5 | **Multi-dataset transfer**（NAVSIM → nuScenes → Waymo Open） | ~200 GPU-h | 3 个数据集的 cross-dataset 迁移 |
| I6 | **Closed-loop 验证**（CARLA / nuPlan） | ~100 GPU-h | 证明 pruning 在 closed-loop 下不降低安全性 |
| I7 | **Long-context scaling**（8-camera, multi-frame, >4k tokens） | ~150 GPU-h | 证明 FLOPs 节省随 context 增长而放大 |

### 3.3 时间线

```
8.2-8.7     AAAI P0 实验收尾 (A1 Budget RL 重训)
8.8-8.14    P0-B 全量 baseline 重跑 + P0-C nuScenes 修复 (并行)
8.15-8.26   P0-E 3B异构VLA + P0-F 3B→7B nuScenes + P0-G scaling law (三线并行)
8.27-9.7    ICLR I1 per-token counterfactual RL 开发+训练 (主力)
9.8-9.10    ICLR I1 eval + ablation
9.11-9.23   ICLR 论文写作 (12天)
9.24        AAAI 出分日
9.25        ICLR ddl (TBC)
```

---

## 四、论文结构规划

### 4.1 Title (draft)

> **DriveToken: Token-Level Credit Assignment for Learning Dynamic Vision-Token Budgets in Autonomous VLAs**

### 4.2 核心贡献（重新表述）

1. **Per-token counterfactual credit assignment** for visual token pruning — 首次将 token-level RL 引入 driving VLA 的 vision token compression
2. **Hidden-dimension-invariant transfer** — per-token advantage 是 scalar，跨 backbone 迁移只需 adaptation of the input projection
3. **Scaling law for vision token redundancy** — 定量证明模型越大 → 注意力越集中 → 越能剪
4. **Comprehensive cross-model, cross-dataset evaluation** — 3 backbone architectures × 2 datasets

### 4.3 与 AAAI 版本的主要差异

| Section | AAAI | ICLR |
|---|---|---|
| Method (Stage-2) | GroupNorm PG + Gaussian budget | Per-token counterfactual REINFORCE + Gaussian budget |
| Cross-model | 3B→7B NAVSIM only | 3B异构→3B→7B × NAVSIM + nuScenes |
| Ablation | LambdaRank vs MSE, budget mechanism | + token-level vs scene-level credit assignment |
| Analysis | Attention concentration | + PDMS vs model size scaling curve |
| Appendix | — | Per-token advantage distribution, counterfactual sampling efficiency |

---

## 五、关键风险和缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Per-token counterfactual 计算量太大，12天训不完 | 中 | ICLR 无法提交 | 用 importance sampling 缩减 candidate tokens；降低 M 和 T |
| 找不到合适的 3B 异构 VLA | 中 | 跨架构迁移实验无法做 | 退而求其次：同一 backbone 不同 pretrain dataset 的 3B model |
| nuScenes image mapping 仍然有问题 | 中 | 只能做 NAVSIM | nuScenes 作为 ICLR 的 additional experiments (P1)，非必须 |
| AAAI 中了（概率低但可能） | 低 | 不需要投 ICLR | 仍然做 ICLR 方法创新，作为 journal 扩展 |
| ICLR ddl 不是 9.25 | 高 | 时间线全部后移 | **8.1 前确认**；如果 ddl 是 10月初，P0 实验时间更充裕 |

---

## 六、待决策

- [ ] **ICLR 2027 确切 ddl** — 9.25 还是 10月初？直接影响时间线
- [ ] **方法方向确认** — Per-token credit assignment (方向A) 还是 Process Reward Model (方向B) 还是其他？
- [ ] **3B 异构 VLA 候选** — 需要尽快确定（8.15 前要开始训）
- [ ] **GPU 资源** — 8.2-9.25 期间需要 8×H20 连续可用
- [ ] **人员分工** — 谁负责 P0 实验重跑，谁负责 ICLR 方法创新开发
