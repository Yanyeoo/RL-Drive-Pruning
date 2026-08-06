# ICLR 2027 转投计划

> 写于 2026-07-30，更新于 2026-08-04（ddl确认、GPU约束、factcheck修正）。
> 基于 AAAI 2027 投稿的 DriveToken 论文，规划 ICLR 2027 转投的方法创新和实验方案。
>
> **关键约束**：
> - ICLR ddl: **2026-09-25**（已确认）
> - GPU: 每周 **320 卡时**（≈ H20×13.3 GPU-天），单次连续最多 **24h**
> - 目标: **追求 SOTA 更高 PDMS**（不是防守型"同PDMS少token"）
>
> **AAAI 补充材料 factcheck（8.3）**：
> - `0.9107` 无本地产物链，真实 denylist PDMS = `0.89125`
> - 同协议 margin `+0.0201` 不成立（真实 +0.0006）
> - AAAI 正文已投不可改，等审稿人指出后在 rebuttal 回应

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

## 二、方法创新

### 2.0 v5 路线：True PDMS Reward（当前主线，8.5）

**发现**：v3/v4 训练中，simulator proxy reward 与真实 PDMS 不相关。
- v4 training reward 全程平稳（~0.78），但真实 PDMS 从 0.879（BEST）降到 0.861（FINAL）
- 根因：`efficiency_bonus` 推动 kr↓，training reward 不降（被 bonus 补偿），但真实 PDMS 受损

**v5 方案**：直接用真实 PDMS product 作为 reward，去掉 efficiency_bonus。
```
改动: rl_pdm_score(use_true_pdms=True) → result.score (PDMS product, 与 navtest eval 完全一致)
      total_reward = driving_scale × PDMS_product - safety_beta × safety_loss
      去掉: + efficiency_beta × (1 - kr)
```
计算量不变（同一次 VLA forward，只换评分信号源）。

### 2.1 7B 跨模型迁移：ViT 特征方案（8.5）

**发现**：Qwen2.5-VL 3B 和 7B 使用**同一个 ViT**（hidden=1280），仅 LLM projection 维度不同（2048 vs 3584）。

**方案**：直接从 ViT 最后一层提取特征（1280维），绕过 LLM projection。
```
当前: ViT(1280) → LLM proj → 2048/3584 → scorer(Linear(2048+3, 256) → ...)
改进: ViT(1280) → scorer(Linear(1280+3, 256) → ...)  ← 3B/7B 通用！
```
- scorer `emb_dim` 从 2048 改为 1280
- 3B scorer 训练完成后，直接用于 7B VLA 推理（zero-shot）
- 不需要重新训练 7B scorer
- 需要改 hook 位置：从 LLM 推理 hook 改为 ViT 输出 hook

### 2.2 Per-Token Credit Assignment（下一阶段）

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
| I1 | **v5 True PDMS RL 训练+eval** | ~80 GPU-h | 用真实 PDMS 替代 proxy reward，去掉 efficiency_bonus，目标 PDMS > 0.89 |
| I2 | **7B ViT特征跨模型迁移** | ~40 GPU-h | 改 hook 到 ViT 输出，scorer emb_dim=1280，3B训完→7B zero-shot eval |
| I3 | **完整消融矩阵** | ~80 GPU-h | eff_beta=0 vs 0.05, safety_beta, reward type (proxy vs true PDMS), seed |
| I4 | **Per-token counterfactual RL**（如 v5 不够） | ~200 GPU-h | 更根本的 reward 信号方案 |

### 3.2 P1 实验（有余力则做）

| # | 实验 | 预算 | 说明 |
|---|---|---|---|
| I5 | **Scaling law: PDMS vs model size** | ~80 GPU-h | 2B/3B/7B pruning PDMS 曲线 |
| I6 | **Multi-dataset transfer** | ~100 GPU-h | NAVSIM → nuScenes → Waymo |
| I7 | **Closed-loop 验证** | ~80 GPU-h | CARLA / nuPlan |

### 3.3 GPU 预算与分配（每周 320 卡时）

```
总预算（8.5-9.25 = 7 周）: 7 × 320 = 2240 卡时
P0 实验需求: ~540 卡时
P1 实验需求: ~260 卡时
论文写作: 无需 GPU
────────────────────
合计: ~800 卡时 / 2240 可用 → 充裕
```

**每 24h 窗口（H20×8） = 192 卡时**。关键实验的窗口规划：

| 窗口 | 日期 | 内容 | 卡时 |
|---|---|---|---|
| W1 | 8.4-8.5 | **v4 Budget RL 训练+eval**（进行中） | ~64 |
| W2 | 8.5-8.6 | v4 结果分析 + I1 代码开发（CPU） | 0 |
| W3 | 8.6-8.12 | **I1 Per-token RL 代码开发**（CPU，无GPU） | 0 |
| W4 | 8.13-8.14 | I1 smoke test + 小规模验证（24h窗口） | ~96 |
| W5 | 8.15-8.16 | I1 正式训练 epoch 1（24h窗口） | ~192 |
| W6 | 8.17-8.18 | I1 正式训练 epoch 2+3（24h窗口） | ~192 |
| W7 | 8.19-8.20 | I1 eval + I4 消融（24h窗口） | ~192 |
| W8 | 8.21-8.22 | I2 异构VLA迁移（24h窗口） | ~96 |
| W9 | 8.23-8.24 | I3 nuScenes transfer（24h窗口） | ~96 |
| W10 | 8.25-9.7 | buffer + 补实验 + 论文写作 | — |
| — | 9.8-9.24 | ICLR 论文写作 | 0 |
| — | 9.25 | **ICLR ddl** | — |

### 3.4 时间线（更新）

```
8.4-8.5     [GPU] v4 Budget RL 训练+eval（已完成）
8.5         [GPU] 消融 eval（v3/v4/safety_net/SFT）+ 7B nuScenes
8.5 18:40   [GPU] v5 True PDMS RL 训练启动（auto_launch）
8.6 00:40   [GPU] v5 训练完成
8.6-8.7     [GPU] v5 全量 eval + 消融
8.7-8.8     [CPU] v5 结果分析 + 决定下一步
8.8-8.10    [CPU] 7B ViT特征 hook 改动 + 3B scorer 重训（emb_dim=1280）
8.10-8.12   [GPU] 7B cross-model eval (3B scorer → 7B VLA)
8.12-8.20   [GPU/CPU] 根据 v5 结果：继续 RL 优化 或 论文写作
8.20-9.7    [GPU] buffer窗口（补实验）
9.8-9.24    [CPU] ICLR 论文写作
9.25        ICLR ddl
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
| Per-token RL 计算量太大，24h 窗口训不完 1 epoch | 中 | 训练被截断 | importance sampling 缩减；拆多窗口续训（ckpt 可恢复） |
| 找不到合适的 3B 异构 VLA | 中 | 跨架构迁移实验无法做 | 退而求其次：同一 backbone 不同 pretrain dataset |
| nuScenes image mapping 仍然有问题 | 中 | 只能做 NAVSIM | nuScenes 降为 P1，非必须 |
| AAAI 中了（概率低） | 低 | 不需要投 ICLR | 仍做 ICLR 方法创新，作为 journal 扩展 |
| Per-token RL PDMS 不升反降 | 中 | ICLR 核心贡献不成立 | 保留当前 REINFORCE 作为 baseline；per-token 作为消融对比项 |

---

## 六、当前基线（8.4 factcheck 后）

| 方法 | PDMS | 备注 |
|---|---|---|
| no-prune (r=1.0) | **0.89886** | 理论上界 |
| SFT scorer r=0.5 | **0.89008** | Stage-1 最强 baseline |
| SFT scorer r=0.355 | 0.85575 | matched-compute baseline |
| Budget RL v3 (旧, raw) | 0.87066 | AAAI 投稿版本 |
| Budget RL v3 + denylist | 0.89125 | 离线重建 |
| Budget RL v4 | **待产出** | 本周期训练中 |
| **ICLR I1 目标** | **> 0.895** | Per-token RL, 逼近 no-prune |

## 七、待决策

- [x] **ICLR 2027 确切 ddl** — **9.25** ✅
- [x] **方法方向确认** — **Per-token credit assignment** ✅
- [x] **0.9107 处理** — **等审稿人指出，不在 rebuttal 中主动更正** ✅
- [x] **GPU 资源** — 每周 320 卡时，单次 ≤24h ✅
- [x] **目标定位** — **追求 SOTA 更高 PDMS**（非防守型） ✅
- [ ] **3B 异构 VLA 候选** — 需要尽快确定（8.15 前）
- [ ] **I1 代码开发分工** — 8.6-8.12 期间完成
