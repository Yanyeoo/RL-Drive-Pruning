# 论文故事决策记录 (2026-07-21 讨论确定)

---

## 1. 核心故事方向

**三个关键词**: Driving-aware × Dynamic × Plug-and-play

**一句话故事**: 
> "Vision token 的重要性应该由 driving context 动态决定。我们提出一个轻量即插即用的动态剪枝模块，通过 RL 直接从 driving reward 学习场景自适应的 token selection 策略，并且能 zero-shot 泛化到不同模型和数据集。"

---

## 2. 标题候选（待确定，避免和同门 "Divide and Prune" 撞）

1. **TokenRL: Reinforcement Learning for Driving-Aware Vision Token Pruning in Autonomous VLAs**
2. RL-Drive: Reinforcement Learning Driven Adaptive Token Pruning for Efficient Vision-Language-Action Models in Autonomous Driving
3. DriveTokens: Learning Scene-Adaptive Vision Token Selection for Efficient Driving VLAs
4. AdaPrune: Driving-Aware Adaptive Vision Token Pruning for Efficient VLA Inference
5. DriveScore: Scene-Adaptive Vision Token Pruning with Driving Reward for Autonomous VLAs
6. ScorePrune: Driving-Reward-Learned Token Selection for Efficient Autonomous VLAs

---

## 3. Contributions

### C1: Driving-reward-guided dynamic pruning framework

A driving-reward-guided dynamic pruning framework. The scorer is trained via RL with a **dual-objective shaped reward**:
- Option 1 (绝对项, α=0.3): `Σ(w_i * sub_i_pruned)` — scene-difficulty-aware
- Option 3 (相对项, β=0.7): `Σ(w_i * (sub_i_pruned - sub_i_baseline))` — measures pruning-induced delta

6 个 driving sub-metrics (collision, drivable, progress, TTC, direction, comfort) 各自独立给梯度信号，解决 naive composite metric 的 reward 稀疏性问题。

### C2: Plug-and-play 0.6M scorer with τ-cut

A plug-and-play 0.6M-parameter MLP scorer that, combined with a single global threshold (τ-cut), simultaneously decides:
- **Which** tokens to retain (by score ranking)
- **How many** to retain (by the number exceeding τ per scene, naturally varies)

无需 explicit budget controller。Scorer 是 model-agnostic：接在任何 VLA 的 ViT→LLM 接口，<1ms overhead。

### C3: Cross-model generalization + scaling law

Cross-model and cross-dataset generalization:
- 3B 上训的 scorer **zero-shot** 迁移到 7B，ranking quality 保持 (pairwise acc 0.856 vs 3B: 0.839)
- 迁移到 Impromptu-VLA + nuScenes 也有效 (TODO: 等实验)

Scaling law insight: 7B 模型 95.9% attention 集中在 top-25% tokens (vs 3B: 91.6%) → 大模型冗余更大，更耐剪。

**C3 同时证明了方法的泛化性和 scaling insight。**

---

## 4. SFT 故事 (事实核实通过 ✅)

### 论文讲法（三步递进）

1. **SFT 是 warm-start**: 用 L12 attention (last instruction → vision tokens, head-avg) 做标签，快速学到 rough importance signal
2. **SFT 的 ceiling = attention teacher 本身**: attention 高 ≠ driving-important（天空/亮云 attention 高但无用；远处小车 attention 低但关键）。但 MLP 通过 camera position 等特征学到了超越 attention 的 pattern（scorer 0.8920 > teacher 0.8901, +0.19pt）
3. **RL 通过 driving reward 突破 ceiling**: shaped sub-metric reward 让 scorer 学到 attention 看不到的 driving-critical pattern

### 事实基础
- 输入: [layer-0 ViT embedding (2048) + cam_onehot (3)] = 2051 维
- 标签: L12 attention (last instruction → 720 vision tokens, 16-head averaged)
- Loss: LambdaRank (listwise ranking)
- 训练: 4000 navtrain scenes, <30s on 1 GPU
- 结果: pairwise acc 0.8388, NDCG@360 0.8745
- **Scorer > Teacher**: 0.8920 > 0.8901 (+0.19pt at r=0.5, full navtest N=11572)

### SFT → RL 的因果逻辑
- 不是"为了 RL 而 RL"
- 而是: "SFT 有已知局限 (attention≠driving importance) → RL 用 driving reward 来校正这个偏差"

### MSE vs LambdaRank (两种 SFT，服务不同目的)
- **LambdaRank**: 排序准 (acc 0.839) → 用于 fixed top-K selection
- **MSE**: 分数 calibrated (跨帧有绝对含义) → 用于 τ-cut adaptive
- 论文中: 主方法用 LambdaRank; τ-cut 变体用 MSE (作为设计选择 ablation)

---

## 5. RL 设计 (Option 1+3 合并)

### Reward 公式
```
reward = α * Σ(w_i * sub_i_pruned) + β * Σ(w_i * (sub_i_pruned - sub_i_baseline))
```
- α=0.3 (绝对质量项: scene 难度感知, 训练稳定性)
- β=0.7 (相对 delta 项: 剪了之后变好还是变差)
- weights: progress=0.35, collision=0.20, drivable=0.15, ttc=0.15, direction=0.10, comfort=0.05
- baseline = 同 scene r=1.0 的子指标分数

### 参数来源 (motivation-driven, 非 grid search)
- 子指标权重: 按改进空间排 (progress=0.83 空间最大→0.35; collision 重要但已 0.99→0.20; comfort 几乎不变→0.05)
- α/β: 相对 delta 是主信号 → β>α; 绝对项辅助稳定

### vs 旧 reward (naive PDMS 乘积) 的改进
旧 reward 失败原因:
1. PDMS = 6 子指标乘积 → 任何一个=0 则 reward=0 (太稀疏)
2. 5 个子项 >0.96, 同 scene 不同 selection 差异 <0.01 (advantage≈0)

新 reward 改进:
1. 子指标分开 → 每个维度独立给信号
2. Delta 放大 pruning 边际效应 (信号量级大 10-100×)
3. 绝对项保留 scene 难度感知 (不会振荡)

### 实验状态
- 代码已改好 (`score.py` + `train_scorer_grpo.py`)
- baseline sub-scores JSON 已提取
- **等 GPU 跑** (预计 4-5h train + 3h eval)

---

## 6. 论文结构 (大框架)

1. **Introduction**: gap = 没人对 AD-VLA 用 RL 做 token pruning; 现有方法 scene-agnostic
2. **Related Work**: VLM pruning / AD pruning / RL for token optimization
3. **Method**:
   - 3.1 Overview (动态三层: 选什么 + 选多少 + 跨模型泛化)
   - 3.2 Scorer Architecture (0.6M MLP, <1ms)
   - 3.3 SFT Distillation (LambdaRank from L12 attention)
   - 3.4 RL Fine-tuning (shaped dual-objective driving reward)
   - 3.5 τ-cut Adaptive Pruning (MSE variant, what=how many)
   - 3.6 Variant B: True Token Drop
4. **Experiments**:
   - 4.1 Setup
   - 4.2 Main Table (3B, r=0.5, all baselines) — RL 行等填
   - 4.3 Pareto Analysis (r=0.25/0.5/0.75/1.0 + τ-cut)
   - 4.4 RL Ablation (shaped vs naive reward)
   - 4.5 Cross-Model Generalization (7B + nuScenes) — 等实验
   - 4.6 Efficiency (FLOPs, wall-clock, Variant B)
   - 4.7 Ablations (LambdaRank vs MSE, Layer selection, etc.)
5. **Discussion**: scaling insight + reward design insight + limitations
6. **Conclusion**

---

## 7. 待跑实验 (等 GPU)

| # | 实验 | 预计时间 | 论文位置 |
|---|------|---------|---------|
| 1 | Baseline sub-scores → JSON | 10min | RL 前置依赖 |
| 2 | RL shaped reward 重训 (3B, 4卡) | 4-5h | C1 核心数据 |
| 3 | RL scorer eval on navtest | 3h | 主表 RL 行 |
| 4 | Impromptu-VLA 7B nuScenes eval | 4-6h | C3 泛化性 |
| 5 | 动态证据离线分析 (CPU, 2-3min) | 3min | §4.3 或 figure |

---

## 8. "动态"证据 (离线分析)

脚本已写: `scripts/analyze_taucut_dynamic.py`

做什么:
- 加载 MSE scorer + 4000 scenes 的 features
- 对每个 scene 打分 → 数 score > τ 的 token 数 → per-scene keep ratio
- 按场景难度 (baseline PDMS) 分组统计
- 输出 histogram + scatter plot + correlation

期望结果:
- 难场景保留 ~70%, 简单场景保留 ~40%
- keep ratio 和 baseline PDMS 有显著相关

状态: 脚本写好，等 conda 环境可用时跑 (纯 CPU, 2-3 分钟)

---

*记录时间: 2026-07-21 16:57*
