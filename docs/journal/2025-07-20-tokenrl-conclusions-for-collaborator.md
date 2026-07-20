# TokenRL-AutoVLA 项目讨论结论 & 下一步行动方案

> 生成日期: 2025-07-20
> 用途: 发给合作者/AI沟通用，确保理解一致
> 论文目标: AAAI 2027 (abstract 7/22, full paper 7/28)

---

## 一、当前实验结果总结（2B模型，闭环PDMS评测）

### 核心数据

| 方法 | Token保留率 | PDMS | Δ vs Baseline | 统计显著性 |
|------|------------|------|---------------|-----------|
| B0 Baseline（无剪枝） | 100% | 0.8988 | — | — |
| SFT Scorer r=0.75 | 75% | 0.8983 | −0.05pt | p=0.581 不显著 |
| SFT Scorer r=0.50 | 50% | 0.8920 | −0.69pt | 显著 |
| SFT Scorer r=0.25 | 25% | 0.8508 | −4.80pt | 崩溃 |
| Random r=0.50 | 50% | 0.8635 | −3.53pt | 崩溃 |
| Attn_L12 r=0.50 | 50% | 0.8901 | −0.87pt | — |
| τ-cut kr060 | ~60% | 0.8940 | −0.48pt | — |
| Variant B (真drop) | ~50% | ~0.8725 | ~−2.63pt | — |

### 关键结论
1. **Learned Scorer 显著优于 naive baselines**：r=0.50 时 scorer (0.8920) >> random (0.8635)，差距 2.85pt
2. **r=0.75 是 2B 模型的甜蜜点**：25% token 换近乎零损失
3. **2B 模型剪枝空间受限**：r=0.50 已开始掉分，r=0.25 崩溃
4. **RL 暂未超过 SFT**：GRPO best=0.889 < SFT=0.892
5. **自适应 budget (Budget Policy) 不如固定 ratio**：负面结果

---

## 二、竞品对标分析

| 工作 | 会议 | 模型大小 | 剪枝比例 | 效果变化 | 加速 |
|------|------|---------|---------|---------|------|
| **FastDriveVLA** | AAAI 2026 | 大模型(≥7B) | 75% | L2 error 不升反降 | **7x** |
| **LightVLA** | NeurIPS 2025 | OpenVLA 7B | ~59% FLOPs | **+2.6% 成功率** | 38% latency↓ |
| **FastV** | ECCV 2024 Oral | 7B/13B | 50% | 近无损 | 45% FLOPs↓ |
| **我们 (当前)** | — | **2B** | 25% | −0.05pt (无损) | 待测 |

### 核心 Insight
> **模型越大，冗余越多，剪枝空间越大。**
> - 7B 模型普遍可以做到 50-75% 无损甚至提升
> - 2B 模型信息密度高，每个 token 承载更多信息，25% 已是极限
> - 竞品全部在 7B+ 上做的，我们必须补 7B 实验才能对齐

---

## 三、论文叙事方向（已确认）

**定位: Empirical Study — "When Does Token Pruning Break in AD-VLAs?"**

### 贡献点
1. **首个闭环自动驾驶 (PDMS) 上系统评估 vision token pruning** — 竞品都是开环或机器人
2. **Learned Scorer 实现 25% 无损剪枝，显著优于 training-free baselines (random/attention)**
3. **2B→7B Scaling Law**: 验证模型规模 vs 剪枝容忍度的关系（新实验）
4. **完整 Pareto 曲线 + Failure Analysis**: r=0.25 崩溃 4.8pt 的 insight
5. **τ-cut 自适应阈值方案**: 无需手工调 ratio
6. **RL/Budget Policy 负面结果作为有价值发现**

### RL 的定位
- 放在 Discussion/Future Work
- 不作为核心贡献（目前没超 SFT）
- 如果 7B 上 RL 能超 SFT 则可以升级为贡献

---

## 四、下一步行动方案：上 AutoVLA 7B

### 为什么选 AutoVLA 7B
- **架构**: Qwen2.5-VL backbone（和我们 2B 实验同系列，迁移最顺）
- **会议**: NeurIPS 2025 已中，热度高
- **代码开源**: github.com/ucla-mobility/AutoVLA
- **评测**: nuScenes / CARLA 闭环，和我们一致

### 核心问题：SFT 还是 RL？

**推荐答案: 用 SFT，不用 RL**

理由：
1. **SFT 已验证有效**：2B 上 scorer SFT 已经 work (r=0.75 无损)
2. **RL 在 2B 上没超 SFT**：没理由认为 7B 上会突然超
3. **时间紧迫**：7/28 交稿，只有 8 天，SFT 可以 1-2 天出结果，RL 需要更多调参
4. **竞品 LightVLA 也是 SFT 路线**（可微分剪枝 = 本质也是监督学习）
5. **叙事不依赖 RL**：新叙事是 empirical study，SFT 够用

### 是否需要重新训数据？

**不需要从零训，但需要适配：**

| 步骤 | 说明 | 预计耗时 |
|------|------|---------|
| 1. 部署 AutoVLA 7B | 从 GitHub 拉代码 + HF 下载权重 | 2-3h |
| 2. 适配 Scorer 训练 | 用 AutoVLA 7B 前向推理生成 attention map，标注 token importance | 半天 |
| 3. 训练 SFT Scorer (for 7B) | 复用 2B 的 scorer 架构，换 7B 的 hidden dim | 1天 |
| 4. 跑评测 | r=0.75, r=0.50 两个点，PDMS 闭环 | 1-1.5天 |
| **总计** | | **~3天** |

**关键：scorer 数据生成方式和 2B 完全一样**
- 输入: 多帧图像 → vision encoder → 得到 vision tokens
- 标注: 用原始 7B 模型跑一遍完整推理，记录每个 token 对 action loss 的贡献（importance score）
- 训练: 二分类/回归，预测 token importance
- 推理: scorer 选 top-r tokens → 只用这些 token 做 action prediction

**唯一区别: hidden dim 从 2B 的 1536 → 7B 的 3584**，scorer MLP 改一下维度就行。

---

## 五、时间线（到 7/28）

| 日期 | 任务 | Owner |
|------|------|-------|
| 7/20-21 (今天-明天) | 聚合 2B FastV + Variant B 数据，跑完最后 2B 实验 | 我 |
| 7/21-22 | 部署 AutoVLA 7B，开始 scorer 数据生成 | 我/合作者 |
| 7/22 | 提交 AAAI abstract（基于 2B 数据 + 7B 计划） | 我 |
| 7/22-24 | 7B SFT Scorer 训练 + 评测 (r=0.75, r=0.50) | GPU 上跑 |
| 7/24-25 | 出 7B 结果，验证 scaling hypothesis | — |
| 7/25-28 | 写论文 + 画图 + 完善实验 | 我 |
| 7/28 | 提交 full paper | — |

---

## 六、给合作者的一句话摘要

> **结论：2B 模型上我们的 SFT Scorer 已验证 25% 无损剪枝，但 2B 天花板低（竞品 7B 上能做 50-75% 无损）。下一步直接迁移到 AutoVLA 7B (Qwen2.5-VL backbone)，用 SFT 路线（不用 RL），预计 3 天出结果。不需要重新造数据，只需用 7B 模型重新跑一遍 token importance 标注 + 换 scorer 维度。**

---

## 七、硬件需求确认

- **当前可用**: 8× H20 (96GB each, 768GB total)
- **7B Full SFT Scorer 训练**: ~4 卡 × 1 天
- **7B 推理评测**: ~2 卡 × 1.5 天
- **结论**: 硬件够用，不需要额外申请

---

## 八、风险与 Plan B

| 风险 | 概率 | 对策 |
|------|------|------|
| AutoVLA 7B 部署问题 | 中 | 备选: 直接用 Qwen2.5-VL-7B-Instruct + 自己的 AD head |
| 7B scorer 效果不如预期 | 低 | 即使 r=0.50 掉 1pt，相比竞品开环结果仍有 insight |
| 时间不够 | 中 | 最小可行实验: 只跑 r=0.75 一个点 + wall-clock speedup |
| RL 被 reviewer 追问 | 高 | 论文中明确说 "RL as future work, SFT scorer is our proposed method" |
