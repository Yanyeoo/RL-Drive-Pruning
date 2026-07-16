# Related Work Survey — Vision Token Pruning for VLA/VLM

> 写于 2026-07-14。调研 20+ 篇最新相关工作，按类别整理。

---

## 1. AD-VLA 专用 Token Pruning（直接竞品）

### 1.1 FastDriveVLA [AAAI 2026, 小鹏+北大]
- **方法**: MAE 风格前景像素重建 → ReconPruner（0.07B 参数）
- **核心思路**: 对抗前景-背景重建策略，保留驾驶相关 foreground token
- **结果**: nuScenes 开环 7.5× FLOPs reduction
- **与我们的差异**: pixel-level 重建 vs task-level ranking；他们需要额外训练数据(nuScenes-FG)；我们用 LLM 内部 attention 蒸馏

### 1.2 Prune2Drive [CVPR 2026, 上交+上海AI Lab+CMU+港大]
- **方法**: (i) T-FPS (Token-level Farthest Point Sampling) 保多样性; (ii) 视图自适应剪枝率优化
- **核心思路**: 多视角 VLM，每个视角分配不同 ratio；DRA (Diversity-aware Retrieval Attention) + CCTS (Cross-Camera Token Selection)
- **结果**: DriveLM/DriveLMM-o1 上显著加速
- **与我们的差异**: 多视角自适应（per-view ratio）vs 我们的 per-scene adaptive；他们是 training-free diversity heuristic

### 1.3 MVPruner [arXiv 2026.06]
- **方法**: Dynamic token pruning for multi-view VLM in AD
- **核心思路**: per-view diversity score → adaptive budget
- **与我们的差异**: 类似 Prune2Drive 的多视角自适应

### 1.4 ST-Prune [arXiv 2026.04, 中科院]
- **方法**: 免训练时空 Token 剪枝
- **核心思路**: spatial + temporal redundancy，减少高达 90% token，2.5× 推理提速
- **与我们的差异**: training-free vs learned selector；他们利用时间冗余（视频帧间）

---

## 2. 通用 VLM Token Pruning（方法论参考）

### 2.1 FastV [ECCV 2024 Oral, 北大]
- **方法**: 早层 attention 分数 → 剪枝（layer-2 之后）
- **结果**: 45% FLOPs reduction
- **局限**: 在 AD 场景表现差（我们验证：比 random 还差！原因是浅层 attention 对驾驶任务无信号）

### 2.2 SparseVLM [ICML 2025]
- **方法**: Text-guided training-free token sparsification
- **核心思路**: 用 text token 的 attention 评估 vision token 重要性 + 自适应 ratio + token 回收
- **与我们的差异**: training-free vs learned；他们用 text-vision cross-attention

### 2.3 LLaVA-PruMerge [ICCV 2025]
- **方法**: Prune + Merge（先剪再合）
- **核心思路**: CLS-attention 异常检测筛选重要 token → Key 向量相似度合并剩余 token
- **结果**: 18× token 压缩
- **与我们的差异**: prune+merge vs pure prune；他们无 task-aware signal

### 2.4 Variation-aware Vision Token Dropping [CVPR 2026]
- **方法**: 基于 token 变化量（variation）的 dropping 策略
- **核心思路**: 高分辨率图像中相邻 token 冗余大 → variation 小的可以 drop

### 2.5 VisionZip [arXiv 2024]
- **方法**: 基于 attention sink 的 token 选择 + merging
- **核心思路**: LLM 前几层会固定 attend 少量 vision token (attention sink) → 这些是关键

### 2.6 ToMe (Token Merging) [ICLR 2023]
- **方法**: Bipartite soft matching → token merging in ViT
- **经典基线**: 通用 ViT 加速，不含 task-specific signal

---

## 3. RL-based Token Pruning（方法论最相关！）

### 3.1 TOP-RL [AAAI 2026 (v40)]  ⚠️ 直接竞品
- **方法**: Task-Optimized Progressive token pruning via RL
- **核心思路**: 把 token pruning 建模为 sequential decision → RL policy 逐层决定 prune/keep
- **RL 细节**: layer-wise progressive pruning, task reward (VQA accuracy etc.)
- **与我们的差异**: 他们是 general VLM (VQA/Caption) 上的 RL；我们是 AD-specific + driving reward
- **重要性**: 证明 RL for token pruning 这个方向有人在做且被 AAAI 接收！

### 3.2 RL4EViT [ICME 2025]
- **方法**: Multi-agent RL for ViT token pruning
- **核心思路**: 每个 token 是一个 agent，决定自己 prune or keep（Multi-Agent Markov Game）
- **与我们的差异**: ViT-only (classification) vs VLA (driving)；他们用 multi-agent RL 我们考虑 GRPO

### 3.3 VPPO-RL [ICLR 2026]
- **方法**: Token-level visual perception RL for multimodal reasoning
- **核心思路**: 不是 pruning 而是用 RL 给 token-level reward 做视觉感知优化
- **与我们的关联**: RL 在 token level 的应用先例

---

## 4. VLA Efficiency（更广的 context）

### 4.1 AutoVLA [NeurIPS 2025]
- 我们的 backbone，Qwen2.5-VL-3B + GRPO for driving

### 4.2 π0.5 [Google, 2025]
- 动态动作 token 化（FAST），简单任务短 token，复杂任务长 token
- **启发**: "adaptive token length" 的 idea 在 action space 已被验证

---

## 5. 关键发现总结

### 5.1 我们的定位 vs 竞品

| 维度 | FastDriveVLA | Prune2Drive | ST-Prune | TOP-RL | **Ours** |
|---|---|---|---|---|---|
| Venue | AAAI'26 | CVPR'26 | arXiv'26 | AAAI'26 | AAAI'27 target |
| Selector | MAE重建 | T-FPS diversity | 时空heuristic | RL policy | **Learned ranker** |
| Ratio | fixed | per-view adaptive | fixed | layer-wise adaptive | **per-scene adaptive (τ-cut)** |
| Training | 额外数据 | training-free | training-free | RL (task reward) | **蒸馏 (+ RL planned)** |
| AD-specific | ✅ | ✅ | ✅ | ❌ (general VLM) | ✅ |
| RL-driven | ❌ | ❌ | ❌ | ✅ | **✅ (planned)** |
| True speedup | ✅ | ✅ | ✅ | ✅ | **✅ (Variant B done)** |

### 5.2 创新空间分析

**我们能 claim 的独特性：**
1. **第一个 RL-driven AD-VLA token pruning** — TOP-RL 做了 RL 但在 general VLM 上；FastDriveVLA/Prune2Drive 做了 AD 但没 RL。交叉点是空的。
2. **Driving reward 直接优化 token selection** — 不是 VQA/Caption accuracy，是 trajectory quality (PDMS)
3. **蒸馏 + RL 两阶段** — 先 SFT 打底（cheap），再 GRPO 精调（expensive but principled）
4. **Unified adaptive**: τ-cut 证明了"学什么 = 学多少"的 insight（其他都是分开处理）

**但当前缺失：**
- RL 没做（TOP-RL 已经做了 RL for token pruning 并被接收）
- 没有真实 speedup 数据（竞品都有）
- 效果没有 substantial 提升

### 5.3 对论文方向的建议

**如果要投 AAAI 2027，story 应该是：**
> "First RL-optimized adaptive token pruning for autonomous driving VLAs, 
>  achieving X% real-time speedup while maintaining driving safety."

**必须有的**：
1. RL (GRPO) 训练过的 scorer → 区别于所有 heuristic/distillation-only 方法
2. Real speedup 数据 (Variant B) → 证明实际加速
3. Driving-specific reward → 区别于 TOP-RL 的 VQA reward

---

## 6. 对比实验应该补什么

| 对比方法 | 能否复现 | 优先级 |
|---|---|---|
| FastV (在我们设定下) | ✅ 已做 | done |
| Random baseline | ✅ 已做 | done |
| Attention selector (teacher) | ✅ 已做 | done |
| Fixed-ratio (no selector) | ✅ 已做 | done |
| SparseVLM (text-guided) | 可做（training-free） | 高 |
| LLaVA-PruMerge (CLS-attention) | 可做（training-free） | 中 |
| ToMe (token merging) | 可做（training-free） | 中 |
| Prune2Drive (T-FPS) | 难（多视角设定不同） | 低 |
| FastDriveVLA (MAE重建) | 难（需额外训练） | 低（引用paper数字） |

### 泛化实验建议

| 实验 | 作用 |
|---|---|
| 更大模型 (7B/13B) | 证明 token pruning 在大模型上收益更大 |
| 不同 backbone (InternVL) | 跨架构泛化（已有 ReCogDrive 数据） |
| 不同 benchmark (navhard) | 难场景验证 |
| 不同 pruning ratio curve | Pareto 完整度 |

---

*调研完成。等 user 讨论后确定最终方向。*
