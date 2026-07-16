# AAAI 2027 投稿方案（2026-07-09 晚间更新，含 framing 修正）

> **日期**：2026-07-09 20:27 更新
> **项目**：AutoVLA Vision Token Pruning
> **AAAI 2027 DDL**：摘要 2026-07-21 / 全文 2026-07-28（**仅剩 19 天**）
> **决策状态**：**路线 B 已确认（2026-07-10 user 决策）**

---

## 〇、当前方案（两条路，待老师选）

### 路线 A（保守，现有数据即可）
> "Learn **what** to keep, not **how many** to keep."
> Fixed-r + LambdaRank listwise ranker dominate 所有 baseline。自适应比例在 natural setups 下无收益。

### 路线 B（进攻，需 ~2 天新实验）
> "Learn what to keep **= learn how many** to keep."
> Calibrated scorer + 全局阈值 τ 同时解决"选什么"和"留多少"——无需 budget controller 的 unified adaptive pruning。

**B 比 A 强一个档次（有自适应 + method 统一性）。** 但 B 需要一个还没跑的实验（calibrated scorer + τ-cut 全量实测）来确认。

---

## 一、为什么我们要补自适应（framing 修正）

### 今天发现的关键认知修正：Claim② 没有被证伪

之前的 C2 claim（"自适应预算不可学，triple-proof"）**用词过度了**。准确状态是：**Claim② 在干净条件下根本没被测过**，两次负面结果都是机制受污染的。

| 证据 | 测的问题 | 干净？ | 实际结论 |
|---|---|---|---|
| §11 learned budget（6 configs） | ratio 是不是**场景特征**的函数 | ✅ 干净 | 从场景特征学比例不行（只否定这一种机制） |
| ReCogDrive predictor collapse | 换 backbone/benchmark，同上 | ✅ 干净 | 同上，跨设定一致 |
| C-pilot τ-cut | ratio 能不能从 **scorer 分数**涌现 | ❌ **被 listwise 校准污染** | listwise 分数无跨帧绝对意义 → τ-cut 失败是预期的，不能证明自适应不行 |

**方法论判断**：
- "证伪一个主张"需要排除所有合理机制。我们只排除了两种各有问题的机制，**没有排除最自然的那种（calibrated pointwise scorer + 全局 τ）**。
- 所以 Claim② 的真实状态 = **untested under clean conditions + two confounded negatives**，不是 **falsified**。
- 论文中绝对不能写"adaptive is falsified / unlearnable"。当前只能写"not demonstrated under tested setups"。

**核心 insight（"学哪些" = "学多少"）**：
- LambdaRank 是 listwise → 只保证相对排序 → 分数没有跨帧绝对意义 → 全局 τ 不 transfer → 这是**方法选择的副作用**，不是自适应的本质局限
- 如果 scorer 输出有跨帧绝对意义（calibrated/pointwise），那一个全局 τ 就能**同时决定选什么 + 每帧留多少** → "学哪些" = "学多少"，不需要分开学比例
- 我们选了 listwise（为了超 teacher），结构性地排除了自适应能力，然后反过来宣布"自适应不可学"——**这里有部分循环论证**

→ 正确结论：**从场景特征学比例确实不行（§11 干净），但从 calibrated scorer 分数涌现比例是唯一没测的自然机制。在测它之前，Claim② 是悬而未决，不是已死。**

---

## 二、两条路的详细方案

### 路线 A：fixed-r + 好 selector（保守）

**定位**：
> "Under the natural learning setups we test, adaptive ratio provides no benefit over fixed-r. The dominant lever is selector quality."

**三贡献**：
- C1：LambdaRank listwise ranker 超 teacher +0.18pt、超 random +2.84pt（solid，不受影响）
- C2（降级版）："adaptive doesn't help under natural setups"（不 claim unlearnable）
- C3：fixed-r=0.5 是 Pareto-optimal 操作点，33.6% FLOPs saving

**优势**：现有数据即可，不需新实验。安全。
**弱点**：reviewer 可能问"你试过 calibrated scorer 吗？" → 答不上来。

---

### 路线 B：unified adaptive via calibrated scorer（进攻）

**定位**：
> "A single calibrated scorer simultaneously decides WHAT to keep and HOW MANY to keep — no budget controller needed. We show this unified approach dominates both fixed-ratio and learned-budget methods."

**三贡献**：
- C1：Calibrated listwise ranker，scorer 分数同时编码重要性 + 保留数量
- C2（升级版）：严格分离"从特征学比例"（fails）vs "从 calibrated scores 涌现比例"（works/fails → 两种结果都有价值）
- C3：τ-cut 涌现自适应 dominate fixed-r（如果实验 work）/ 或"即使校准也不行"= clean unlearnable proof

**需要的新实验（~2 天）**：
1. 重训 calibrated scorer（pointwise MSE 或 post-hoc calibration，~1 天）
2. τ-cut 全量实测（多个 τ 值真跑 eval，~14 card-h）

**两种结果都不亏**：
- calibrated τ-cut **赢 fixed-r** → 论文直接升级为"unified adaptive"，比所有竞品都强
- calibrated τ-cut **仍不赢** → C2 变干净："连校准过的分数也不能带动自适应"= 真正的 unlearnable proof（不再被 listwise artifact 污染）

---

## 三、核心实验结果（已有）

### 3.1 主表（全量 navtest, N≈11,570）

| 方法 | PDMS | Δ vs no-prune |
|---|---|---|
| No pruning (r=1.0) | 0.8988 | — |
| **Ours scorer r=0.5** | **0.8920** | −0.69 pt |
| Attention selector (r=0.5) | 0.8902 | −0.87 pt |
| Random (r=0.5) | 0.8636 | −3.52 pt |

### 3.2 Pareto（已出 3/4 点，r=0.75 今晚 ~21:30）

| ratio | PDMS | FLOPs saving |
|---|---|---|
| r=1.0 | 0.8988 | 0% |
| r=0.75 | _(running)_ | 16.9% |
| r=0.5 | 0.8920 | 33.6% |
| r=0.25 | 0.8508 | 49.9% |

### 3.3 C-pilot τ-cut（方向性，插值估算）
- 同一 τ 下 per-scene ratio range = [0.20, 0.97]（自适应性确实涌现）
- 但 aggregate PDMS 未 dominate fixed r=0.5（**注：被 listwise 校准问题污染，不是 clean test**）

### 3.4 Cross-backbone validation（ReCogDrive, 0 GPU 已有数据）

| | AutoVLA (ours) | ReCogDrive |
|---|---|---|
| Backbone | Qwen2.5-VL-3B | InternVL3-2B + DiT |
| Benchmark | navtest (PDMS) | navhard (EPDMS) |
| Oracle headroom | +2pt | +7.35pt |
| Adaptive ratio 学习 | 6 configs 全负面 | predictor collapse |
| Selector quality | scorer >> random (+2.84pt) | norm ≈ random (<0.007pt) |

**结论**：
- "从场景特征学 ratio" 在两个独立 backbone × benchmark × architecture 上都失败 → C2 跨设定验证
- Selector quality 只在 AutoVLA 上有效（ReCogDrive DiT mean-pool 消除位置信息）→ C1 不做跨设定 claim
- 论文中用于 C2 的 "Cross-backbone evidence" 章节，0 GPU 成本

---

## 四、竞品对比

| 工作 | Venue | 方法 | 比例策略 | 我们的差异化 |
|---|---|---|---|---|
| FastDriveVLA | AAAI 2026 | MAE重建+前景mask | 固定 | 不需额外数据；排序蒸馏 |
| Prune2Drive | CVPR 2026 | T-FPS覆盖+视角自适应 | learned自适应 | 我们的 unified τ-cut 更简洁（如 B work）/ 或证明自适应前提不成立（如 A） |
| MVPruner | — | 多样性DRA+贡献CCTS | learned自适应 | 同上 |
| ST-Prune | 2026 | 免训练时空 | 免训练/固定 | learned ranker 超 attention |

---

## 五、执行计划

### 如果老师选 A（保守）

| 日期 | 内容 |
|---|---|
| 7/9-7/10 | r=0.75 完成 → FastV baseline |
| 7/11-7/13 | LambdaRank vs MSE 消融 + FLOPs（已完成） |
| 7/14 | 数据锁 |
| 7/15-7/27 | 写作 |

### 如果老师选 B（进攻）

| 日期 | 内容 |
|---|---|
| 7/9晚 | r=0.75 完成 → 启 FastV |
| 7/10 | FastV 跑完 + **开始训 calibrated scorer** |
| 7/11 | calibrated scorer 完成 + **启 τ-cut 全量实测** |
| **7/12** | **τ-cut 出结果 → 定 A 还是 B** |
| 7/13 | LambdaRank 消融 + gate |
| 7/14 | 数据锁 |
| 7/15-7/27 | 写作 |

**B 路线的 fallback**：如果 calibrated τ-cut 不赢 → 自动回退 A，且 C2 反而变干净（clean unlearnable proof）。两种结果都不亏。

---

## 六、决策记录

**2026-07-10 16:31 — User 确认走路线 B**

执行计划（2卡 H20，分窗口推进）：
- 7/11: τ-cut shard0 quick test（3-4 个 τ，~4h）→ 当场定全量/回退
- 7/11-7/12: τ-cut 全量（如果方向对）
- 7/12-7/13: 补充实验（MSE eval / FastV r=0.75）
- 7/14: 数据锁
- 7/15-7/27: 写作

Win condition: τ-cut @ mean_kr≈0.5 PDMS > fixed r=0.5 (0.892)
Fallback: 如果输 → 自动回退路线 A，C2 = clean negative

---

## 七、已完成的产出清单

| 产出 | 路径 |
|---|---|
| 论文 LaTeX 框架（官方模板） | `paper/aaai2027/main.tex` |
| FLOPs 表 + JSON（含 FFN/Attn 分解） | `results/profiling/flops_table.json` |
| Deployment Note 段落（英文 + LaTeX 表） | `docs/results/deployment_note_draft.md` |
| C-pilot τ-cut 两张图 | `docs/results/figures/c_pilot/` |
| FastV baseline 代码 + 脚本 | `code/.../autovla_with_token_prune.py` + `scripts/run_fastv_baseline.sh` |
| scorer r=0.25 全量（0.8508） | `results/raw/tokenprune_S3_full/MT_scorer_r025_sh*.csv` |
| claim③ 确认（scorer > attn +0.18pt） | `key_results.md §12` |
| Cross-backbone 证据整合 | 本文档 §五-B（已有数据，0 GPU） |
| 完整 journal | `docs/journal/2026-07-09.md` |

---

*最后更新：2026-07-09 20:27。scorer r=0.75 在跑（ETA ~21:30），FastV 脚本就绪。*
