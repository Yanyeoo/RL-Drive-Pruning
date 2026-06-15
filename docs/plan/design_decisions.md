# RL-Drive on AutoVLA — Design Decisions

> 本文档记录论文方向的**核心设计决策**。每条决策包含：选项、最终选择、理由、已 reject 的方案。
> 决策为渐进 commit，每谈定一个 Q 写一条，未谈定的 Q 留空。
>
> 维护原则：决策一旦 commit，不轻易回滚；如要回滚必须新增一条 "Revision" 并写明触发原因。

---

## 元信息

- 项目：RL-Drive (AutoVLA backbone + vision token pruning + GRPO)
- 关联旧工作：a prior internal scorer-based driving pipeline (referred to as **prior work** below)，已停止迭代
- 开始日期：2026-06-14
- Benchmark：**NAVSIM navtest（open-loop EPDMS）**  ← Revision 2026-06-15（原 navhard_two_stage，详见文末 Revision 记录）

## 决策进度总览

| 章节 | 主题 | 状态 |
|---|---|---|
| Q1 | 剪枝目的与优先级 | ✅ 2026-06-14 |
| Q2 | 剪谁的 token (vision-only @ ViT-LLM 接口) | ✅ 2026-06-14 |
| Q3 | 颗粒度 / 两阶段 (Importance + Budget) | ✅ 2026-06-14 |
| Q4.1 | Importance label (attention distill) | ✅ 2026-06-14 |
| Q4.2 | Budget label (B3 Pareto-aware oracle) | ✅ 2026-06-14 |
| Q4.3 | GRPO reward (R3 piecewise Pareto) | ✅ 2026-06-14 |
| Q4.4 | 数据切分 (probe stratified + B=C full) | ✅ 2026-06-14 |
| Q5 | 评测协议 (7 baselines + A1–A10 ablations) | ✅ 2026-06-14 |

**第一轮 design freeze 完成**：Q1–Q5 全部 commit，可进入 plan / experiment 阶段。

---

## Q1. 剪枝的真正目的与优先级 ✅ 2026-06-14

### 决策
- **优先级**：**A 主轴 > B 底线 > C framing**
  - **A. 省算力**：vision token 平均节省 **≥ 50%**（avg kept ratio ≤ 0.5）
  - **B. 不掉点**：iso-compute (avg ratio = 0.5) 下 EPDMS 持平 baseline AutoVLA r=1.0，容忍 ε ≤ 0.5 EPDMS 点
  - **C. Novelty**：scene-adaptive importance + RL-learned budget policy（two-stage rework），作为 method section 卖点

### Paper 形态
- **Main result**：Pareto curve (EPDMS vs avg tokens kept)，须严格 dominate FastV / ToMe / random / fixed-ratio
- **Success criterion**：在 Pareto 曲线 avg ratio = 0.5 这一列上击败所有 baseline，且 EPDMS ≥ r=1.0 baseline − 0.5

### 理由
- 在 AD-VLA 领域，efficiency 是真实痛点（推理慢、显存高），audience 广。
- prior work oracle 已证明 EPDMS 头部空间有限（≈0.41 天花板），追求绝对分数风险高。
- "iso-compute 持平" 是 reviewer 可信的故事，比 "算力大省 + 点数大涨" 更经得起 scrutiny。

### Reject 的方案
- **B 为主轴（刷 EPDMS 绝对分数）**：oracle ceiling 显示空间有限，且与 prior work 难以拉开差距。
- **无 budget target 的软优化** (reward = EPDMS − λ·budget)：缺 anchor，RL 会漂；改用硬约束 + 软优化的混合形式（见后续 Q）。
- **"all-in：算力大省 + EPDMS 大涨 + 方法极简"**：物理上不自洽，reviewer 会质疑评测可信度。

### 待后续 Q 决定的派生项
- λ / budget target 的具体数值 → 待 Q4 (data & loss design) 与 RL setup 一起定
- Iso-compute 是 hard constraint 还是 average constraint → 待 Q3 (颗粒度) 决定

---

## Q2. 剪谁的 token ✅ 2026-06-14

### 决策
- **位置**：**T1 — ViT 输出端 vision tokens**（projector 之后、LLM 输入之前）
- **变体**：**T1 + scene-context**
  - importance scorer 的输入不仅是 vision tokens 自身，还包括 driving scene context（driving instruction、ego state、navigation command 等）
  - novelty 来源：FastV/ToMe 都是 vision-only 的 importance；我们用 **driving-task-conditioned** importance，这是 AD-VLA 特有
- **算力指标**：
  - **主报**：vision token kept ratio（学术干净，与 FastV/ToMe 直接可比）
  - **附录报**：wall-clock 推理时间 / GPU memory 占用（工程吸引力）

### Paper 故事中的位置
- Method section 主干 = "Driving-context-conditioned token importance scoring at the ViT-LLM interface"
- 算力分析子节 = vision token 数 → LLM prefill FLOPs → wall-clock 的链条

### 算力构成参考（AutoVLA + 8 cams 粗估）
| 阶段 | 占比 | 我们能砍 |
|---|---|---|
| ViT 编码 | ~15% | ✗ |
| projector + 拼接 | ~1% | ✗ |
| LLM prefill (vision tokens) | ~70% | ✅ 主战场 |
| LLM decode (action AR) | ~14% | ✗ |

剪 50% vision tokens → LLM prefill ≈ -50% → 总 wall-clock ≈ -30~40%

### 理由
- T1 是 VLA token pruning 主流位置（FastV/ToMe/LLaVA-PruMerge 都在这里），baseline 直接可比。
- LLM prefill 是算力大头（~70%），T1 收益最大。
- T1 + scene-context 拉开与 FastV 的 novelty 差距，且不需要 hack LLM 内部结构。
- 工程改动只在 projector 后加 mask layer，风险可控。

### Reject 的方案
- **T2（LLM 中间层 KV cache 剪枝，FastV 风格）**：novelty 不够，FastV 已经做过；且 layer-k 是超参难调。
- **T3（ViT 内部 patch 剪枝，EViT/DynamicViT 风格）**：动 ViT 结构，工程量大，paper 会被归到 EViT 系列，与 VLA-for-AD 故事不符。
- **T4（action token 压缩）**：算力收益小（decode 只占 ~14%），且不是 token pruning，是 trajectory compression。
- **T1-only（不引入 scene-context）**：与 FastV 重合度过高，novelty 不足。

### 待后续 Q 决定的派生项
- "scene-context" 的具体形态（哪些字段、怎么 inject 到 importance scorer）→ Q3 / Q4
- importance scorer 是独立小模型 还是 复用 AutoVLA 内部表征 → Q3 / Q4

---

## Q3. 剪枝颗粒度 ✅ 2026-06-14

### 主决策（已 commit）
- **主方案**：**G4-with-camera-id**（= G24-b, unified）
  - **输入**：N = 8 cams × 256 tokens = 2048 个 vision tokens
  - **per-token feature** = ViT 输出的 token embedding **+ camera_id embedding**（8 维 learnable）
  - **输出**：
    - **Importance Scorer**：N 维 score ∈ ℝᴺ
    - **Budget Policy**：global budget B（一帧一个）
  - **选择规则**：在 N 个 token 中取 importance top-B，送入 LLM
  - **关键性质**：budget 在相机间**自由流通**（前向简单时可把算力让给后向复杂场景）

### Fallback 方案：G24-a (hierarchical) 或 纯 G2
- **G24-a**：先由 Budget Policy 输出 8 个 per-camera ratio r_i，再在每路相机内用 importance scorer 选 top-(r_i · 256)；budget 在相机间**不流通**
- **纯 G2**：G24-a 的极端简化版，每路相机内按固定规则（如保留前几个、随机）选 token，完全不用 importance；本质是 v4 精神的延伸

### Fallback 触发条件（任一满足即触发）
1. **RL 不收敛**：GRPO 在 global top-B 上训练 N steps 后，policy entropy 不降 / reward 不涨 / 输出退化为 trivial 解（如全选前 B 个、importance 输出常数）
2. **Importance signal 无效**：ablation 时把 learned importance 换成 random，EPDMS 无显著变化 → 说明 global selection 对 importance 信号不敏感，应退到 G2 让 budget 起主导
3. **B 底线破线**：iso-compute (avg ratio = 0.5) 下 EPDMS 比 baseline 掉 > 1 EPDMS 点（Q1 的 ε ≤ 0.5 已经破了 2×）

### 退化路径（按严重程度递进）
```
G4-with-camera-id  ──触发1或2──►  G24-a (hierarchical)  ──仍不行──►  纯 G2
```

### 理由（为什么主选 G4-with-camera-id 而非 G24-a/G2）
- **算力效率上限严格更高**：global budget 自由流通 ⇒ reward 上限 ≥ camera-locked budget
- **G24-a 是 G4-with-camera-id 的特例**：training 时如果 camera_id 是主导信号，模型会自然学到 "近似 per-camera ratio" 的行为；反之 G24-a 学不到 "前向某 token 比后向 token 重要" 这种 cross-camera 比较
- **参数量更少**：一个 scorer + 一个 budget head，而不是 8 个 scorer
- **paper story 更干净**："global token competition under scene-conditioned budget" 一句话讲清楚 method
- **G2 (v4 风格) 已 reject**：4-class 整帧分类器对应不上 "token-level adaptive pruning" 的 title，novelty 不足

### Reject 的方案
- **G1（per-scene scalar，v4 现状）**：撑不起 "token-level" title，novelty 不足
- **G3（per-token binary mask，无 budget 约束）**：N=2048 维 binary action space，RL 几乎不可能收敛
- **G24-a 作为主方案**：budget 不流通，上限严格低于 G24-b

### 细节决策 ✅ 2026-06-14

#### Q3.a — Importance Scorer 体量：**轻量 MLP**
- **架构**：2-3 层 MLP，~1M 参数
- **输入**：`[token_feature_i, camera_id_embedding, scene_context_embedding]`（scene_context broadcast 到每个 token）
- **输出**：N 维 importance score ∈ ℝᴺ
- **推理算力开销**：~2 GFLOPs（相对 LLM prefill 几十 TFLOPs 可忽略，保证 "省算力" 故事干净）
- **训练范式**：与 Budget Policy 一起走 D5（SFT warmup + GRPO RL）—— **MLP 架构与 RL 训练正交，MLP 完全可以 RL 训**

**Reject 的方案**：
- **小型 transformer scorer**：自身推理 ~8 TFLOPs，吃掉相当比例的 LLM 节省算力，破坏 Q1 的 A 主轴
- **复用 LLM 浅层 attention (FastV 风格)**：与 RL 训练冲突（要么 frozen 无法学，要么 unfreeze 整个 LLM = 重训 VLA），违背 Q2 "只在 ViT-LLM 接口加 mask 层" 的工程原则；且与 FastV novelty 重合

#### Q3.b — Budget Policy 输出：**离散 4-class**
- **输出空间**：B ∈ {0.25N, 0.5N, 0.75N, 1.0N}（与 v4 兼容，N = vision token 总数）
- **RL action space** = 4，GRPO 友好
- **Pareto curve** 报 4 个点，已与 FastV / ToMe / LLaVA-PruMerge 等社区习惯对齐

**Reject 的方案**：
- **K=8 离散**：不会提升 paper novelty（reviewer 在乎曲线形状不在乎点数），但 action space ×2 增加 RL 收敛风险，且 v4 训练 pipeline 无法直接迁移
- **连续 B ∈ [0, N]**：GRPO 不天然支持连续 action（需换 PPO），训练难度显著升高，且离散 4-class 已足够覆盖 Pareto curve 头部

**备选 trick**：如 paper rebuttal 阶段需要更精细 Pareto 点，可在**推理时**对多帧 budget 做插值/平均，得到"虚拟" K=8 点；训练保持 K=4。

#### Q3.c — Scene context 字段：**i + ii + iii 三件套（Stage 1）**
| 字段 | 来源 | 算力开销 | 状态 |
|---|---|---|---|
| (i) navigation command (4 类：直行/左转/右转/掉头) | NAVSIM 直接提供 | ~0（lookup embedding） | ✅ Stage 1 必加 |
| (ii) ego speed (标量) | NAVSIM 直接提供 | ~0（linear projection） | ✅ Stage 1 必加 |
| (iii) driving instruction (text) | AutoVLA 自带 | ~0（**复用 AutoVLA 现成 instruction embedding**） | ✅ Stage 1 必加 |
| (iv) 历史轨迹 (T × 3) | NAVSIM 提供 | ~0（小 MLP） | ⏸ **Ablation 备选**：Stage 1 跑通后加进来做 ablation table |
| (v) HD-map (polylines) | NAVSIM 提供，但需要 map encoder (VectorNet/MapTR) | 中等（+2-3 GB 显存 + 单独训 encoder） | ⚠ **Risk item**：暂不做；只在 Stage 1+2 都还有 EPDMS 余量时才考虑；加进来等于 paper 多一个 component，归因复杂度 ×2 |

**注入方式**：所有 scene context 先各自 embed 成定长向量，concat 成 `scene_context_embedding`，broadcast 到 N 个 token，与 token_feature 和 camera_id 一起送 MLP。

**Reject 的方案**：
- **Stage 1 就 all-in (含 map)**：ablation 组合爆炸 (2⁵ = 32)；map encoder 是独立子项目，risk 翻倍；归因困难

#### Q3.d — Training schedule：**Separate（两阶段）**
- **Stage A — Importance Scorer 独立训练**
  - 任务："给定 budget B（来自外部，训练时人为 sample 各种 B），学到能让 EPDMS 在 top-B 选择下最大化的 importance"
  - D5 范式：先用 attention distill SFT warmup，再 GRPO fine-tune
- **Stage B — Budget Policy 独立训练**
  - 任务："给定一个训好且 frozen 的 Importance Scorer，学到在每个 scene 给出最优 budget B"
  - D5 范式：同上
- **对应 README "两阶段 rework"** 叙事：Stage A = "学哪些重要"，Stage B = "学留多少"

**Reject 的方案**：
- **Joint training**：两个 loss 互相打架（importance 想让 budget 变大、budget 想让 importance 变 sharper），λ 难调；且两阶段对应 README 叙事更清晰

### 完整架构图（Q3 主决策 + 细节）
```
        8 cams images (NAVSIM)
              │
              ▼
   ┌───────────────────────┐
   │ AutoVLA ViT (frozen)   │
   └───────────────────────┘
              │
   ┌──────────┴──────────┐
   │ N = 2048 vision tok │
   └──────────┬──────────┘
              │
   ┌──────────▼──────────────────────────────┐
   │ Scene Context Encoder (lightweight)     │
   │  i.  nav_cmd  (4-class lookup)          │
   │  ii. ego_speed (linear)                 │
   │  iii.driving_instr (reuse AutoVLA emb)  │
   │  → scene_context_emb (broadcast)        │
   └──────────┬──────────────────────────────┘
              │
   ┌──────────▼──────────────────────────────┐
   │ Importance Scorer MLP (Q3.a)            │
   │  in:  [tok_feat, cam_id_emb, scene_emb] │
   │  out: score ∈ ℝᴺ                        │
   │  ~1M params, ~2 GFLOPs                  │
   └──────────┬──────────────────────────────┘
              │
   ┌──────────▼──────────────────────────────┐
   │ Budget Policy (Q3.b)                     │
   │  in:  scene_emb only                    │
   │  out: B ∈ {0.25N, 0.5N, 0.75N, 1.0N}    │
   └──────────┬──────────────────────────────┘
              │
   ┌──────────▼──────────────────────────────┐
   │ Top-B Selection (deterministic)         │
   │  keep top-B tokens by score             │
   └──────────┬──────────────────────────────┘
              │
              ▼
   ┌───────────────────────┐
   │ AutoVLA LLM           │  prefill on B vision tokens
   └───────────────────────┘
              │
              ▼
      ego trajectory → NAVSIM EPDMS
```

### 卡数与显存估算（粗）
- **Stage A SFT warmup**：8×A100 80G 足够（AutoVLA 3B frozen，只训 ~1M scorer）
- **Stage A GRPO**：4×A100 80G（per-card batch=2，rollout N=8/scene）
- **Stage B**：与 Stage A 类似
- **如未来加 (v) map**：+2 张卡训 map encoder

---

## Q4. 数据与监督信号 ✅ 2026-06-14（Q4.1/4.2/4.3/4.4 全部 commit）

### 总体范式：D5 = SFT warmup + GRPO RL
- **Stage A (Importance Scorer)**：SFT warmup → GRPO RL
- **Stage B (Budget Policy)**：SFT warmup → GRPO RL
- 两个 stage **separate 训练**（见 Q3.d）

---

### Q4.1 — Importance Scorer 的 SFT label ✅ 2026-06-14

#### 决策（主方案 L1 — Attention-distill from AutoVLA）

| 子问 | 选择 | 备注 |
|---|---|---|
| **Q4.1.a 用哪几层 attention** | **(iii) Probe 选最优 layer** | 在 100 个 scene 子集上跑 layer probing：对 LLM 每层 attention 与 group-perturbation oracle (L3) 算相关性，选 correlation 最高的 layer。paper 有 story ("we probe which LLM layer's attention aligns best with downstream driving quality") |
| **Q4.1.b 用哪个 query token** | **(i) last instruction token → vision token** | Default；(iii) action 首 token 留作 ablation |
| **Q4.1.c importance 的形式** | **(iii) ranking** | 我们只用 top-B 选择，**绝对值无意义，只有 ranking 有意义**。MLP 输出 score，但 loss 在 ranking 空间上算 |
| **Q4.1.d SFT loss** | **(iii) Listwise ranking (LambdaRank / ListNet)** | 与 c-iii 配套；和 top-B objective 对齐；比 MSE/KL 对 scale 更 robust |

#### L3 — Group-perturbation oracle（**保留为 sanity-check，必做**）
- **目的**：验证 L1 attention-distill 的合理性（attention 是否真的反映 EPDMS importance）
- **做法**：
  - 选取 100 个 scene 子集
  - 将 N=2048 vision tokens 按 K=16 个 region 分组
  - 每 region mask 后跑 AutoVLA inference 得 trajectory → 跑 NAVSIM EPDMS
  - region importance = EPDMS_no_mask − EPDMS_mask_this_region
- **使用方式**：
  - **Layer probing 的 ground truth**（Q4.1.a 用它选 layer）
  - **Paper 里报 L1 vs L3 的 Spearman correlation**，证明 attention-distill 合理
  - 如果 correlation 低（<0.3） → 触发 rethink，可能需要换 L1 → L3 直接当训练 label（但成本高）
- **算力预算**：100 scene × 16 region × 1 NAVSIM sim ≈ 1600 次 sim ≈ 1 张卡跑 1-2 天，可接受
- **L3 仅用于 100 scene sanity-check，不用作主训练 label**

#### 完整 pipeline（Q4.1 Stage A SFT 阶段）
```
Step 1: Layer probing (100 scenes)
    - 跑 L3 group-perturbation oracle  → region-level importance ground truth
    - 提取 AutoVLA LLM 每层 attention (last instr tok → vision tok)
    - 对每层算 Spearman(attention, oracle) → 选 best layer L*
    - paper figure: per-layer correlation curve

Step 2: Large-scale attention distill (full navtrain)
    - 跑 AutoVLA full inference,提取 L* 层 attention as label
    - 转 ranking → 存盘

Step 3: SFT train Importance Scorer MLP
    - Loss: listwise ranking (LambdaRank)
    - Input: token_feat + camera_id + scene_context
    - Output: score ∈ ℝᴺ
```

#### Reject 的方案
- **L2 (per-token perturbation oracle)**：N=2048 × 几万 scene × 1 sim = 天文 GPU 小时，**完全不可行**
- **L4 (cross-modal CLIP similarity)**：与 L1 重合（LLM 内部已在做），且不直接 task-relevant
- **L5 (no SFT, pure RL)**：违反 Q3 D5 决策；N=2048 维 importance 从 zero 学不动
- **L3 作为主训练 label**：算力虽可承受，但 group-level granularity 比 L1 还粗，且无 paper novelty 加分
- **L1.a (i) Layer 2 only**：FastV 的 finding 在 VLM-NLU 上验证过，**driving VLA 上不一定成立**，必须 probe
- **L1.c (i)(ii) 绝对值监督**：top-B 选择本质 ranking，绝对值监督引入无关 noise

---

### Q4.2 — Budget Policy 的 SFT label ✅ 2026-06-14

#### 前置 clarification（critical，别忘了）

**Oracle 跑 r ∈ {0.25, 0.5, 0.75} 时，那 r·N 个 token 由谁选？**

答：**由 Stage A 训好的 Importance Scorer 选 top-(r·N)**。
- ❌ 不是随机选（含金量太低，测的是平均水平不是上限）
- ❌ 不是 FastV 选（FastV 是 LLM 内部 attention，会让我们 oracle 跟 FastV baseline 撞车）
- ✅ **必须用 Stage A scorer**：oracle label 学的是"**给定我们自己的 selector 能力，scene 应该选多大 budget**"，这才是 Stage B 真正要解的问题

→ 直接后果：**Stage A 必须先训完，再造 Stage B oracle label**。这也是 Q3.d Separate 训练顺序的工程必然。

#### 术语统一（paper 和 doc 都用这套）

| 术语 | 含义 |
|---|---|
| **External pruning** (我们) | ViT-LLM 接口剪枝，LLM 还没 forward 就已选定 B 个 token |
| **Internal pruning** (FastV / ToMe-in-LLM) | LLM forward 过程中剪 |
| **Selector** | 决定 token 重要性的模块（FastV: LLM 自带 attention；Ours: trained MLP scorer） |
| **Budget controller** | 决定每 scene 留多少 token（FastV: 无，固定；Ours: trained Budget Policy） |

#### 我们 vs FastV 的清晰边界（Q4.1 attention-distill ≠ FastV）

| 维度 | FastV | 我们 |
|---|---|---|
| 剪枝位置 | LLM 内部 (layer 2 后) | ViT 输出端 (LLM 入口) |
| Selector | LLM layer-2 attention (off-the-shelf) | trained MLP scorer (driving-finetuned) |
| Budget | 固定全局 ratio | scene-adaptive (Budget Policy) |
| Training | 不需要 | 需要 (SFT + RL) |
| 算力节省 | 仅省 layer 3-N | 省全部 LLM forward |
| **使用 LLM attention 的时机** | inference time 每次都用 | **只在 training time 造 label，inference 不再依赖** |

**Key insight**："we distill attention once, then run cheap MLP scorer at inference."

#### 决策

| 子问 | 选择 | 备注 |
|---|---|---|
| **Q4.2.a 主方案** | **(i) B3 — Pareto-aware oracle** | "在 EPDMS ≥ max_EPDMS − ε 的所有 r 中，取最小 r" → label 编码省算力 prior，与 Q1 A 主轴对齐 |
| **Q4.2.b ε 怎么定** | **(iii) 100-scene 子集扫 ε ∈ {0.005, 0.01, 0.02, 0.05}** | 看不同 ε 下 label 在 4 类 {0.25, 0.5, 0.75, 1.0} 的分布，选**分布最均衡的 ε** 作为正式值。避免 ε 太大全打 0.25、ε 太小全打 1.0 |
| **Q4.2.c B5 (no-SFT) ablation** | **(i) 必做** | paper 加分项：B5 ≈ B3 → "lightweight RL is enough for budget"；B5 ≪ B3 → "two-stage SFT-then-RL is necessary"。无论哪个结果都能讲故事 |
| **Q4.2.d B5 init** | **(i) uniform init**（输出 4 类等概率） | random init 容易 collapse；prior-work checkpoint init 与 AutoVLA 不同 backbone，transfer 不保证 |

#### 完整 Pipeline（Q4.2 Stage B SFT 阶段）

```
[前提] Stage A scorer_v1 已训完且 frozen

Step 1: Oracle inference (full navtrain)
    - 对每个 scene，用 scorer_v1 选 top-{0.25N, 0.5N, 0.75N, 1.0N} 四组 token
    - 跑 AutoVLA inference → ego trajectory → NAVSIM EPDMS
    - 注意：r=1.0 = baseline，已在 Stage A 阶段产出，复用即可
    - 实际算力 = 3× full navtrain inference（不是 4×）
    - 输出：每 scene 一组 (EPDMS_0.25, EPDMS_0.5, EPDMS_0.75, EPDMS_1.0)

Step 2: ε scan on 100-scene subset (Q4.2.b)
    - 对 ε ∈ {0.005, 0.01, 0.02, 0.05}：
        - 对每 scene：max_EPDMS = max over 4 个 r
        - label B* = min {r : EPDMS_r ≥ max_EPDMS − ε}
    - 看 4 类 label 的分布直方图
    - 选最均衡的 ε* 作为正式值
    - paper figure: label distribution vs ε

Step 3: 用 ε* 在全量 navtrain 上造 label

Step 4: SFT 训 Budget Policy
    - 输入：scene_context_emb only（nav_cmd + ego_speed + driving_instr）
    - 输出：4-class logits over {0.25, 0.5, 0.75, 1.0}
    - Loss: CE (4-class)
    - 备注：可加 class-weighted CE 应对不均衡

Step 5: B5 ablation (parallel branch)
    - 同样架构 Budget Policy，但跳过 Step 3-4 SFT
    - uniform init → 直接 GRPO RL（reward 见 Q4.3）
    - 与主分支比 final EPDMS / efficiency 曲线
```

#### 算力预算（Q4.2 部分）

| 步骤 | 算力 | 备注 |
|---|---|---|
| Oracle r=0.25, 0.5, 0.75 inference | 3× navtrain inference | r=1.0 复用 Stage A |
| ε scan (100 scene) | 已包含在上一步 | 只是 post-processing |
| Budget Policy SFT | 极小（4-class classifier）| 几小时 |
| B5 ablation (no-SFT) | 极小，但需走完 GRPO | 与主 RL 同算力数量级 |

**累计 oracle inference 总成本（Q4.1 + Q4.2）= ~4× navtrain inference**（与之前估算一致）。

#### Reject 的方案
- **S1 随机选 token 造 oracle**：测的不是 budget 上限而是平均水平，含金量低
- **S2 用 FastV 选 token 造 oracle**：(a) oracle 已用 FastV selector，paper 里无法再独立对比 FastV；(b) FastV 是 internal pruning，架构错位
- **S4 用 L2 perturbation oracle 选 token**：算力天文数字
- **B1 vanilla EPDMS-argmax**：不编码省算力 prior，与 A 主轴脱钩
- **B2 soft label (EPDMS softmax)**：EPDMS scale 难标定，softmax 温度难调
- **B4 heuristic label**：违背 "learned" 故事，仅可作 dummy baseline
- **B5 作为主方案**：4-class action space 虽小，但 from-scratch RL 仍有 collapse 风险；保留为 ablation 而非主方案
- **B5 random init**：易 collapse 到单一类
- **B5 prior-work checkpoint init**：该 checkpoint 在不同 backbone 上训得，与 AutoVLA 不同，transfer 无保证（仅作额外 ablation 备选）

#### FastV baselines 计划（与 Q5 评测衔接）
**Q5 必报的 FastV 相关 baseline（在这里先记一笔，Q5 再细化）**：
1. **Vanilla FastV** at multiple ratios → 验证 internal vs external pruning
2. **FastV-selector-at-input**：把 LLM layer-2 attention 拿到 LLM 入口当 selector，固定 ratio → 拆开 "位置 gain" 和 "selector gain"
3. **Ours**：external pruning + learned scorer + adaptive budget

---

### Q4.3 — GRPO reward 函数 ✅ 2026-06-14

#### 设计理念
- **Stage A scorer** 与 **Stage B budget policy** 都走 GRPO，但 reward 结构 **deliberately 不同**：scorer 专注 quality，budget policy 同时管 quality + efficiency
- Stage B reward (R3 piecewise Pareto) 与 Stage B SFT label (Q4.2 B3 Pareto-aware oracle) **同构** —— SFT label 用 "EPDMS ≥ max − ε 内选最小 r"，RL reward 用 "EPDMS ≥ baseline − ε_quality 时奖励省 token"，**两阶段 objective 完全一致**，这是 method 在数学上最 well-motivated 的一环

#### 决策

| 子问 | 选择 | 一行总结 |
|---|---|---|
| **Q4.3.a Stage A reward** | **(i) R4 pure EPDMS-advantage** | scorer 只优化 ranking quality，不掺 efficiency |
| **Q4.3.b Stage B reward** | **(iii) R3 piecewise Pareto** | 不掉点奖励省 token；掉点则只惩罚 quality |
| **Q4.3.c EPDMS 形式** | **(iii) per-scene advantage** `ΔEPDMS = EPDMS_pruned − EPDMS_baseline` | 消除 scene difficulty confound |
| **Q4.3.d α / β** | **(ii) 100-scene grid scan** | α, β ∈ {0.05, 0.1, 0.2, 0.5}，与 Q4.2.b ε scan 合并做 |
| **Q4.3.e Reward clip** | **(ii) clip to [−1, 1]** | 防极端 collision-induced EPDMS 崩盘摧毁 policy |

#### Final reward 公式

**Stage A (Importance Scorer RL)**：
```
R_A = clip( EPDMS_pruned − EPDMS_baseline,  −1, +1 )
```
- 训练时 budget B 从 {0.25N, 0.5N, 0.75N, 1.0N} 随机 sample 喂入
- scorer 学 "given any B, pick the best top-B"

**Stage B (Budget Policy RL)**：
```
ΔEPDMS = EPDMS_pruned − EPDMS_baseline       # 通常 ≤ 0

if ΔEPDMS >= −ε_quality:                     # 没掉超过 ε
    R_B = ΔEPDMS + α · (1 − B/N)             # 奖励省 token
else:                                         # 掉超过 ε
    R_B = ΔEPDMS − β · |ΔEPDMS + ε_quality|  # 仅惩罚 quality，不奖励 efficiency

R_B = clip(R_B, −1, +1)
```
- `ε_quality` 复用 Q4.2.b 扫出的 ε*
- `α, β` 在同一 100-scene 子集上 grid scan
- Stage A scorer 已 frozen，B 由 policy 决定

#### RL 视角的设计理由（paper Method 章节直接取用）

**Q4.3.a — Decoupling（解耦）**
> Importance Scorer 的唯一职责是 "Given B, find the best subset"。如果把 efficiency penalty 加给它，它为了赚这个 reward，可能会倾向于输出无区分度的 score 导致大面积失效，从而扰乱 policy。让它纯粹地专注于 quality (EPDMS)，是维持训练稳定性的最佳策略。

**Q4.3.b — Asymmetric reward landscape（非对称奖励地貌）**
> 在掉点超过 ε_quality 时，只惩罚 quality 不奖励 efficiency，这本质上是一个 "safety hook"，强迫 policy 退回到安全区；而在不掉点时，给 efficiency 发 bonus。这种设计完美契合了 autonomous driving 这种对 safety/quality 容忍度极低的任务。

**Q4.3.c — Variance Reduction（方差缩减）**
> NAVSIM 中不同 driving scene 的 intrinsic difficulty 差异极大（直行可能轻松拿到 0.9，复杂左转可能 baseline 只有 0.3）。如果用 raw EPDMS，RL 很容易迷失在 scene 级别的噪声中。引入 per-scene baseline 就像是加了一个完美的 Critic，让 advantage ΔEPDMS 纯粹反映 pruning 带来的边际影响。

**Q4.3.d — Hyperparameter sensitivity**
> RL 对 hyperparameters 极其敏感，α 和 β 的相对大小决定了探索的方向。在 100-scene 小集上通过 grid scan 锁定一组合适的 scale，比盲目假设 α = β = 0.1 稳妥得多，且极具可操作性。

**Q4.3.e — Stability against tail events**
> GRPO 在 rollout 阶段，一旦某个极端的 pruning 导致碰撞，EPDMS 瞬间掉到底，算出来的 ΔEPDMS 会产生巨大的负值梯度，瞬间摧毁 policy。Clip to [−1, 1] 是兜底操作。

#### 完整 RL pipeline

```
[共用前提]
  - EPDMS_baseline 已在 Stage A oracle 阶段为每 scene 算好（r=1.0 那次）
  - 全部 EPDMS 走 NAVSIM evaluator

Stage A RL (Importance Scorer fine-tune):
  for each scene:
    sample B ∈ {0.25N, 0.5N, 0.75N, 1.0N} (uniform)
    for k = 1..K rollouts (GRPO group):
      score ~ scorer(token_feat, cam_id, scene_ctx)   # stochastic sampling
      select top-B by score
      EPDMS_pruned = NAVSIM(AutoVLA(top-B tokens))
      R_k = clip(EPDMS_pruned − EPDMS_baseline, -1, 1)
    update scorer via GRPO advantage over K rollouts

Stage B RL (Budget Policy fine-tune):
  freeze Stage A scorer (post Stage A RL)
  for each scene:
    for k = 1..K rollouts (GRPO group):
      B ~ budget_policy(scene_ctx)                    # 4-class sample
      select top-B by frozen scorer
      EPDMS_pruned = NAVSIM(AutoVLA(top-B tokens))
      ΔE = EPDMS_pruned − EPDMS_baseline
      if ΔE >= -ε_quality:
          R_k = ΔE + α·(1 - B/N)
      else:
          R_k = ΔE − β·|ΔE + ε_quality|
      R_k = clip(R_k, -1, 1)
    update budget_policy via GRPO advantage over K rollouts
```

#### Reject 的方案
- **R1 linear scalarization** `R = EPDMS − λ·B/N`：λ 极难调；EPDMS 与 B/N 分布不同 scale，λ 一变 policy 就完全变；与 SFT label (B3) 不同构
- **R2 constrained (hard B_max)**：hard constraint 在 GRPO 里 reward shaping 难处理；与 B3 oracle 不同构
- **R4 pure EPDMS for Stage B**：policy 直接 collapse 到 r=1.0，efficiency 完全失效
- **R5 sub-component weighted reward**（collision/progress/comfort 各加权）：调参空间爆炸，且违背 "用 EPDMS 作为统一评测" 的 paper 原则；**仅作 ablation 备选**
- **Raw EPDMS as reward**：scene difficulty confound，variance 极大
- **不 clip reward**：极端 collision case 会瞬间摧毁 policy
- **α/β 固定 0.1**：RL 对超参极敏感，无 scan 风险高
- **Curriculum α/β**（训练前期重 quality 后期重 efficiency）：**留作 future work**，本工作先 plateau 一组 α/β

---

### Q4.4 — 训练数据切分 ✅ 2026-06-14

#### 数据用途分类

| 用途 | 来源 | 数据量 | 用在哪 |
|---|---|---|---|
| **A. Probe set** | navtrain 子集（独立切出） | 100 scene | layer probing (Q4.1)、ε scan (Q4.2.b)、α/β scan (Q4.3.d) |
| **B. Oracle 数据集** | navtrain \ A | 全量 | Stage A attention 提取 + Stage B oracle EPDMS 跑分 |
| **C. RL 数据集** | navtrain \ A | 全量（= B） | Stage A/B GRPO rollout |
| **D. 评测集** | navtest（主报）/ navhard（future work，见文末 Revision 2026-06-15） | 既定 | 最终 EPDMS 报数 |

#### 决策

| 子问 | 选择 | 一行总结 |
|---|---|---|
| **Q4.4.a Probe 怎么选** | **(ii) Stratified by nav_cmd**（4 类各 25） | 对抗 "go straight bias"，保证 4 类 nav cmd 在 probe 中均有代表 |
| **Q4.4.b B 与 C 是否同批** | **(i) B = C = navtrain \ A** | oracle 是 NAVSIM 物理仿真 ground truth，非 fitted parameter，无 leakage |
| **Q4.4.c Train/val 切分** | **(ii) 90/10 random split** | 给 SFT early stopping 和 RL model selection 用 |
| **Q4.4.d SFT/RL 是否共用 val** | **(i) 共用同一 val** | 消除 distribution shift，SFT loss 与 RL reward 在同一基准比 |
| **Q4.4.e 评测集** | **主报 navtest（navhard 列入 future work）** | AutoVLA 原生不支持 navhard 双路径范式；详见文末 Revision 2026-06-15 |

#### 数据切分理由（paper Experimental Setup 直接取用）

**Q4.4.a — Stratified by nav_cmd**
> 自动驾驶数据集有极强的 "go straight bias"（直行场景可能占 70%+）。如果纯随机抽样，100 个 scene 里可能只有几个转弯。这样扫出来的超参（特别是 ε_quality）会完全 overfit 到简单的直行场景，导致模型在复杂交叉路口时 policy 崩溃。Stratified by nav_cmd（直/左/右/掉头各 25）完美解决了这个问题。

**Q4.4.b — Oracle 不是 fitted parameter**
> Oracle 跑 EPDMS 是物理仿真算出来的客观 ground truth，它不是一个通过梯度下降拟合出来的 parameter 分布。因此不存在 model-level 的 leakage，B 与 C 用同一批数据没问题。把数据量拉满对 RL 的泛化性最有帮助。

**Q4.4.c & d — 共用 val 的工程意义**
> 共用同一个 validation set 是非常好的 engineering habit。这样在看 Stage A (SFT) 的 val loss 和 Stage B (RL) 的 val reward 时，是在完全相同的分布基准上做对比，消除了一切 data distribution shift 带来的干扰。

**Q4.4.e — 评测呈现策略**
> 主表格报 navtest，保证和 FastV、ToMe 以及其他 AutoVLA baseline 绝对公平可比，堵住 reviewer 挑刺的嘴。
>
> ⚠️ **Revision 2026-06-15**：原计划 ablation 章节加 navhard robustness 数据，但因 AutoVLA 上游 navsim fork 不原生支持 navhard_two_stage 双路径评估范式（详见文末 Revision 段落），navhard 数据降级为 future work。在 paper 中可在 §Limitation/Future work 一笔带过，主表 navtest 站得住即可。

#### 最终数据流图

```
                    navtrain (full)
                          │
              ┌───────────┴───────────┐
              │                       │
       Probe set A                navtrain \ A
       (100 scene,                (剩余全量)
        nav_cmd                       │
        stratified)             ┌─────┴─────┐
              │                90% train  10% val
              ▼                  │           │
   layer probing (Q4.1)         ▼           ▼
   ε scan (Q4.2.b)         Stage A/B    共用 val
   α/β scan (Q4.3.d)       SFT + RL    (SFT early stop
                          (B = C 同一批)  + RL model select)

                    navtest (主报)     navhard (future work)
                          │                  │
                          ▼                  ▼
                  Main table EPDMS    [Revision 2026-06-15: 见文末]
```

#### Reject 的方案
- **P1 (probe ⊂ B/C)**：probe 与训练集重叠，超参选完后又拿同样数据 train，轻微 leakage，严谨度差
- **P3 (B ≠ C 不重叠)**：oracle 是物理仿真不是拟合，无 leakage 风险；切分只会无谓损失数据量
- **(i) probe 随机抽样**：必然 overfit 直行场景，转弯 / 复杂路口超参 underfit
- **(iii) probe stratified by nav_cmd × ego_speed bin**：100 scene 摊到 12+ 桶过细，每桶样本不足
- **(iv) probe diversity clustering**：实现复杂，nav_cmd stratify 已足够
- **(i) train/val 不切**：缺 model selection / early stopping 信号
- **(ii) SFT/RL 独立 val split**：引入 distribution shift，两阶段对比不可信
- **(ii) navhard only**：与社区主流脱钩，无法与 FastV/ToMe 直接对比
- **(iv) navtest + navhard 全报**：稀释 main result，evaluator 容易混乱
- **(v) 主报 navtest + ablation 报 navhard**：原计划方案，因 AutoVLA 不原生支持 navhard 已降级（Revision 2026-06-15）

---

## Q5. 评测协议 ✅ 2026-06-14

### 设计理念
- 评测体系是 paper credibility 的根基。Q5 的目标：**让 reviewer 无处挑刺**
- 主轴策略：**主报 navtest（社区主流，公平对比）**（详见 Q4.4.e；navhard ablation 已降级为 future work，见文末 Revision 2026-06-15）
- 评测不堆砌、不模糊：每个 baseline、每个 ablation 都对应回答一个明确的 reviewer 问题

### 决策

| 子问 | 选择 | 一行总结 |
|---|---|---|
| **Q5.a 主报指标** | **(ii) EPDMS + safety (collision) + comfort + progress 3 子项** | 直接回答 "省 token 是否伤 safety" 这个核心质疑 |
| **Q5.b efficiency 指标** | **(ii) avg kept ratio + FLOPs + latency 三件套** | kept ratio 对齐 Q1 A 主轴；FLOPs 是 reviewer 偏爱的硬指标；latency 拓宽 audience |
| **Q5.c Baselines** | **精简 7 个核心 baseline**（见下表） | 砍掉 ToMe / VisionZip / prior-work line，避免堆砌同质化 SOTA |
| **Q5.d Main result 形态** | **(ii) Pareto curve figure + ratio=0.5 main table** | 一图一表 ML 顶会标配 |
| **Q5.e Ablation 范围** | **(ii) A1–A10 全做** | Q1–Q4 已埋好的伏笔（ε / α / β / layer probing）surface 出来即可 |

### Q5.a — 主报指标体系

| 指标 | 类型 | 目的 |
|---|---|---|
| **EPDMS** | 综合分 | 主报 KPI |
| **Collision rate** | 子项 | 直接回答 "省 token 是否影响 safety" |
| **Comfort** | 子项 | 平稳性 |
| **Progress** | 子项 | 任务完成度 |

> Reviewer 必然会问 "你省了 token，是不是 collision 变多了？" 主报 EPDMS + 拆 3 个核心子项（safety / comfort / progress）正好回答这个问题，且不至于信息过载。

### Q5.b — efficiency 三件套

| 指标 | 含义 | 必要性 |
|---|---|---|
| **avg kept ratio** = avg(B/N) | Q1 A 主轴指标 | ≥ 50% saving 是 paper 的 success criterion |
| **FLOPs** (vision token forward) | 学术界硬指标 | reviewer 喜欢 |
| **Latency** (real wall-clock, GPU 推理) | 产业界关心 | 拓宽 audience |

三个数互补，reviewer 没法挑刺。

### Q5.c — Baselines（精简 7 个核心，砍掉同质化 SOTA）

| # | Baseline | 角色 | 回答的问题 |
|---|---|---|---|
| 1 | **AutoVLA r=1.0** (vanilla, no pruning) | Upper bound | 不剪枝的香草版本 |
| 2 | **Random pruning** at r=0.5 | Lower bound | 证明 method 确实学到了东西，而不是 NAVSIM 太简单可以随便乱扔 token |
| 3 | **FastV (internal pruning)** at matched FLOPs | SOTA 主竞争对手 | 直面对手；本地已跑通端到端推理闭环，工程 overhead 极低 |
| 4 | **FastV-selector-at-input** at matched ratio | Critical naive baseline | 控制变量：把 FastV 的 attention 均值作为重要性分数，但在 T1（ViT-LLM 接口）位置执行剪枝。**打赢这个，"Driving-context-conditioned Scorer 远胜纯 Vision Attention" 的 novelty 就焊死** |
| 5 | **Ours Fixed-Budget** (Stage A only, r=0.5 fixed) | Adaptive budget ablation | 打赢它就证明 "动态 budget > 固定 budget"，Stage B RL 故事立住 |
| 6 | **Ours RL-Drive Full** (Stage A + Stage B) | Main method | Adaptive budget 下 Pareto 最优 |
| 7 | **Fixed-ratio + LLM raw attention** at r=0.5 | Sanity baseline | 最 naive 的"位置正确但 selector naive"对照 |

#### Reject / 降级的 baseline

| Baseline | 处理 | 原因 |
|---|---|---|
| **ToMe** (Token Merging) | **降级到 Appendix / Rebuttal 弹药** | Token Merging 会改变 feature 表达，与 Pruning 属于两个赛道。主表格没必要混在一起 |
| **VisionZip / SparseVLM** | **降级到 related work + Appendix** | 引言重点 cite 即可。如果 reviewer 强制要求比，再拿出来作为 rebuttal 弹药 |
| **Prior internal work (different backbone)** | **坚决 Reject** | 内部技术债，外界 reviewer 不 care 个人工作迭代。放进去要花大量篇幅解释其细节，会冲淡 RL-Drive 的独立性和创新性 |

> Baseline 选择策略的核心：**每个 baseline 都对应回答一个明确的 reviewer 问题**，不为了凑数堆砌同质化 SOTA。

### Q5.d — Main result 形态

| 元素 | 形态 |
|---|---|
| **Main figure** | Pareto curve: EPDMS (y) vs FLOPs (x)，曲线必须严格 dominate baseline 1–4, 7 |
| **Main table** | ratio=0.5 这一列，列出 7 个 baseline × 4 个指标（EPDMS / collision / comfort / progress）+ FLOPs / latency |

> Pareto curve 是 Q1 定的"main result"形态，必须画。Main table 锁 ratio=0.5（与 Q1 A 主轴对齐），方便逐 baseline 对比。一图一表是 ML 顶会标配。

### Q5.e — Ablations 体系（A1–A10）

| # | Ablation | 回答的问题 | 对应决策 | 数据来源 |
|---|---|---|---|---|
| **A1** | w/o RL (SFT only) | SFT 够不够？ | Q3.d 两阶段 | 单独跑 |
| **A2** | w/o SFT (RL from scratch, B5) | SFT warmup 必要？ | Q4.2.c | 单独跑 |
| **A3** | w/o Budget Policy (fixed ratio at r=0.25/0.5/0.75) | adaptive budget 有用？ | Q1 C novelty | 已是 baseline 5 |
| **A4** | w/o Importance Scorer (random / uniform select) | scorer 有用？ | Q1 C novelty | 已是 baseline 2 |
| **A5** | Attention layer choice ({3, 6, 9, 12, ...}) | layer probing 对结果敏感？ | Q4.1 | 复用 layer probing 数据 |
| **A6** | ε_quality sensitivity ({0.005, 0.01, 0.02, 0.05}) | label rule 健壮？ | Q4.2.b | 复用 ε scan 数据 |
| **A7** | α/β reward sensitivity grid | reward 健壮？ | Q4.3.d | 复用 α/β scan 数据 |
| **A8** | EPDMS sub-component breakdown | 哪个子项受影响？ | Q5.a | 主表自带 |
| ~~**A9**~~ | ~~navhard performance~~ | 复杂场景 robust？ | Q4.4.e | 降级为 future work（Revision 2026-06-15）|
| **A10** | FastV-selector-at-input | selector 自身 gain？拆 "位置 gain vs selector gain" | Q4.2 末尾 | 已是 baseline 4 |

> A5–A10 都是 Q1–Q4 里已经埋好的伏笔（ε scan、α/β scan、layer probing 本来就要做），ablation table 把这些数据 surface 出来即可，**几乎没有额外算力开销**。CVPR/NeurIPS 一篇好 paper 通常 10+ ablations 表格。
> ⚠️ **Revision 2026-06-15**：A9 navhard performance 已降级为 future work，剩余 9 个 ablation (A1–A8, A10) 全部保留。

### Reject 的方案
- **(i) Only EPDMS for main metric**：reviewer 必问 collision，不拆子项就被挑
- **(iii) EPDMS + 全部子项**：信息过载，reader 抓不到重点
- **(i) Only kept ratio for efficiency**：缺 FLOPs / latency reviewer 不买账
- **(iii) Only FLOPs**：缺 latency 没法对接产业界 audience
- **(i) Baseline 1–4 minimal**：缺 FastV-selector-at-input 这个关键 ablation，selector gain 故事讲不圆
- **(iii) 1–9 全报**：堆砌同质化 SOTA，稀释主线
- **(i) Main table only**：Pareto 是 Q1 定的卖点形态，必须画图
- **(iii) Multiple tables across ratios**：信息冗余，Pareto curve 已涵盖
- **(i) Ablations A1–A4 only**：A5–A10 数据已经在跑，不 surface 浪费
- **(iii) A1–A4 + 选 3–4 个**：取舍标准模糊，不如全做

---

## Revision 记录

### Revision 2026-06-15: Benchmark 切换 navhard_two_stage → navtest

**触发**

AutoVLA 上游 navsim fork（基于 navsim v1.x，CVPR 2024 challenge 时期）原生**不支持** navhard_two_stage 双路径 + reactive synthetic 评估范式：
- `default_evaluation.yaml` 只有单一 `sensor_blobs_path`（无 `synthetic_sensor_path` / `original_sensor_path`）
- `SceneLoader` 不区分 stage1（真实 log 图像）与 stage2（合成图像）
- 全代码库零字眼匹配 `navhard / two_stage / synthetic_sensor / reactive_synthetic`
- 上游官方评估目标本身就是 `navtest`（见 `default_run_pdm_score.yaml override train_test_split: navtest`）

要让 AutoVLA 跑 navhard，需 patch SceneLoader / merge sensor 目录 / port 双路径机制（参考 prior work `tokenrl/code/third_party/navsim` 的 v2 实现），属于框架级改造，超出本项目核心 scope（vision token pruning method），且引入大量与 method 无关的工程债。

**决策**

主评测改为 **navtest**（AutoVLA 原生支持，0 改造），navhard 评测降级为 **future work**：
- 主表（M6 main table）EPDMS 全部基于 navtest
- 数据切分：仅 `navtrain` + `navtest`，**不再下载 `navhard_two_stage` (~? GB)**
- M0 baseline、M3 oracle generation、M6 ablation 全部对齐 navtest

**Future work 的 navhard 入口**（保留但不在本轮跑）：
- 若需要补 navhard 数字，可复用 prior-work 内部仓库下的 navsim v2 完整实现（已含双路径 SceneLoader、navhard_two_stage split 配置、`run_navhard_4gpu.sh` 评估脚本）
- 接入路径：让 AutoVLA inference wrapper 输出 ReCogDrive submission.pkl 兼容格式，复用 prior work 的 NAVSIM v2 eval entry（即原 R-M0-1 Plan B 路线）

**影响**

| 模块 | 变化 |
|---|---|
| `implementation_plan.md` M0 | 删 `baseline_r1.0_navhard.pkl` 产物；splits 不再含 `navhard.pkl` |
| `implementation_plan.md` M6 | 删 ablation A9 (navhard performance)；产物表删 `M6_navhard.csv` |
| `risks.md` R-M0-3 | 关闭（navtest 不需要 stage1 cv cache） |
| `risks.md` R-M6-2 | 关闭（不再评测 navhard） |
| `risks.md` R-M0-1 / R-M0-2 / R-M4-1 | 失败现象描述里的 `navhard` 字眼改为 `navtest` |
| `design_decisions.md` Q4.4.e | 由 "主报 navtest + ablation 报 navhard" 简化为 "主报 navtest（navhard 列入 future work）" |
| 配置文件 | `qwen2.5-vl-3B-navhard.yaml` 删除 / 重命名为 `qwen2.5-vl-3B-navtest.yaml`，sensor 路径改回单 `sensor_blobs/test` |

**Reject 的方案**
- **A1**：给 AutoVLA navsim 打补丁强行支持 navhard —— 改造量大、与 method 无关、易引入 silent bug
- **A2**：把 AutoVLA 的 PYTHONPATH 切到 prior work 的 navsim v2 —— 风险中等但 import 拓扑改动大，且 AutoVLA 上游升级时合并冲突高
- **A3**：合并 sensor 目录 hack —— 能跑但脏，paper 里说不清
- **A4**：换 backbone（LMDrive / OmniDrive 原生支持 navhard）—— 推迟整体进度过多，且失去 AutoVLA 这个 SOTA 对标点

**论证**：navtest 在社区主流中仍是公平可比的标准 benchmark（FastV / ToMe / 各 AutoVLA 衍生工作均以 navtest 为主报），不影响 paper main claim 的 reviewer 接受度。
