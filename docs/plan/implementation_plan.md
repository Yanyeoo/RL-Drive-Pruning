# RL-Drive Implementation Plan (post Q1–Q5 design freeze)

> **Created**: 2026-06-14 19:05（design freeze 之后）
> **Source of truth**: 本文档把 `design_decisions.md` 的 Q1–Q5 翻译为可执行 milestone。
> **Companion docs**:
> - `design_decisions.md` — 决策本身（why）
> - `risks.md` — 每个决策点的失败 contingency（what if）
> - `paper_outline.md` — 决策映射到 paper 章节（write-up）

---

## 总览：里程碑全图

```
M0 Infra & Baseline
   ├─ 环境 / repo / NAVSIM pipeline
   └─ AutoVLA r=1.0 EPDMS baseline (per-scene)
        │
        ▼
M1 Stage A: Importance Scorer SFT  (Q4.1 attention distill)
   ├─ M1a Layer probing (100 scene)
   ├─ M1b 全量 navtrain attention 提取 + ranking label
   └─ M1c LambdaRank SFT → scorer_v1 (frozen)
        │
        ▼
M2 Stage A: Importance Scorer RL  (Q4.3.a)
   ├─ pure EPDMS-advantage reward
   ├─ random-B sampling
   └─ → scorer_v2 (RL-tuned, frozen for Stage B)
        │
        ▼
M3 Stage B Oracle Generation  (Q4.2.a B3 Pareto-aware)
   ├─ 用 scorer_v2 选 r ∈ {0.25, 0.5, 0.75} top-B
   ├─ navtrain 全量跑 EPDMS（r=1.0 复用 M0）
   ├─ ε scan on 100-scene probe (Q4.2.b) → 选 ε*
   └─ → 全量 budget label
        │
        ▼
M4 Stage B: Budget Policy SFT  (Q4.2)
   ├─ 4-class CE on budget label
   └─ → budget_policy_v1
        │
        ▼
M5 Stage B: Budget Policy RL  (Q4.3.b R3 Pareto reward)
   ├─ α/β grid scan on 100-scene probe (Q4.3.d)
   ├─ piecewise Pareto reward + clip[-1,1]
   └─ → budget_policy_v2 (final)
        │
        ▼
M6 Evaluation & Ablations  (Q5)
   ├─ 7 baselines × navtest main table (ratio=0.5)
   ├─ Pareto curve figure
   ├─ A1–A10 ablations
   └─ navhard robustness analysis
```

---

## 算力预算总览（全 pipeline）

> 单位 = "1 次 navtrain inference"（约 navtrain 全量 forward 一遍 AutoVLA 的算力）

| 阶段 | 算力 | 备注 |
|---|---|---|
| M0 baseline (navtrain r=1.0) | 1× | 即 EPDMS_baseline，per-scene 存盘 |
| M1a layer probing (100 scene × N layers) | ~0.05× | 100 scene 量小 |
| M1b 全量 attention 提取 | 1× | inference 一遍即可，额外存 attention |
| M1c SFT scorer | 极小 | MLP，几小时 |
| M2 Stage A RL | ~K× rollouts | K = GRPO group size（典型 4–8） |
| M3 oracle r ∈ {0.25, 0.5, 0.75} | 3× | r=1.0 复用 M0 |
| M4 SFT budget policy | 极小 | 4-class classifier |
| M5 Stage B RL | ~K× rollouts | 同 M2 |
| M6 主评测 | navtest × 7 baselines + ablations | 详见 M6 |

**核心成本估算**：oracle inference (M0 + M1b + M3) 约 **5× navtrain inference**，是 pipeline 的算力大头。

---

# M0 — Infra & Baseline

**目标**：环境就位 + 拿到 AutoVLA r=1.0 在 navtrain 全量的 per-scene EPDMS（= EPDMS_baseline）

**对应决策**：Q4.4（数据切分前置）、Q5（baseline #1）

## M0 子任务

### M0.1 环境
- [ ] AutoVLA repo clone + 环境装齐（复用 `envs/navsim`，增量装 trl/peft）
- [ ] 单帧 inference < 5s sanity check
- [ ] 双 H20 推理脚本就位（参考 prior-work 的 `run_oracle_navhard_dual_gpu.sh` 模式）

### M0.2 数据切分（Q4.4）
- [ ] navtrain 列表 dump
- [ ] **probe set A**：从 navtrain 按 nav_cmd stratified 抽 100 scene（直/左/右/掉头各 25），存 `data/splits/probe_100.pkl`
- [ ] **train pool**: navtrain \ A，按 90/10 random split → `train.pkl` / `val.pkl`
- [ ] navtest / navhard 列表存 `data/splits/navtest.pkl` / `navhard.pkl`

### M0.3 NAVSIM pipeline
- [ ] AutoVLA 接入 NAVSIM submission pipeline（参考 prior-work 的 submission generation 脚本）
- [ ] EPDMS evaluator 单元测试
- [ ] 子项分数（collision / comfort / progress）抽取脚本（Q5.a 主表用）

### M0.4 Baseline 全量跑
- [ ] navtrain 全量 r=1.0 inference → per-scene EPDMS + 4 子项 + FLOPs/latency
- [ ] 存到 `results/baseline_r1.0_navtrain.pkl`（key=scene_id, val=dict）
- [ ] navtest r=1.0 → `results/baseline_r1.0_navtest.pkl`
- [ ] navhard r=1.0 → `results/baseline_r1.0_navhard.pkl`

## M0 验收
- [ ] EPDMS_baseline (navtest) 数值；预期 0.45–0.65
- [ ] **止损线**：navtest EPDMS < prior-work ceiling (≈0.41) → 重评估骨干
- [ ] 单 token 推理时延 + token 数（FLOPs reference）记录在 `docs/journal/M0_baseline.md`

## M0 产物
| 产物 | 路径 |
|---|---|
| Splits | `data/splits/{probe_100, train, val, navtest, navhard}.pkl` |
| Baseline EPDMS | `results/baseline_r1.0_{navtrain, navtest, navhard}.pkl` |
| Inference timing | `docs/journal/M0_baseline.md` |

---

# M1 — Stage A: Importance Scorer SFT (Q4.1)

**目标**：训出一个 frozen MLP scorer，给定 (token_feat, cam_id, scene_ctx) 输出 importance score，与 LLM attention ranking 对齐

**对应决策**：Q3 (Stage A)、Q4.1 (attention distill via LambdaRank)

## M1.a — Layer probing（找最优 attention layer）

- [ ] 在 100-scene probe (M0.2) 上跑 AutoVLA r=1.0 inference，dump 每层 LLM attention（query=last text token, key=vision tokens）
- [ ] 对每个候选 layer ∈ {3, 6, 9, 12, ...}：
  - 用 attention 排序，选 top-0.5N
  - 跑 NAVSIM 得 EPDMS_layer
- [ ] 选 EPDMS 最高的 layer L*
- [ ] **同时是 Q5 ablation A5 的数据源**

**验收**：layer EPDMS 表 + 选定 L* + 写入 `docs/journal/M1a_layer_probing.md`
**算力**：~0.05× navtrain（100 scene × N layers）
**风险**：见 risks.md R-M1a

## M1.b — 全量 navtrain attention 提取

- [ ] 在 navtrain \ probe（即 M0.2 train pool）全量 inference，只 dump 选定 layer L* 的 attention
- [ ] 每 token 算 attention score → ranking label
- [ ] 存 `results/attention_ranking_label.pkl`（key=scene_id × token_id, val=score）

**验收**：label 文件存在，sanity check 抽样 5 scene 看 ranking 是否合理（前景 token 排名高）
**算力**：1× navtrain inference（与 M0.4 可合并跑，省 1×）

## M1.c — LambdaRank SFT

- [ ] Scorer 架构：MLP(token_feat ⊕ cam_id_emb ⊕ scene_ctx_emb) → 1-d score
- [ ] Loss: LambdaRank（与 attention ranking 对齐，pairwise）
- [ ] 训练：90/10 train/val（Q4.4.c）
- [ ] Early stop on val NDCG
- [ ] 输出 `ckpt/scorer_v1.pt`（**frozen 待用**）

**验收**：
- val NDCG@k (k = 0.5N) ≥ 0.X（threshold 待 M1c 实测后定）
- iso-ratio 0.5 下，用 scorer_v1 选 top-0.5N 跑 100-scene probe → EPDMS ≥ M1a 选定 layer 的 EPDMS

**算力**：极小（MLP + 离线 attention label）

## M1 产物
| 产物 | 路径 |
|---|---|
| Layer probing report | `docs/journal/M1a_layer_probing.md` |
| Attention ranking label | `results/attention_ranking_label.pkl` |
| Scorer v1 ckpt | `ckpt/scorer_v1.pt` |
| Scorer eval | `results/scorer_v1_iso0.5_probe100.pkl` |

---

# M2 — Stage A: Importance Scorer RL (Q4.3.a)

**目标**：在 SFT scorer_v1 基础上做 GRPO，让 scorer 在 "given any B, pick best top-B" 上更强

**对应决策**：Q3 (Stage A 第二阶段)、Q4.3.a (R4 pure EPDMS-advantage)、Q4.3.c (per-scene baseline)、Q4.3.e (clip)

## M2 任务

- [ ] GRPO trainer 接入（TRL ≥ 0.15.0 或自实现）
- [ ] Rollout 循环：
  ```
  for each scene in train pool:
    sample B ∈ {0.25N, 0.5N, 0.75N, 1.0N}（uniform）
    for k = 1..K rollouts:
      score_k ~ scorer(...)（stochastic）
      top-B by score_k → AutoVLA inference → EPDMS_k
      R_k = clip(EPDMS_k − EPDMS_baseline, -1, 1)
    GRPO advantage update
  ```
- [ ] 监控指标：avg R, val EPDMS@iso-0.5, scorer divergence from v1（KL）
- [ ] Early stop on val EPDMS plateau
- [ ] 输出 `ckpt/scorer_v2.pt`（**frozen，从此 Stage B 用**）

## M2 验收
- [ ] iso-ratio 0.5 在 val set 上 EPDMS(scorer_v2) ≥ EPDMS(scorer_v1)
- [ ] 训练曲线稳定（无 reward collapse）
- [ ] 抽样 case study：scorer_v2 选的 token 在视觉上比 v1 更合理

**算力**：~K × train pool inference（K=4–8）。**这是 pipeline 第二大算力开销**
**风险**：见 risks.md R-M2

## M2 产物
| 产物 | 路径 |
|---|---|
| Scorer v2 ckpt | `ckpt/scorer_v2.pt` |
| RL training log | `logs/M2_stageA_rl/` |
| Val EPDMS curve | `results/M2_val_curve.pkl` |

---

# M3 — Stage B Oracle Generation (Q4.2)

**目标**：用 frozen scorer_v2 跑 r ∈ {0.25, 0.5, 0.75} 的 oracle EPDMS，按 B3 Pareto-aware 规则造 budget label

**对应决策**：Q4.2.a (B3)、Q4.2.b (ε scan)

## M3.a — Oracle inference

- [ ] 对 navtrain \ probe，用 scorer_v2 选 top-{0.25N, 0.5N, 0.75N} 跑 AutoVLA → EPDMS
- [ ] 每 scene 一组 (EPDMS_0.25, EPDMS_0.5, EPDMS_0.75)
- [ ] r=1.0 复用 M0.4 baseline
- [ ] 存 `results/oracle_4r_navtrain.pkl`

**算力**：3× navtrain inference

## M3.b — ε scan on probe-100 (Q4.2.b)

- [ ] 用 probe-100 子集
- [ ] 对 ε ∈ {0.005, 0.01, 0.02, 0.05}：
  - 对每 scene：max_E = max over 4 r；label B* = min{r: E_r ≥ max_E − ε}
  - 统计 4 类 label 分布
- [ ] 选**分布最均衡的 ε** 作为 ε*（Q5 ablation A6 数据源）
- [ ] 出 figure：label distribution vs ε

**验收**：ε* 选定，label 分布在 4 类不全 collapse 到 1 类

## M3.c — 全量 label 生成

- [ ] 用 ε* 在 navtrain \ probe 全量造 budget label
- [ ] 存 `results/budget_label_eps{ε*}.pkl`

## M3 产物
| 产物 | 路径 |
|---|---|
| Oracle EPDMS table | `results/oracle_4r_navtrain.pkl` |
| ε scan report | `docs/journal/M3b_eps_scan.md` |
| Budget label | `results/budget_label_eps{ε*}.pkl` |

---

# M4 — Stage B: Budget Policy SFT (Q4.2)

**目标**：训 4-class budget classifier，输入 scene context，输出 budget ∈ {0.25, 0.5, 0.75, 1.0}

**对应决策**：Q4.2、Q3 (Stage B 第一阶段)

## M4 任务

- [ ] Budget Policy 架构：MLP(scene_ctx_emb) → 4-class logits
  - scene_ctx = nav_cmd_emb ⊕ ego_speed ⊕ driving_instr_emb（与 M2/M5 一致）
- [ ] Loss: 4-class CE（必要时 class-weighted 应对不均衡）
- [ ] 训练 90/10 train/val（共用 Q4.4 split）
- [ ] Early stop on val acc
- [ ] 输出 `ckpt/budget_policy_v1.pt`

## M4 验收
- [ ] val 4-class acc > 50%（4 类基线 25%）
- [ ] 4 类 logit 分布合理，不全 collapse 到主导类

**算力**：极小

## M4 产物
| 产物 | 路径 |
|---|---|
| Budget policy v1 | `ckpt/budget_policy_v1.pt` |

---

# M5 — Stage B: Budget Policy RL (Q4.3.b)

**目标**：在 budget_policy_v1 基础上 GRPO，用 R3 piecewise Pareto reward

**对应决策**：Q4.3.b (R3)、Q4.3.c (per-scene baseline)、Q4.3.d (α/β scan)、Q4.3.e (clip)

## M5.a — α/β grid scan on probe-100 (Q4.3.d)

- [ ] 对 (α, β) ∈ {0.05, 0.1, 0.2, 0.5}² grid（16 组）
- [ ] 每组在 probe-100 上跑短 RL（少量 iter），看：
  - val ΔEPDMS
  - avg kept ratio
- [ ] 选 (α*, β*) 使得在保 ΔEPDMS ≥ -ε* 的前提下 kept ratio 最低（Q5 ablation A7 数据源）

**算力**：16 × probe-100 短 RL，可控

## M5.b — 全量 RL

- [ ] 用 (α*, β*, ε*) 在 train pool 跑 GRPO
- [ ] Rollout：
  ```
  freeze scorer_v2
  for each scene:
    for k = 1..K rollouts:
      B ~ budget_policy(scene_ctx)（4-class sample）
      top-B by scorer_v2 → AutoVLA → EPDMS_k
      ΔE = EPDMS_k − EPDMS_baseline
      if ΔE >= -ε*:
        R_k = ΔE + α*·(1 - B/N)
      else:
        R_k = ΔE − β*·|ΔE + ε*|
      R_k = clip(R_k, -1, 1)
    GRPO advantage update
  ```
- [ ] 监控：avg R, val EPDMS, val kept_ratio
- [ ] 输出 `ckpt/budget_policy_v2.pt`（**final main method**）

## M5 验收
- [ ] val EPDMS ≥ M4 SFT 版本
- [ ] val avg kept_ratio ≤ 0.5（Q1 A 主轴 success criterion）
- [ ] val ΔEPDMS ≥ -0.5 EPDMS 点（Q1 B 底线）

**算力**：~K × train pool inference

## M5 产物
| 产物 | 路径 |
|---|---|
| α/β scan report | `docs/journal/M5a_alpha_beta_scan.md` |
| Budget policy v2 | `ckpt/budget_policy_v2.pt` |
| RL training log | `logs/M5_stageB_rl/` |

---

# M6 — Evaluation & Ablations (Q5)

**目标**：在 navtest 跑全部 baselines + ablations，生成 paper main table 和 figure

**对应决策**：Q5 整章

## M6.a — Main table (navtest, ratio=0.5 column)

| # | Baseline | 实施路径 |
|---|---|---|
| 1 | AutoVLA r=1.0 | M0.4 复用 |
| 2 | Random pruning r=0.5 | 简单实现，跑 navtest |
| 3 | FastV (internal) at matched FLOPs | 复用本地已跑通 FastV pipeline |
| 4 | FastV-selector-at-input r=0.5 | 提 FastV layer-2 attention，搬到 ViT 出口当 selector |
| 5 | Ours Fixed-Budget (scorer_v2 + r=0.5 fixed) | 跳过 budget_policy，固定 B=0.5N |
| 6 | **Ours RL-Drive Full** (scorer_v2 + budget_policy_v2) | 主方法 |
| 7 | Fixed-ratio + LLM raw attention r=0.5 | 用 layer L* 的 attention 直接 rank |

**主表列**：EPDMS / collision / comfort / progress / kept_ratio / FLOPs / latency

## M6.b — Pareto curve figure

- [ ] 对 baseline 1, 2, 3, 4, 7 + Ours 6（多个 budget mix）扫多个 ratio
- [ ] 画 EPDMS vs FLOPs 散点 + 凸包
- [ ] **Ours 必须严格 dominate baseline 1–4, 7**

## M6.c — Ablations A1–A10

| # | Ablation | 实施 |
|---|---|---|
| A1 | w/o RL (Ours SFT only = scorer_v1 + budget_policy_v1) | 直接评测，不跑新东西 |
| A2 | w/o SFT (B5: budget policy uniform init + GRPO directly) | 单独跑一次 |
| A3 | w/o Budget Policy (= baseline 5) | 复用 |
| A4 | w/o Importance Scorer (= baseline 2) | 复用 |
| A5 | Attention layer choice | 复用 M1.a 数据 |
| A6 | ε_quality sensitivity | 复用 M3.b 数据 |
| A7 | α/β reward sensitivity | 复用 M5.a 数据 |
| A8 | EPDMS sub-component breakdown | 主表自带 |
| A9 | navhard performance | 跑全部 7 baseline 在 navhard 上 |
| A10 | FastV-selector-at-input (= baseline 4) | 复用 |

**只有 A2 (B5) 和 A9 (navhard) 需要新跑**，其他 8 个全部复用前面 milestone 数据。

## M6 产物
| 产物 | 路径 |
|---|---|
| Main table CSV | `results/M6_main_table.csv` |
| Pareto figure | `results/M6_pareto.pdf` |
| Ablation tables | `results/M6_ablation_{A1..A10}.csv` |
| navhard results | `results/M6_navhard.csv` |

## M6 验收
- [ ] Ours RL-Drive Full 在 navtest ratio=0.5 列 EPDMS 击败所有其他 baseline
- [ ] Pareto curve Ours dominate
- [ ] navhard ratio=0.5 EPDMS ≥ baseline ratio=0.5 EPDMS

---

## 时间预算（按 design freeze 后重估）

> 假设：双 H20 可用，inference 速度参考 prior-work 经验

| Milestone | 工作日 | 备注 |
|---|---|---|
| M0 | 2 | infra 装包 + baseline 全量跑 |
| M1 | 2 | layer probing 1 天 + SFT 1 天 |
| M2 | 2–3 | Stage A RL 调参敏感 |
| M3 | 1–2 | 3× inference 是大头 |
| M4 | 0.5 | 4-class classifier 极快 |
| M5 | 2–3 | Stage B RL + α/β scan |
| M6 | 2 | 跑 baselines + 出图表 |
| **合计** | **12–15 工作日** | |

按今天 2026-06-14 起算，乐观 6-26、保守 6-30 出齐主表，与 paper 投稿 deadline 匹配视情况调。

---

## 关键依赖与解锁顺序

```
M0 ──> M1a ──> M1b ──> M1c ──> M2 ──> M3 ──> M4 ──> M5 ──> M6
        ↓
       (Q5 A5 数据)
```

**critical path**：M0 → M1c → M2 → M3 → M5 → M6
**可并行**：
- M3.a 跑的同时 M4 可以先架代码
- M5.a (α/β scan) 跑的同时 M6 baseline 2/3/4/7 可以先跑
- M6 ablations 大部分是数据复用

---

## 与 design_decisions.md 的反向追溯表

| Decision | 实施于 | 验证于 |
|---|---|---|
| Q1 A 主轴 (kept_ratio ≤ 0.5) | M5 reward + M6 main table | M5 验收 + M6.a |
| Q1 B 底线 (ΔEPDMS ≥ -0.5) | M5 reward 的 ε_quality | M5 验收 |
| Q2 vision-only @ ViT-LLM 接口 | M0.1 backbone 改造 | M0.4 sanity (r=1.0 lossless) |
| Q3 two-stage separate | M1–M2 vs M4–M5 | M2/M5 各自验收 |
| Q4.1 attention distill | M1a + M1b + M1c | M1c val NDCG |
| Q4.2 B3 Pareto oracle | M3.a + M3.b + M3.c | M3.b ε 分布 |
| Q4.2.c B5 ablation | M6 A2 | M6.c |
| Q4.3.a Stage A reward | M2 | M2 验收 |
| Q4.3.b Stage B reward (R3) | M5.b | M5 验收 |
| Q4.3.d α/β scan | M5.a | M5.a report |
| Q4.4 数据切分 | M0.2 | M0 验收 |
| Q5 7 baselines | M6.a | M6.a 主表 |
| Q5 A1–A10 ablations | M6.c | M6.c |
