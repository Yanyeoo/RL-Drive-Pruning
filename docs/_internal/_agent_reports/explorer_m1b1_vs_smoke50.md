# Explorer Report: M1.b₁ V1/V2/V3 + rank-variance ↔ M1.b₂ navtrain smoke 50

> Agent: explorer (archaeology)
> Date: 2026-06-26
> Sources: docs/_internal/m1b2_v4_spec_2026-06-25.md, docs/_internal/m1b2_phase2_design_2026-06-25.md,
>          exp/m1b2_rank_variance/rank_variance.json (n=19,225 navtrain), 
>          /tmp/smoke50_R1pp/dataset_R1pp_target12_botK4.summary.json (n=50 navtrain smoke),
>          docs/specs/m1b_freelunch_spec.md, docs/results/key_results.md.

## TL;DR

1. **L12:h13 在 navtrain 全集 19,225 上 bot-K freq = 100.00%**，rank_mean=0.0 / rank_std=0.0（来自 rank_variance.json `m1b1_mask_probe.V1`）。smoke 50 上 h13=1.00 不是小样本巧合，是**结构性事实**，与 M1.b₁ V1 (`L12:{h13}`) 设计原假设完全一致。
2. **smoke 50 的其余 5 个活跃 head（h14, h6, h0, h2, h4）的频次需要在全集上重新核对**：rank_variance.json 只 dump 了 m1b₁ mask 涉及的 L12/L24/L27 head，**没有 L12 全 16 head 的全集 bot-K freq 表**。这是关键缺口 — smoke 50 的"h14=0.90 / h6=0.74 / h0=0.50"在 19,225 上是否成立**无法直接由现有产出验证**，必须等 dataset full build 完成才能确认。
3. **V4 spec 已经 FINAL**（2026-06-26 11:15），mask = `{12: [13], 24: [7, 9, 10]}`，C1 V0 dryrun 已在 navtrain_probe100 上跑完（PDMS=0.7883，作为 V0 floor）。这意味着 Phase 2 v0 不是孤立工作，而是 **V4 的"学习版上位替代候选"** —— 如果 P1/P2 在 v0 跑通，下一步走 RL；如果跑不通，V4 是 fallback。
4. **rank-variance analysis 证实了 V3 brittleness 假设**：L24 的 11 head 里只有 3 个（h7=93.67%, h9=99.98%, h10=98.92%）是真正稳定 bot，其余 8 个 freq < 60%。这与 navtest V3 PDMS=0.8537 的 −4.4 pp cliff 自洽。
5. **navtrain ↔ navtest distribution shift**：rank-variance.json 的全集结果与 navtest M1.b₁ V1 free-lunch (Δ=−0.0004) **指向同一个稳定 head**，目前**没有 distribution drift 的强证据**。但 L12 整体 16-head freq 表只在 navtrain 跑过，navtest 端没有同口径产出。

---

## 1. M1.b₁ V1 / V2 / V3 速览

来源：`docs/specs/m1b_freelunch_spec.md` §3 + `docs/results/key_results.md` §6.1（部分通过 explorer subagent 提取）+ V4 spec §2.3 行 64。

### V1 minimal — `{12: [13]}`
- Heads removed: 1 (L12:h13)
- KV saving: 0.39%（1/256 head-slots）
- navtest PDMS = **0.8981** vs V0=0.8985, B0=0.8983
- Δ vs B0 = **−0.0002**（在 ±0.001 free-lunch acceptance 内）
- Verdict: **free-lunch ✅**

### V2 moderate — `{12: [13], 27: [0, 8, 9]}`
- Heads removed: 4
- navtest PDMS = **0.8545**
- Δ vs B0 = **−0.0438** —— cliff
- Mechanism (V4 spec §1)：L27 是模型 readout layer 之一，mask 后破坏最终 token 预测
- Verdict: **fail acceptance ❌**

### V3 moderate-large — `{12: [13], 24: [0,1,2,6,7,8,9,10,12,14,15], 27: [0,8,9]}`
- Heads removed: 15
- navtest PDMS = **0.8537**
- Δ vs B0 = **−0.0446** —— cliff（与 V2 几乎一样）
- 关键发现（V4 spec §1）：V3 的 L24 mask 只有 3 个 head 是真正稳定 bot，其余 8 个是"compensation by redundancy"（mask 它们但模型自动 reroute 到其他 head）
- Verdict: **fail acceptance ❌, 但启发了 V4**

### V4 (FINAL 2026-06-26 11:15) — `{12: [13], 24: [7, 9, 10]}`
- Heads removed: 4（与 V2 同数量，不同位置）
- 来源：rank-variance analysis 在 19,225 navtrain 上识别出 L24 的"真正稳定 dead heads"
- 验收 gate：
  - G1: V4 navtest PDMS within 0.1 pp of V1 → claim "L24 add-on 在 navtest 不亏"
  - G2: V4 navtrain PDMS > V3 navtrain by ≥ 0.3 pp → claim "rank-variance pruning 比 blanket V3 mask 强"
- 状态：**FINAL but not yet executed**（C1 V0 dryrun PDMS=0.7883 只是 V0 floor，不是 V4 数）

---

## 2. rank-variance analysis 速览

来源：`exp/m1b2_rank_variance/rank_variance.json`（n=19,225 navtrain, K=4, L=28, H=16, wall=114s）

### 2.1 V1 head（L12:h13）— 100% 稳定

```
layer=12, head=13:
  rank_mean=0.00, rank_std=0.00, bot_k_freq=1.00
  score_mean=2.67e-7, score_std=1.33e-7
```

**这是 navtrain 全集 19,225 个 token 的精确数字**。每个 scene h13 都是 L12 的 rank-0（最弱 head），不是 small-n smoke artifact。

### 2.2 V2 add-on heads（L27）— 全部稳定 bot

| layer:head | rank_mean | rank_std | bot_k_freq |
|---|---:|---:|---:|
| L27:h0 | 1.997 | 0.059 | **1.000** |
| L27:h8 | 0.000 | 0.000 | **1.000** |
| L27:h9 | 1.003 | 0.056 | **1.000** |

L27 的三个 head 在 navtrain 上 **100% 稳定 bot-4**，可是 navtest V2 PDMS 还是 cliff 了 −4.4pp —— 这说明 **"navtrain bot-4 freq 高" ≠ "mask 不损失 navtest PDMS"**。是 V4 spec §1 提到的"L27 readout layer dependency"问题，而非 distribution shift。

### 2.3 V3 add-on heads（L24）— 只有 3/11 稳定

| layer:head | rank_mean | rank_std | bot_k_freq |
|---|---:|---:|---:|
| L24:h7  | 2.16 | 0.83 | **0.937** |
| L24:h9  | 0.41 | 0.52 | **0.9998** |
| L24:h10 | 0.72 | 0.71 | **0.989** |
| L24:h0  | 3.91 | 1.39 | 0.535 (borderline) |
| L24:h6  | 4.98 | 1.39 | 0.130 |
| L24:h12 | 5.06 | 1.47 | 0.166 |
| L24:h15 | 5.11 | 1.57 | 0.198 |
| L24:h14 | 5.93 | 1.36 | 0.045 |
| L24:h1  | 7.93 | 0.81 | 0.0003 |
| L24:h2  | 9.83 | 0.46 | 0.000 |
| L24:h8  | 8.96 | 0.53 | 0.000 |

**只有 h7/h9/h10 是真正稳定 bot**，V4 mask 只取这 3 个。V3 的另外 8 head 平均 freq 远 < 50%，强行 mask 它们是损失而非节省。

### 2.4 全局 layer entropy

`global.per_layer_entropy_bits` 显示 L12 entropy = **1.765 bits**（max=4），中等熵 — 意味着 L12 bot-4 head 集合**有但只有有限的 per-scene 变化**。L21 entropy=2.08（最高），L1/L2 entropy=0.0（完全稳定）。

**对 v0 的含义**：L12 entropy=1.77 bits 大致说明 per-scene bot-4 set 在 ~ 2^1.77 ≈ 3.4 个不同集合之间漂移。如果 v0 P1/P2 学到的 per-scene 信号上限 ≤ 1.77 bits，对应 macro-F1 提升空间是有限的。

---

## 3. navtrain smoke 50 ↔ navtest M1.b₁ 横向对照

### 3.1 smoke 50 的 L12 bot-4 per-head freq

```
h13: 1.0000   h14: 0.9000   h06: 0.7400
h00: 0.5000   h02: 0.4600   h04: 0.4000
其余 10 个 head: 0.0000
```

### 3.2 navtest M1.b₁ 端的 L12 head 排名

来源：`docs/_internal/m1b_per_head_analysis_2026-06-18.md`（M1.b₁ navtest n=100 token 的 L12 mean attention）—— 通过 explorer subagent 报告间接引用。

navtest 端 L12 mean-attn 的"末尾 6 head"（按从小到大排）= **h13, h14, h0, h6, h4, h2**

navtrain smoke 50 端"非零 freq 6 head"（按 freq 从大到小）= **h13, h14, h6, h0, h2, h4**

### 3.3 对照表

| head | navtest M1.b₁ 末位排名 | navtrain smoke 50 freq | navtrain rank-variance 全集 |
|---:|---:|---:|---|
| h13 | 1st (lowest) | 1.00 | **bot-K freq = 1.00 (n=19,225)** ✅ |
| h14 | 2nd | 0.90 | 未单独 dump（不在 m1b1 mask 集） |
| h06 | 4th | 0.74 | 未单独 dump |
| h00 | 3rd | 0.50 | 未单独 dump |
| h02 | 6th | 0.46 | 未单独 dump |
| h04 | 5th | 0.40 | 未单独 dump |

**关键观察**：
- ✅ navtest 末位 6 head 与 navtrain smoke 50 活跃 6 head 是**同一组 head id**
- ✅ navtest L12:h13 排第 1（最弱）⇆ navtrain h13 freq=1.00（每个 scene 都最弱）—— 完全自洽
- ⚠ smoke 50 的频率数（h14=0.90 等）未被 19,225 全集验证，需要 full dataset build 才能确认
- ⚠ navtest 端没有 19,225 同口径全集 dump，目前**严格的 navtrain ↔ navtest L12 全 head freq 对照不存在**

## 4. distribution shift 信号

**结论**：目前没有 navtrain ↔ navtest L12 的 distribution shift 强证据。

证据：
1. L12:h13 在 navtest M1.b₁ free-lunch 验证（PDMS Δ=−0.0002）+ navtrain rank-variance bot_k_freq=1.00 → 同一个 head 在两端都最弱
2. navtest 末位 6 head 集合 = navtrain smoke 50 活跃 6 head 集合（顺序略有差异但成员一致）
3. global per_layer_entropy 在 L12 = 1.77 bits 与 V1 free-lunch 行为自洽（足够变化造成 L12 不全是 dead，但 h13 永远在 bot）

**未验证的潜在风险**：
- navtrain L12 entropy=1.77 是否在 navtest 端也成立？没有数据。
- smoke 50 的 h14=0.90 在全集是否可能塌到 1.0（让 v0 退化为 2-of-14）？需要 full build 验证（critic Q6 也提了同样问题）。
- L24:h7/h9/h10 在 navtrain 上 99% bot，但 navtest 端 V4 还没跑完，**真正的跨集合 transferability 还是开放问题**（V4 acceptance gate G1/G2 就是为此设计）。

## 5. 文档/结果不一致项 (drift)

| # | 位置 | 问题 |
|---|---|---|
| 1 | conversation summary 里说"V4 spec 待写" | 实际 `docs/_internal/m1b2_v4_spec_2026-06-25.md` 已 FINAL（2026-06-26 11:15） |
| 2 | conversation summary 里说"navtest dump 不存在 → G_v0_3 → G_v0_3'" | 这个判断仍然成立（rank-variance.json 是 navtrain only），但 V4 spec 隐含了一个未来 navtest sweep 计划，在 V4 pdms 跑完后 navtest 端会有同口径数 |
| 3 | design doc §10.z const top-4 写 `{h13, h14, h6, h2}` | dataset 代码动态取，smoke 50 实际是 `[13, 14, 6, 0]`（h0 而非 h2）。critic Q5 已点出 |
| 4 | rank_variance.json 只覆盖 m1b1 mask 涉及的 head（L12:h13, L24:11个, L27:3个）| L12 其他 15 head 没有在全集 freq 表里，**关键缺口** |
| 5 | global.per_layer_entropy_bits 包含全 28 层信息 | 但 design doc §10.z 没引用这个结果做 v0 task ceiling 估算 |

## 6. v0 路线 ↔ V4 关系（额外发现）

V4 spec §7 Q1 自问："Is V4 worth running before Phase 2 learned policy?" —— **明确把 v0 P1/P2 视为 V4 的潜在上位替代**。这意味着：

- 如果 v0 P1/P2 在 G_v0_1 通过 → 走 RL，V4 可以延后或跳过
- 如果 v0 P1/P2 失败 → V4 是 fallback baseline（已 FINAL，可立即跑）
- 但目前**没有人决定 V4 是否在 v0 跑通前就提前跑**。这是个调度决策，不是技术决策

## 附录：引用的关键文件路径

- `docs/_internal/m1b2_phase2_design_2026-06-25.md` (§10, §10.y, §10.z)
- `docs/_internal/m1b2_v4_spec_2026-06-25.md` (FINAL)
- `docs/specs/m1b_freelunch_spec.md` (V0/V1/V2/V3 baseline mask 定义)
- `docs/results/key_results.md` (§6.1 navtest PDMS)
- `exp/m1b2_rank_variance/rank_variance.json` (n=19,225 全集 rank stats)
- `exp/m1b2_rank_variance/SUMMARY.md`（rank-variance 配套 summary）
- `/tmp/smoke50_R1pp/dataset_R1pp_target12_botK4.summary.json`（smoke 50 build 输出）
- `scripts/m1b2_phase2_v0_build_dataset.py`（dataset build 实现）
