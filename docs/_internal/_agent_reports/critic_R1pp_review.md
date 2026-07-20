# Critic Review: R1'' Design + Smoke 50

> Agent: critic (red-team)
> Date: 2026-06-26
> Inputs read: docs/_internal/m1b2_phase2_design_2026-06-25.md §10.y/z, scripts/m1b2_phase2_v0_build_dataset.py, /tmp/smoke50_R1pp/dataset_R1pp_target12_botK4.summary.json + build_dataset.log

## TL;DR

1. **h13=1.00 暴露了 v0 的有效任务维度其实接近 "3-of-15"**：const baseline macro-F1=0.2165 / EM=0.26 是在一个被先验吃掉的标签空间里测出来的，G_v0_1 阈值 (B0 + 0.05 macro-F1) 站不住，需要按 "h13-masked macro-F1" 重新校准。
2. **closed-form macro-F1=0 是"实现方式造成的零"，不是"feature 与 label 解耦"的强证据**：`bot-K(rank(x_concat)) % 16` 对 R^96→R^16 的折叠是个有偏的弱诊断，不能由它推出 R1'' 任务非平凡。
3. **R1'' 排除 L11/L12/L13 不够**：L8/L16 通过残差流与 L12 强耦合，必须加 "only far layers" 控制组。
4. **50 → 19,225 的统计风险高**：h14=0.90 的 50-样本 95% Wilson CI ≈ [0.79, 0.96]，全集上若 head 频次进一步集中（h13/h14 双双 ≈1.0，其余 ≪ 0.5），任务退化到 const baseline 几乎已是全局最优，留给 learner 的 headroom 接近 0。
5. **设计-代码漂移**：design doc §10.z 的 const top-4 写作 `{h13, h14, h6, h2}`，代码实际取 `[13, 14, 6, 0]`（h0 而不是 h2）。Smoke 上 h2=0.46 / h0=0.50 的次序差异决定了哪个 head 进入 baseline，全集上很可能再变。

**推荐动作**：在跑 P1/P2 之前，先用 19,225 全集做一次 head-freq 复算 + "h13-masked" 评测口径决定 + far-layer-only 控制实验。

---

## Q1. h13=1.00 的含义

50/50 全部 scene 中 h13 都进 bot-4。这不是统计噪声，是 M1.b₁ V1 早就证明的结构事实（design doc §3.4 明确写 "V1 (L12:h13) is provably 100% bot-K on 19,225 scenes"）。问题严重程度：

- **任务有效维度从 4-of-16 退化到 3-of-15**。h13 占死一格，剩下 3 个 bot 名额从 15 个 head 里选。C(15,3)=455 vs C(16,4)=1820，random EM 从 5.5e-4 升到 2.2e-3，但**任何模型预测 h13 是免费 +1/4 的 recall**。
- **macro-F1 把 h13 当成 16 个 head 之一平均**，h13 的 F1 ≈ 1.0 直接给整个 macro-F1 注入一个 0.0625 的"白送"。const baseline macro-F1=0.2165 里至少有 0.06–0.10 来自 h13 的 trivial perfect。
- **后果**：P1/P2 只要学会"恒输出 h13"就拿到这 0.0625。G_v0_1 的 "+0.05 over B0" 阈值落在 0.265，**P1 完全可能靠"h13 + B0 const 的另外三 head"就达标，而不需要任何 per-scene 推理**。

**建议**：评测时 mask 掉 h13 这一维（design doc 已识别 h13 为 V1 的 static optimum，本应不参与"per-scene gating"的可学习部分）。报告 **h13-masked macro-F1**（在 15 个非-h13 head 上）作为主要指标，原 16-head macro-F1 仅作 reference。loss 端是否要 mask 取决于训练目标——我倾向于保留（让模型学到 h13 始终在 bot 是 free signal），但**评测必须 mask**，否则 G_v0_1 是被 inflated 的。

## Q2. R1'' 真的避开了泄漏吗

排除 L11/L12/L13 是必要不充分。28 层 transformer 残差流在结构上让任意层的 attention pattern 都受附近层影响，**L8 → L12 之间隔 3 个 block，这在 7B 类模型里仍然是高度耦合的**——而 FEATURE_LAYERS 里 L8 / L16 都在 L12 的 ±4 半径内。

具体担忧：

1. **L16 是 L12 的"下游 readout"**：L12 把信息写进残差流后，L13–L15 还在继续读写，到 L16 时 L12 的 attention 决定哪些 vision token 被 amplify 仍然部分可恢复。`attn[16].mean(-1)` 携带 L12 bot-K head 的影子。
2. **L8 是 L12 的"上游 setup"**：注意力路由（哪些 head 关心哪类 vision token）通常在 mid-layer 已经分化，L8 的 head pattern 是 L12 head 行为的强 prior，二者**对同一类场景共变**。
3. R1'' 的"非平凡"如果只是 P1/P2 学到了"L8 / L16 的 mean-attn 与 L12 mean-attn 强相关 → 把 L8/L16 的 bot 直接复制到 L12"，那它在 design doc §10.z 的科研问题（"per-scene bot-K 是否跨层可预测"）里得到的是**最弱版本的 yes**，离"R1''-feature RL 在 v1 可用"还很远。

**建议补 control**（不写代码，仅描述）：
- **C-far**：FEATURE_LAYERS = (0, 4) only（24 维）。如果 P2 在 C-far 上 macro-F1 ≈ const baseline，而在原 R1'' (96 维) 上显著高，说明信号主要来自 L8/L16 的近邻泄漏。
- **C-randperm**：把 X 在 N 维度上 random permute（破坏 scene-level 配对，保留 marginal 分布），重训 P2。如果 macro-F1 不掉到 const baseline，说明 learner 在记忆 marginal，不是学 per-scene。
- **C-pure-static**：只用 head freq 全局向量（16 维 const 输入）训一个 P2，作为"learner 能在零 per-scene 信号下做到多好"的上界。

## Q3. closed-form macro-F1=0 是否过强信号

**不是**。它过强到反而可疑。两层问题：

**(a) 算法本身有偏**。代码里的 closed-form 是 `bot_x_idx % 16`：取 X (R^96) 中最小的 4 个维度，对 16 取模映射回 head id。这意味着如果 4 个最小值集中在某一层（比如全在 L0 的 16 维内），它会输出该层的 bot-4 head ids；如果 4 个最小值跨层分布在不同 head id 上，模 16 后会输出"最'冷'的 4 个 head 维度"，**但这个映射对'同一 head 在不同层都冷'的场景退化到只输出 1–2 个 unique id**——结果就是 pred 只覆盖 1–2 个 head，pred 和 y={h13, h14, h6, X} 之间几乎不重叠，**F1=0 是被映射方式塑造出来的，不是 feature ⊥ label 的强证据**。

**(b) R1' (100%) → R1'' (0%) 的跳变本身可疑**。在 R1' 下 closed-form 是 100% 因为 y = bot-K(x) by construction；在 R1'' 下立刻塌到 0% 也太干净。一个可学的中间问题应该让 closed-form 给出非零的弱基线（比如 0.1–0.3 macro-F1）。0.0 + EM 0.0 这个组合**更像是评测口径退化**（参见 (a)），而不是"feature 完全无信号"。

**建议 sanity**（描述）：
1. **per-layer closed-form**：对每个 L ∈ {0, 4, 8, 16, 20, 24}，单独算 `bot-4(rank(attn[L].mean(-1)))` 在 L12 上的 macro-F1。每个是真正的 16→16 映射，不需要 % 16。如果 L8/L16 的 per-layer closed-form macro-F1 显著 > 0（比如 > 0.3），就是"非平凡可学"的强证据；如果全部 6 个 layer 都 ≈ 0.05–0.15，那 R1'' 是**弱可学甚至不可学**，G_v0_1 (+0.05 over B0) 几乎不可能达标。
2. **best-single-layer 上界**：6 个 per-layer closed-form 取 max。这是"不学 cross-layer combination，只挑一层最好"的天花板，给 P1/P2 设了一个"必须超过"的有意义参考。

## Q4. B0 const baseline EM=0.26 的解读

K=4 multi-hot 在 16 维空间，**uniform random** baseline 的 EM = 1/C(16,4) = 1/1820 ≈ 5.5e-4。const baseline 的 EM 上界由"const top-4 与 ground-truth 4 集合 exact 重合的 scene 占比"给出，在 50 样本上 = 0.26（13/50）。这数字本身**完全由 h13=1.0、h14=0.9、h6=0.74、h0=0.5 这四个频率的组合决定**：const top-4 = {13,14,6,0}，要 EM 命中需要这 4 个 head 同时在 bot-4 → 概率约等于 0.5（h0 是瓶颈）× 0.74 × 0.9 × 1.0 ≈ 0.33（独立性假设下）。实测 0.26 在小样本上和这个粗估一致。

**P1/P2 必须超过的有意义阈值**：

| 指标 | const B0 | random | 我建议的 G_v0_1 |
|---|---|---|---|
| macro-F1 (16-head) | 0.2165 | 0.25\*4/16=0.0625 | B0 + 0.05 = **0.27**（design doc 当前值）|
| **macro-F1 (h13-masked, 15-head)** | ~0.16 | ~0.06 | **0.21**（B0' + 0.05）|
| EM | 0.26 | 5.5e-4 | B0 + 0.10 = **0.36** |
| **per-active-head F1**（仅 h00/h02/h04/h06/h14）| ~0.30 | ~0.25 | **0.40** |

**核心问题**：design doc 当前的 G_v0_1 = "P1 holdout per-head F1 > B0 + 0.05" 在 16-head macro-F1 上是 0.27，**这个数 P1 极可能仅靠"恒输出 const top-4 + 把 h0 换成 h2 当 scene-level 信号 ≈ 0.5"就达到**。这不是"学到了 per-scene 信息"，而是"学到了'h0 vs h2 哪个频率更高'的二元决策"。

**建议**：把主指标换成 **per-active-head F1**（只在 freq ∈ (0.05, 0.95) 的 head 上做 macro 平均），并要求 G_v0_1 = B0' + 0.05 在这个新指标上达成。

## Q5. G_v0_0/1/2/3' 接收闸的覆盖性

§10.z 当前四闸：
- G_v0_0：B0 const baseline reported（floor，不是 gate）
- G_v0_1：P1/P2 holdout per-head F1 > B0 + 0.05
- G_v0_2：报告 P2 − P1 capacity gap
- G_v0_3'：shifted-holdout 与 in-dist holdout 差距 < 0.05

**未被覆盖的失败模式**（按重要性递减）：

1. **"learner 学到的全是 head-freq prior，没有 per-scene 信号"**——这是 Q4 的核心。当前没有任何 gate 强制要求 learner 必须在 *per-scene 不同* 的预测上超过 B0。补救：加 **G_v0_extra-A**：在 holdout 上，要求 P1/P2 输出的 4-head 预测集合**不是恒等于 const top-4**的 scene 占比 > 30%，且这部分 scene 上的 macro-F1 单独算仍然 > B0。
2. **"learner 完美但任务退化"**——h13 占死一格，h14 占 0.9 格，real per-scene 决策只在 h0/h2/h4/h6 之间 2-of-4 选择。如果 G_v0_1 通过了，但**实际可学的 per-scene 信号只有 ~2 bit / scene**，v1 RL 的 PDMS reward 在这点信号上根本驱动不动 policy。补救：加 **G_v0_extra-B**：在 19,225 全集上报告 "per-scene bot-K set 的 unique 个数"。如果 unique 集合 < 20，per-scene gating 的 ceiling 已经低到不值得做 v1 RL。
3. **"R1'' 学到的是残差近邻泄漏"**——Q2 的核心。补救：加 **G_v0_extra-C**：要求 "only L0+L4" far-layer-only 控制 P2 的 macro-F1 与原 R1'' P2 的 gap < 0.05。如果 gap 很大，证明信号主要来自 L8/L16 的近邻泄漏，v1 不应该把它当 "cheap representation"，跨层迁移结论无效。

第 1 个最关键，第 2 个决定 v1 RL 值不值得跑，第 3 个保护 design doc §10.z 的核心研究 claim。

## Q6. 50 → 19,225 的统计风险

50 样本下的频率 95% **Wilson CI**：

| head | 50-样本 freq | Wilson 95% CI |
|---|---|---|
| h13 | 1.00 | [0.93, 1.00] |
| h14 | 0.90 | [0.79, 0.96] |
| h06 | 0.74 | [0.60, 0.84] |
| h00 | 0.50 | [0.37, 0.63] |
| h02 | 0.46 | [0.33, 0.60] |
| h04 | 0.40 | [0.27, 0.54] |

**关键风险情景**：

- **情景 A（最危险）**：全集上 h14 也塌到 1.00。那 v0 的 effective task = 2-of-14 选择，C(14,2)=91，**const baseline EM 直接冲到 ~0.7**，**learner 跑赢 const 的 headroom 跌到不到 0.1 EM**，G_v0_1 (+0.05) 反而可能失败——不是因为学不到，是因为没东西可学。
- **情景 B（中等）**：h14 维持 0.85–0.95，h6 升到 0.85+，h0/h2/h4 各自下降到 0.3 左右。const top-4 = {h13, h14, h6, h0/h2 二选一}，**const baseline F1/EM 略升**，learner 的有效信号集中在 h0 vs h2 vs h4 vs ... 的微弱差异上，**信噪比极低**。
- **情景 C（最理想，概率最低）**：h14/h6 维持现状，但 h0/h2/h4 之间频次接近且 per-scene 强相关于 prompt（例如夜间 vs 白天对应 h2 vs h4）。这是 R1'' design 唯一能产出强结果的情景，但 50 样本无法验证它。

**强烈建议**：在跑 P1/P2 之前，**用全 19,225 集 dump 出 head freq 表**（CPU only，分钟级，比 smoke 50 多花不了多少时间），据此决定 const top-4、G_v0_1 阈值、是否 ABORT。

---

## design-code drift

| 位置 | design doc §10.z | code 实际产出 (smoke 50) |
|---|---|---|
| const top-4 head ids | `{h13, h14, h6, h2}` | `[13, 14, 6, 0]` |

doc 写的 h2 是 design 阶段拍脑袋估的，code 不是按 doc 硬编码而是动态从数据里取 top-4。低风险但 reviewer-style 该指出：在跑全集前把 §10.z 的 baseline head 列表改成"由 dataset summary 给出"，而不是写死 4 个具体 id。

---

## 推荐的下一步 sanity（按价值排序）

1. **【最高价值，预算 < 5 分钟 CPU】全 19,225 集 head-freq 复算 + active-head 子集决定**
   把 `m1b2_phase2_v0_build_dataset.py` 跑全集（无 `--limit`），输出 `L_target_botK_head_freq` 全集版，看 h13/h14/h6 的真实频率、active head 数、const baseline F1/EM。基于这个数字**重新校准 G_v0_1 阈值**。如果 h14 ≥ 0.97 或 active head < 4 → v0 直接 ABORT 进 plan B（R1 hidden re-dump）。这一步零 GPU 成本，但**直接决定 v0 是否值得跑**。

2. **【高价值，预算 < 10 分钟 CPU + 1 GPU × 5 min】per-layer closed-form 诊断 + far-layer-only 控制**
   在全集 dataset 上算 6 个 per-layer closed-form macro-F1（仅 16→16 映射，不要 `% 16`），再训一个 P2 只用 (L0, L4)（24 维）。比较：
   - 如果 best-single-layer closed-form > 0.3 → R1'' 任务非平凡且单层就够，cross-layer 故事弱；
   - 如果 P2(L0+L4) ≈ const baseline 但 P2(full R1'') ≫ const → 信号主要来自 L8/L16 残差近邻 → §10.z 跨层迁移结论站不住。

3. **【中价值，预算 0 GPU，纯口径】"h13-masked + active-head" 评测口径正式化**
   不跑实验，把 §10.z 的 G_v0_1 / G_v0_3' 改写成基于 (a) h13-masked macro-F1，(b) per-active-head F1（freq ∈ (0.05, 0.95)），(c) "P1/P2 实际偏离 const top-4 的 scene 占比 > 30%"三条联立。改完口径再跑实验。
