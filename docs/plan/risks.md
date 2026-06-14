# RL-Drive Risks & Contingencies

> **Created**: 2026-06-14 19:30（design freeze 之后，与 `implementation_plan.md` 配套）
> **格式**：每条风险三段式 — `[失败现象] → [诊断方法] → [contingency plan]`
> **维护原则**：风险触发后，不论是否走 contingency，都在该条目下追加 `Triggered YYYY-MM-DD: ...` 一行做事后记录。
> **优先级**：🔴 致命（破坏 paper main claim） / 🟡 重要（拖延 1 周以上） / 🟢 局部（可绕开）

---

## 索引

| ID | 阶段 | 风险 | 优先级 |
|---|---|---|---|
| [R-M0-1](#r-m0-1) | M0 | NAVSIM pipeline 在 AutoVLA 上跑不通 | 🔴 |
| [R-M0-2](#r-m0-2) | M0 | AutoVLA r=1.0 baseline EPDMS 远低于论文报告值 | 🔴 |
| [R-M0-3](#r-m0-3) | M0 | navtrain stage_one cache 缺失/格式不兼容 | 🟡 |
| [R-M1a-1](#r-m1a-1) | M1a | 没有任何 ViT 层的 attention 与 EPDMS 显著相关 | 🔴 |
| [R-M1b-1](#r-m1b-1) | M1b | 全量 attention 提取磁盘/显存爆炸 | 🟡 |
| [R-M1c-1](#r-m1c-1) | M1c | LambdaRank scorer 在 val 上 NDCG ≈ random | 🔴 |
| [R-M1c-2](#r-m1c-2) | M1c | scorer 学到了 trivial bias（中心 / 远景区域） | 🟡 |
| [R-M2-1](#r-m2-1) | M2 | scorer RL 阶段 reward 不收敛 / collapse | 🟡 |
| [R-M2-2](#r-m2-2) | M2 | scorer_v2 比 scorer_v1 更差 | 🟢 |
| [R-M3-1](#r-m3-1) | M3 | ε scan 全部 collapse 到同一 budget 类 | 🔴 |
| [R-M3-2](#r-m3-2) | M3 | navtrain 全量 oracle generation 算力超预算 ×2 | 🟡 |
| [R-M3-3](#r-m3-3) | M3 | budget label 极度类不平衡（>80% 一类） | 🟡 |
| [R-M4-1](#r-m4-1) | M4 | budget policy SFT val_acc 高但 EPDMS 不涨 | 🟡 |
| [R-M5-1](#r-m5-1) | M5 | GRPO α/β grid 全部 dominated by SFT | 🟡 |
| [R-M5-2](#r-m5-2) | M5 | piecewise Pareto reward 数值不稳 / clip 频繁触发 | 🟢 |
| [R-M6-1](#r-m6-1) | M6 | 主表 iso-compute 列输给 FastV / ToMe | 🔴 |
| [R-M6-2](#r-m6-2) | M6 | navhard 上方法严重退化（OOD 不 robust） | 🟡 |
| [R-D-1](#r-d-1) | Design | attention-distill 假设根本不成立 | 🔴 |
| [R-D-2](#r-d-2) | Design | two-stage（importance × budget）联合训练崩溃 | 🟡 |
| [R-D-3](#r-d-3) | Design | M0 oracle headroom 在 AutoVLA 上 ≪ prior work 的 +0.07 | 🔴 |
| [R-D-4](#r-d-4) | Design | reviewer 质疑 token-mean vs official combined 口径 | 🟡 |
| [R-Ops-1](#r-ops-1) | Ops | 4 卡 H20 deadline 突然提前回收 | 🟡 |
| [R-Ops-2](#r-ops-2) | Ops | ckpt 丢失 / 训练中断无 resume | 🟢 |

---

## R-M0 类（infra & baseline）

### <a id="r-m0-1"></a>R-M0-1 🔴 NAVSIM pipeline 在 AutoVLA 上跑不通

**失败现象**
- AutoVLA forward 接进 NAVSIM `run_pdm_score_from_submission.py` 后报 shape / dtype / interface mismatch
- 或 stage_two closed-loop 无法调用 AutoVLA 的 action head

**诊断方法**
1. 先用 prior work 已 patched 的 NAVSIM glue 代码当 reference（`/apdcephfs/private_shayladeng/tokenrl/code/navsim/...`）
2. 单帧 forward 走通后再 batch
3. 用 r=1.0（无剪枝）跑通 5 scene 的 stage_two，确认 trajectory 输出格式与 NAVSIM 期望一致

**Contingency**
- **Plan A（首选）**：参考 prior work 的 9 处适配 patch（`docs/_internal/handoff.md` §映射表有路径），逐处 port 到 AutoVLA wrapper
- **Plan B**：把 AutoVLA 的 trajectory 输出格式封装成 ReCogDrive submission.pkl 兼容格式，**复用 prior work 的 NAVSIM eval 入口**（牺牲一点优雅度换稳定）
- **Plan C（最后兜底）**：放弃 navhard_two_stage closed-loop，改用 navtest open-loop EPDMS 作为主 metric，但需在 paper §4.1 写明并附 OL→CL 相关性论证

---

### <a id="r-m0-2"></a>R-M0-2 🔴 AutoVLA r=1.0 baseline EPDMS 远低于论文报告值

**失败现象**
- AutoVLA paper 报告 navhard EPDMS ≈ 0.45+，但本地复现 < 0.30
- 且不是 stage_one cv 占位拖累（已扣除）

**诊断方法**
1. 检查 ckpt 是否对：HuggingFace 的 official AutoVLA ckpt vs 训练自己的；mismatch 概率最高
2. 检查 image normalization / camera 顺序 / token order：AutoVLA 用的是 8-cam 还是 6-cam，与 NAVSIM 输入对齐没
3. 检查 trajectory horizon / dt：AutoVLA 默认 vs NAVSIM 期望
4. 跑 5 个 scene 的可视化对比 GT，看 trajectory 是不是质量本身就差

**Contingency**
- **Plan A**：联系 AutoVLA 作者 / Issues 区确认 navhard 评测脚本（很多 paper 报的是 navtest）
- **Plan B**：接受 AutoVLA 本地复现值作为新 baseline，paper 中明确"reproduced under NAVSIM v2"，所有方法都基于这个 baseline 增量比较，不与论文原值横向对比
- **Plan C（致命兜底）**：换 backbone 到 LMDrive 或 OmniDrive（开源且评测脚本清晰），design 不变

---

### <a id="r-m0-3"></a>R-M0-3 🟡 navtrain stage_one cache 缺失/格式不兼容

**失败现象**
- navtrain 全量 EPDMS 跑 oracle 时，stage_one closed-loop pre-rollout cache 不存在或格式与 navhard 的不一样

**诊断方法**
1. `ls $NAVSIM_EXP_ROOT/metric_cache_navtrain/` 看是否存在
2. 抽 1 个 token 进 stage_one runner，看是否能直接 hit cache

**Contingency**
- **Plan A**：自己跑一遍 navtrain 的 stage_one cv（参考 prior work `compute_navtrain_metric_cache.sh`），约 8 GPU·hour
- **Plan B**：navtrain 也用 stage1 占位（与 prior work 06-01 同口径），但需在 §3.4 注明 oracle label 受占位影响，并报告 navhard 上的 ablation 验证 ranking 一致性

---

## R-M1 类（Stage A: Importance Scorer）

### <a id="r-m1a-1"></a>R-M1a 🔴 没有任何 ViT 层的 attention 与 EPDMS 显著相关

**失败现象**
- 100-scene probing：所有 ViT 层（1–24）attention rollout 与 per-token EPDMS-drop 的 Spearman ρ < 0.1
- 即 attention distill 假设在 AutoVLA 上压根不成立

**诊断方法**
1. Sanity check：attention rollout 是否计算正确（与 ViT paper 公式一致）
2. EPDMS-drop 的统计有没有问题：单 token mask 后 Δ EPDMS 的方差是否远大于 mean（信噪比太低）
3. 换粒度：从 single-token mask 改成 patch-block (4×4) mask，再算相关
4. 检查 LLM cross-attention（LLM → vision tokens）是否信号更强（很可能是这个比 ViT self-attn 更靠谱）

**Contingency**
- **Plan A（最可能命中）**：换信号源——用 **LLM 第一层 cross-attention 对 vision tokens 的注意分布** 当 distill target，不再用 ViT self-attn。这是 FastV 路线，已被验证
- **Plan B**：放弃 distill，**直接用 EPDMS-drop label 监督 scorer**（label 噪声大但信号真实），训练成本 ×3
- **Plan C**：合并 Stage A 和 Stage B，end-to-end 用 GRPO 学 importance（去掉 SFT warmup），风险更高但保留 paper 主轴

> ⚠️ 这条触发即影响 R-D-1（design 假设崩塌），需在 design_decisions.md 加 Revision 条目。

---

### <a id="r-m1b-1"></a>R-M1b 🟡 全量 attention 提取磁盘/显存爆炸

**失败现象**
- navtrain 全量（~85k frames）× 8-cam × 256 token × 24 层 attention map 一次性存下需 TB 级
- 或 forward hook OOM

**诊断方法**
1. 单 frame 估算：`8 × 256 × 256 × 24 × 4 bytes ≈ 50MB`，全量 ≈ 4 TB
2. 看 metric_cache 类似存储有没有现成的 chunk + zstd 压缩 schema 可参考

**Contingency**
- **Plan A**：只存 **rollout 后的 token 重要性 vector**（256 维 / frame），不存原始 attention map，磁盘 ÷ 24
- **Plan B**：on-the-fly 计算 ranking label，不落盘——M1c 训练时 dataloader 实时跑 ViT forward 拿 attention（增加训练时间但省存储）
- **Plan C**：降采样到 navtrain 的 50% 子集（按 log_name stratified sample）

---

### <a id="r-m1c-1"></a>R-M1c 🔴 LambdaRank scorer val NDCG ≈ random

**失败现象**
- 训完 scorer_v1，val set NDCG@k 在 0.50 ± 0.05（random baseline ~0.5）
- 意味着 scorer 没学到任何东西

**诊断方法**
1. label 本身是否 informative：算 train label 的 entropy / 看 top-k overlap across frames（如果 top-k 永远是同一批 token → label 退化）
2. 模型容量太小？换 backbone 试 DINOv2-base
3. loss 实现 bug：sanity 用 toy dataset (n=100) 跑 overfit test，应在 50 epoch 内 train NDCG → 1.0

**Contingency**
- **Plan A**：改 loss 为 pairwise hinge / listwise softmax（LambdaRank 实现复杂，常见数值坑）
- **Plan B**：从 ranking label 换成 **二分类 label（top-50% vs bottom-50%）+ BCE**，简化任务
- **Plan C**：触发 R-D-1 contingency，整体改走 EPDMS-drop label 直接监督

---

### <a id="r-m1c-2"></a>R-M1c-2 🟡 scorer 学到 trivial bias（中心 / 远景）

**失败现象**
- scorer attention map 可视化全是图像中心或远景一整条
- val NDCG 看着不错但本质学的是 spatial prior，不是 scene-adaptive

**诊断方法**
1. 算 scorer 输出与 "纯中心 mask" / "纯远景 mask" 的 IoU；若 > 0.7 即 trivial collapse
2. 跨场景一致性：同一 token 位置在不同 scene 的 score 标准差 / 均值；如果 < 0.1 即 location bias dominant

**Contingency**
- **Plan A**：在 loss 加 **per-position bias 惩罚项**（减去 batch 内同位置 score 的均值再算 ranking loss），强制学 residual
- **Plan B**：作为 ablation 论点保留——"trivial spatial prior 已是强 baseline"，转而强化 budget policy 的卖点
- **Plan C**：扩大训练 scene 多样性（确保城市 / 高速 / 路口 / 夜晚分布均衡）

---

## R-M2 类（Stage A: Importance Scorer RL）

### <a id="r-m2-1"></a>R-M2-1 🟡 scorer RL 阶段 reward 不收敛 / collapse

**失败现象**
- GRPO 训 1000 step 后 reward mean 不增长 / 抖动剧烈
- 或 policy entropy 急剧下降到 < 0.1（mode collapse）

**诊断方法**
1. reward 信号 SNR：算 group 内 advantage 的方差，若 < 0.01 即区分度不够
2. KL(π||π_ref) 是否炸：> 0.5 说明 reference 拉不住
3. random-B sampling 的 B 是否合理（B 太小 → 高方差；太大 → 区分度差）

**Contingency**
- **Plan A**：增大 KL coef（β: 0.04 → 0.1）；减小 lr
- **Plan B**：改用 **REINFORCE + baseline**（去掉 group reweighting），训练更稳但慢
- **Plan C**：跳过 M2 直接 freeze scorer_v1 进 Stage B，paper 中 M2 改写成 ablation A6（"RL-tuning is not necessary for scorer"，反向叙事）

---

### <a id="r-m2-2"></a>R-M2-2 🟢 scorer_v2 比 scorer_v1 更差

**失败现象**
- scorer_v2 在 100-scene probe 上 EPDMS 比 scorer_v1 低

**诊断方法**
- 直接测：scorer_v1 vs scorer_v2 在 r ∈ {0.25, 0.5, 0.75} × 100 scene 的 EPDMS 网格

**Contingency**
- 直接采用 scorer_v1 进 Stage B，M2 在 paper 中按 ablation 表 A6 处理（negative result 也是结果）

---

## R-M3 类（Stage B: Oracle Generation）

### <a id="r-m3-1"></a>R-M3-1 🔴 ε scan 全部 collapse 到同一 budget 类

**失败现象**
- ε ∈ {0.005, 0.01, 0.02, 0.05, 0.1} 任一选择，oracle label 都 > 80% 落到同一类（最常是 r=0.25 或 r=1.0）
- 与 prior work M0 oracle 47% / 38% 两极分化对比，AutoVLA 上更极端

**诊断方法**
1. 看 per-r EPDMS 分布：如果 r=1.0 vs r=0.25 的 mean diff < 0.005 → 可剪空间太小
2. 看 per-token Δ EPDMS 方差：如果 std < 0.02 → 信号弱
3. 与 prior work 同样口径下数字对比，找 backbone 差异

**Contingency**
- **Plan A**：ε 改自适应，按 frame 设阈值 ε_i = α · σ(EPDMS_i 跨 r) ，避免全局 ε 太硬
- **Plan B**：扩 r 候选集到 {0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0}，更细粒度
- **Plan C（致命）**：触发 R-D-3 — AutoVLA 上 headroom 太小，重新评估是否换 backbone

---

### <a id="r-m3-2"></a>R-M3-2 🟡 navtrain 全量 oracle generation 算力超预算 ×2

**失败现象**
- 计划 navtrain × 4 r × stage_two ≈ X GPU·day，实测 ≥ 2X

**诊断方法**
- M3 启动前先跑 1000 frame benchmark，外推总时间

**Contingency**
- **Plan A**：navtrain 降采样到 50%（log-stratified），M6 主表用全量 navtest 评测仍合规
- **Plan B**：跳过 r=0.5、r=0.75，只跑 r=0.25 / 1.0，oracle 退化为 2 类（heavy-prune vs no-prune）
- **Plan C**：用 scorer_v2 作 importance × 2 类 budget 直接生成 label，跳过完整 oracle search

---

### <a id="r-m3-3"></a>R-M3-3 🟡 budget label 极度类不平衡（>80% 一类）

**失败现象**
- ε 选合理但 label 仍极不平衡

**诊断方法**
- 直接看 4 类频率分布

**Contingency**
- **Plan A**：M4 SFT 用 class-weighted CE，权重 = 1/freq（参考 prior work v4 教训：单 weight 不够，需配合 5+ epoch）
- **Plan B**：focal loss (γ=2)
- **Plan C**：重采样 + mixup

---

## R-M4 类（Budget Policy SFT）

### <a id="r-m4-1"></a>R-M4-1 🟡 val_acc 高但 EPDMS 不涨

**失败现象**
- budget_policy_v1 val_acc 0.7+，但 navhard 上 adaptive EPDMS ≤ best fixed-r
- ⚠️ 这正是 prior work v3 的失败模式（acc 0.6 但 collapse 到一类）

**诊断方法**
1. 出 confusion matrix：是不是单类占优造成的虚假高 acc
2. 看 per-class recall（不只是 weighted avg）
3. 算 adaptive 决策的 entropy；若 < 0.3 即 collapse

**Contingency**
- **Plan A**：直接走 R-M3-3 Plan A 的 class-weighted CE 重训
- **Plan B**：跳过 SFT 直接 GRPO（M5），SFT init 用 "all-r=0.5" 均匀策略
- **Plan C**：在 SFT loss 上加 **决策 entropy 正则项** 鼓励多样化预测

---

## R-M5 类（Budget Policy RL）

### <a id="r-m5-1"></a>R-M5-1 🟡 GRPO α/β grid 全部 dominated by SFT

**失败现象**
- α/β scan 出来的 budget_policy_v2 在 100-scene probe 上 Pareto 严格劣于 v1

**诊断方法**
1. 看 reward shaping 是否合理：α 太大压 EPDMS、β 太大压 token cost
2. group size 是否够（建议 G≥8）
3. KL 限制是否过严

**Contingency**
- **Plan A**：扩 grid（α ∈ {0.5,1,2,5}，β ∈ {0.1,0.3,1,3}）
- **Plan B**：改 reward 结构为 **constrained optimization**（拉格朗日，自适应 β）
- **Plan C**：M5 negative result，主表用 v1（SFT-only），M5 写进 ablation A8

---

### <a id="r-m5-2"></a>R-M5-2 🟢 piecewise Pareto reward 数值不稳

**失败现象**
- reward clip[-1,1] 频繁触发上下界（>30% step）
- training curve 呈台阶状

**Contingency**
- 把 piecewise 改 smooth：`r = tanh(α·ΔEPDMS - β·Δcost)`，去掉 hard piecewise

---

## R-M6 类（Evaluation）

### <a id="r-m6-1"></a>R-M6-1 🔴 主表 iso-compute 列输给 FastV / ToMe

**失败现象**
- avg ratio = 0.5 这一列上，FastV 或 ToMe EPDMS 比本方法高 / 持平
- 直接威胁 paper main claim

**诊断方法**
1. 抽 100 scene 看 FastV 在哪些 frame 上比本方法好；找 pattern
2. 检查是不是 scorer + budget 两阶段反而引入了误差，end-to-end FastV 反而更稳

**Contingency**
- **Plan A**：把 FastV / ToMe 当 importance 信号 ensemble 进本方法（"我们的 budget policy 通用"，把 baseline 变成 component）
- **Plan B**：换 ratio 战场——若 ratio = 0.5 输了，看 ratio = 0.3 / 0.7 是否赢；调整 main table 主比较点
- **Plan C**：故事重写——从 "Pareto 全面 dominate" 改 "scene-adaptive ratio control"，主卖点变成 budget policy 本身（不再要求 scorer 全面赢）

---

### <a id="r-m6-2"></a>R-M6-2 🟡 navhard 上严重退化

**失败现象**
- 在 navtest 主表赢，但 navhard（OOD）上比 fixed-r baseline 差

**诊断方法**
- 看 navhard 哪类场景退化最重（cluster by weather / city / traffic density）

**Contingency**
- **Plan A**：M3 oracle generation 加进 navhard 的小 subset 一起训
- **Plan B**：navhard 结果作为 limitation 写进 §6 discussion，主表 navtest 站得住即可
- **Plan C**：训一个 navhard-specific budget head（小代价，1 epoch）

---

## R-Design 类（设计本身崩塌）

### <a id="r-d-1"></a>R-D-1 🔴 attention-distill 假设根本不成立

**失败现象**
- R-M1a-1 触发：所有 ViT 层 attention 与 EPDMS-drop 不相关

**Contingency**
- 已在 R-M1a-1 列出（首选 LLM cross-attn → EPDMS-drop direct supervision → end-to-end RL）
- 触发后必须在 `design_decisions.md` 加 Revision 条目，并通知用户

---

### <a id="r-d-2"></a>R-D-2 🟡 two-stage 联合训练崩溃

**失败现象**
- scorer_v2 + budget_policy_v2 联合 inference 时，EPDMS 比单独 v1+v1 还差
- 两个模块互相 confuse

**Contingency**
- **Plan A**：放弃 joint inference，固定 scorer_v2 freeze 后再训 budget policy（保持顺序训练，禁止反向梯度互通）
- **Plan B**：合并成 single-policy 输出 (importance_score, budget)，end-to-end GRPO

---

### <a id="r-d-3"></a>R-D-3 🔴 AutoVLA 上 oracle headroom ≪ prior work +0.07

**失败现象**
- M0 / M3 阶段算出 AutoVLA 上 oracle headroom < +0.02 EPDMS
- 意味着 AutoVLA 本身对 vision token 剪枝不敏感，paper 卖点立不住

**诊断方法**
- 直接复现 prior work 的 M0 oracle headroom 协议在 AutoVLA 上跑

**Contingency**
- **Plan A**：换 metric——从 "EPDMS 持平 + 省 token" 改 "**速度 / 显存 / 端到端 latency** 实测加速"，把 efficiency 从 token count 升级到 wall-clock，故事不依赖 EPDMS headroom
- **Plan B**：换 backbone（LMDrive / OmniDrive），M0 重做
- **Plan C**：合并 prior work 的 ReCogDrive 实验作为第二个 backbone，paper 变成 "method generalizes across backbones"

---

### <a id="r-d-4"></a>R-D-4 🟡 reviewer 质疑 token-mean vs official combined 口径

**失败现象**
- prior work M0 报的是 token-mean (0.4377)，但 NAVSIM 官方主报 combined（含 stage_one cv 占位的乘法聚合）
- reviewer 指责 cherry-pick 口径

**Contingency**
- **Plan A（必须做）**：M3 完成后跑一次 oracle mask 的 closed-loop 真评测，给出 official combined 上的 oracle 上界（验证 token-mean 的 +0.07 在 combined 口径下能保留多少）
- **Plan B**：paper main table **同时报 token-mean 和 official combined 两列**，限制把 limitation 提前写在 §3.4
- **Plan C**：放弃 token-mean 口径，全程用 official combined（更保守但更经得起 review）

> 关联：prior work `results/M0_summary.md §7` 已列入 limitation；M3 完成时必须验证。

---

## R-Ops 类（运维 / 算力）

### <a id="r-ops-1"></a>R-Ops-1 🟡 4 卡 H20 deadline 突然提前回收

**失败现象**
- 训练中途 GPU 被回收（参考 prior work 06-13 v4 失败案例）

**Contingency**
- **Plan A**：所有训练脚本默认带 watchdog + 每 epoch checkpoint（参考 `tokenrl/code/scripts/predictor_watchdog.sh` 模板），最多损失 1 epoch
- **Plan B**：长任务拆成 ≤ 6h 的子任务串行
- **Plan C**：申请 backup 算力（A100 / H800 节点）作 failover

---

### <a id="r-ops-2"></a>R-Ops-2 🟢 ckpt 丢失 / 训练中断无 resume

**Contingency**
- 所有训练默认每 epoch save + `backups/` 目录每天 rsync 一次（已在 .gitignore，但物理保留）
- 关键 ckpt（每个 milestone 的 final）单独 archive 到 `ckpt/archived/`

---

## 风险地图（dependency）

```
R-M0-1 ──┐
R-M0-2 ──┼──> 决定能否 launch
R-M0-3 ──┘
            │
            ▼
R-D-3 ◄── M0 oracle headroom 检查 ──► R-M3-1
            │                                │
            ▼                                ▼
R-M1a-1 ──► R-D-1（design pivot）       budget label 立不立得住
            │
            ▼
R-M1c-1 / R-M1c-2  ──► scorer 质量
            │
            ▼
R-M2-1 / R-M2-2  ──► RL 是否必要（可降级 ablation）
                                    │
                                    ▼
                           R-M4-1 ──► budget policy 是否真 adaptive
                                    │
                                    ▼
                           R-M5-1 ──► GRPO 是否必要（可降级 ablation）
                                    │
                                    ▼
                           R-M6-1 ──► main claim 是否成立
                           R-M6-2 ──► OOD robustness
                           R-D-4 ──► 口径 defense
```

**最高优先级看护点**：
1. M0 baseline reproducibility（R-M0-2）
2. M0 oracle headroom（R-D-3）
3. M1a attention-EPDMS 相关性（R-M1a-1 → R-D-1）
4. M6 主表 iso-compute 列（R-M6-1）

这 4 条任一触发，paper main claim 都需要重新评估。

---

## 维护日志

- 2026-06-14 19:30 created（risks v1，覆盖 M0–M6 + Design + Ops 共 23 条）
