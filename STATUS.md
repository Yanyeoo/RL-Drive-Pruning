# STATUS

> 2026-08-26/27 RL SOTA H20 周期（v11：hard-example mining）
> 周期号 20260826_v11r1 ｜ **已完成**：mining → 训练 → gate → full 全部 DONE（08-27 14:28）
> full 胜者 = **hard50**（full 4-shard **0.898360**，距 no-prune SOTA **-0.000485**）
> ⚠️ **gate 反超（+0.000675）但 full 未反超（-0.000485）—— 第三次 gate/full 背离**
> 下一个 AI 接手先读：本文件 + 「v11 结论」+ 「v12 建议方向」

## 状态：🟡 v11 完成，仍未反超 SOTA（0.898360 vs 0.898845）

## 不可更改的总目标

**在当前 RL idea 无大改的情况下达到 SOTA 且具备 efficiency。后续 AI 不得降低、替换或改写该目标。**

## 目标与验收

- 基准：learned SFT full-navtest PDMS = 0.89199；SOTA 上界（no-prune）= **0.898845**；SFT r0.75 = 0.898353。
- 验收1：超过 0.89199 ✅（v7 0.894798 → v10 0.898274 → v11 **0.898360**）。
- 验收2：反超 0.898845 ❌ **仍差 -0.000485**（v10 -0.000571 → v11 -0.000485，仅 +0.000086）。
- 验收3：keep_ratio ✅（≈0.75）/ FLOPs ✅（理论表已算）/ latency ❌（profiler 有 bug，见风险）。

## v11 结论（本轮核心，已完成）

**方法 = hard-example mining**：不改超参、不改 RL idea，只改**训练分布**。

### 阶段 1：mining（navtrain 8190 scenes，v10 full_hi 确定性策略逐场景 rollout）

| 类别 | 场景数 | 占比 | sum delta |
|---|---|---|---|
| 灾难（delta ≤ -0.3） | 200 | 2.44% | **-150.07** |
| 轻微负 | 928 | 11.33% | -13.50 |
| 恰好为 0（剪枝免费） | 5852 | 71.45% | 0 |
| 正（剪枝反超 no-prune） | 1210 | 14.77% | +178.32 |

→ **灾难场景仅 2.44% 却占总负 delta 的 91.7%**（navtest 诊断 87.6%，跨集一致 ⇒ 结构性、可迁移）。
→ 71.45% 场景剪掉 ~28% token 完全免费 ⇒ efficiency 基础扎实。

### 阶段 2-3：训练 + 评估（4 臂全部 warm-start 自 v10 full_hi，只改采样分布）

| arm | 训练分布 | kr 变化 | 训练 reward | gate(sh0+1) | full |
|---|---|---|---|---|---|
| **hard50 ⭐** | 50% hard + 50% normal（hard ×2） | 0.691→0.747 | -1.11→-0.41 | **0.899520** | **0.898360** |
| hard50_kl | 同上，kl_beta 0.02 | 0.691→0.686 | -1.11→-0.55 | 0.898909 | — |
| hard50_lr | 同上，budget_lr 2e-4 | 0.685→0.710 | -1.03→-0.57 | 0.898425 | — |
| hard75 | 75% hard + 25% normal | 0.691→0.765 | -0.98→**-0.29** | 0.897437 | — |

**hard50 full 分 shard**：sh0 0.898353 / sh1 0.900397 / sh2 **0.892993** / sh3 0.901926（N=11576）。

### 🔴 为什么 gate 反超却 full 没有：逐场景归因（v10 vs v11 vs no-prune，对齐 11573）

| | mean | delta vs no-prune | 灾难场景数 | 反超 no-prune 场景数 |
|---|---|---|---|---|
| v10 full_hi | 0.898288 | -0.000579 | 90 | 639 |
| v11 hard50 | 0.898360 | -0.000506 | **82** | **550** |

**救回 42 个灾难场景，但新弄坏 34 个**（净减 8），同时**反超场景从 639 掉到 550（-89）**。
→ 净收益仅 +0.000072，属噪声级。

**根因（方法层面，非 bug）**：全局抬升 keep_ratio 是**钝器** —— 它同时改变所有场景的行为。
换来的灾难场景安全性，代价是丢失了原本剪枝带来的正向收益（那 89 个场景）。
hard-mining 成功让 budget head 学到「危险场景少剪」（kr 0.691→0.747，reward -1.11→-0.41），
但**场景级单一 keep_ratio 的表达能力不足**，无法做到「危险场景多留 + 安全场景照常剪」。

### 另一个关键教训

🔴 **hard75 训练 reward 最好（-0.29）但 gate 最差（0.897437）** ——
hard 比例过高会过拟合灾难场景。**训练 reward 与泛化表现明确背离，不得据训练 reward 选 winner。**
`--hard-frac 0.5` 的 hard+normal 混采是必要设计（保留 efficiency 锚）。

## 历史结论

### v10（追平 SOTA）
- winner **full_hi**：full **0.898274**（-0.000571）；kr≈0.72 → 省 19.0% 总 FLOPs / 22.3% LLM prefill。
- 方法 = value baseline（advantage = reward − V(s)）+ efficiency floor + budget-init 锚定。

### v9（教训）
- 🔴 单 knob 超参到头：gate 0.895413 但 full 回归 0.894345（< v7 0.894798）。

## v12 建议方向（基于 v11 归因）

v11 证明「改分布」也已接近天花板，瓶颈是**策略表达能力**。三个候选（按优先级）：

1. **场景级 kr → 条件化 kr（推荐）**：当前 budget head 只用 mean-pooled 特征输出一个标量。
   改为让 kr 显式条件于「风险信号」（如 baseline 子分数、障碍物密度、attention 熵），
   使策略能区分「该多留」与「可照常剪」，直击 v11 暴露的钝器问题。
2. **per-token 保护而非全局抬 kr**：对灾难场景做 counterfactual（`--counterfactual-k` 已实现但本轮关闭），
   定位真正关键的少数 token 并保护它们，而不是整体少剪。
3. **非对称 reward**：对「剪枝后崩盘」施加远大于收益的惩罚（risk-averse 目标），
   让策略只在有把握时才激进剪枝。

## 已完成（倒序）

- [x] 08-27 14:28 v11 full 完成（hard50 0.898360，-0.000485，未反超）+ 逐场景归因
- [x] 08-27 10:46 v11 gate 完成（hard50 0.899520，gate 反超 +0.000675）
- [x] 08-27 03:34 v11 训练完成（4 臂 warm-start 自 full_hi）
- [x] 08-27 01:20 v11 mining 完成（navtrain 8190，200 灾难场景，占 91.7% 负 delta）
- [x] 08-26 代码：`--mine-mode` / `--scene-list` / `--init-budget-ckpt`；orchestrator 全流程
- [x] 08-26 09:29 v10 r1 full（full_hi 0.898274）
- [x] v9 r1（full 回归 0.894345）；v7 full（st_topk 0.894798）

## 待办（优先级降序）

1. [ ] **v12：按上方「建议方向 1」做条件化 kr**（v11 已证明全局 kr 是瓶颈）
2. [ ] 修 `profile_wallclock.py`：agent 推理路径与评估不一致，24/24 场景解码失败（见风险）
3. [ ] 可选：hard50_kl（gate 0.898909，第二名）补 full，确认是否比 hard50 稳

## 决策记录

| # | 时间 | 决策 | 选择 | 理由 | reverse | 确认 |
|---|---|---|---|---|---|---|
| 1 | 08-18 | 主线方法 | A(可微Top-K)+C(delta reward) | 用户拍板 | 回退 softmax | ✅ |
| 2 | 08-18 | A 实现选型 | st_topk + gumbel 都做 | 用户拍板 | — | ✅ |
| 3 | 08-19 | winner | st_topk | min-shard 最高且超 SFT | 改 tau1 | ✅ full 确认 |
| 4 | 08-25 | 方法级改动 | value baseline + kr 上浮 | v9 单 knob 到头 | — | ✅ 追平 SOTA |
| 5 | 08-26 | v11 方向 | hard-example mining（改分布） | 灾难场景 2.4% 占 91.7% 负 delta | 回退 full_hi | ⚠️ gate 反超 full 没有 |
| 6 | 08-26 | 训练分布 | hard+normal 混采 frac 0.5 | 保 efficiency 锚 | 调 frac | ✅ hard75 更差已验证 |
| 7 | 08-27 | v11 归因 | 瓶颈 = 场景级单一 kr 表达力不足 | 救回 42 但弄坏 34、反超场景 -89 | — | ✅ 逐场景数据支撑 |

## 卡点 / 风险

- 🔴 **gate ≠ full（已第三次）**：v9、v11 均是 gate 乐观、full 回落。**任何结论必须 full 4-shard 定论**。
- 🔴 **latency 未验收**：`profile_wallclock.py` 已修 3 处（hydra 配置名/路径、SceneLoader 签名、sensor 路径改用
  `$OPENSCENE_DATA_ROOT`、改直读 navtest json），但 agent 推理仍在
  `decode_token_ids_to_trajectory` 全部失败（24/24）。**efficiency 目前只有 kr + 理论 FLOPs，latency 待修**。
- 🔴 **`data/navsim_v2_local` 不存在**：多个脚本（含 eval/train 的 `SENSOR` 变量）指向该路径。
  评估能跑通说明该参数在 eval 链路未被真正使用；真实 sensor 在
  `$OPENSCENE_DATA_ROOT/sensor_blobs/test/openscene-v1.1/sensor_blobs/test`。
- 🔴 **eval worker>1 有 race（H20）**：必须 `EVAL_WORKERS=1`。
- ⚠️ **本节点仅 4 张 H20**：gate 8 job 分 2 波，单波约 3.7h；kr 升高后 eval 比 v10 慢约 15%。
- ⚠️ **不得据训练 reward 选 winner**（hard75 已实证背离）。

## 关键路径

- **v11 全流程（无人值守）**：`scripts/orchestrate_v11_pipeline.sh`
- mining：`scripts/run_v11_mining_4gpu.sh [n_scenes]` → `results/mining_20260826_v11/mine_shard*.jsonl`
- 组装：`scripts/build_v11_scene_list.py --hard-frac 0.5 --hard-repeat 2`
- 训练（v11）：`scripts/run_v11_hardmine_4gpu.sh <list50> <list75> [round]`（CYCLE_ID=`20260826_v11r1`）
- 评估：`EVAL_WORKERS=1 EVAL_GPUS="0 1 2 3" bash scripts/eval_v7_folds_4gpu.sh <CYCLE_ID> [gate|full] [arms...]`
- 训练核心：`scripts/train_scorer_budget_rl.py`（`--mine-mode` / `--scene-list` / `--init-budget-ckpt`）
- 效率：`scripts/compute_flops_table.py`（可用）、`scripts/profile_wallclock.py --budget-ckpt <dir>`（待修）
- ckpt（v11）：`ckpt/v7_surrogate_20260826_v11r1/{hard50,hard50_lr,hard75,hard50_kl}/checkpoint.pt`
- CSV（v11）：`results/raw/v7_surrogate_20260826_v11r1_{gate,full}/`
