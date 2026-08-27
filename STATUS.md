# STATUS

> 2026-08-26/27 RL SOTA H20 周期（v11：hard-example mining）
> 周期号 20260826_v11r1 ｜ **已完成并收尾**：mining → 训练 → gate → full ×2 全部 DONE（08-27 19:06）
> full 最佳 = **hard50_kl**（full 4-shard **0.898719**，距 no-prune SOTA **-0.000126**）
> ⚠️ 仍未反超，但已是历史最接近（v10 -0.000571 → hard50 -0.000485 → **hard50_kl -0.000126**）
>
> 🔴 **节点已于 08-27 20:00 回收。所有产物已入 GitHub（`db9ba83`），无本地遗留依赖。**
> **下一个 AI 接手：先读本文件「v12 起步指令」一节，可直接开跑，无需重跑 mining。**

## 状态：🟡 v11 完成，hard50_kl 逼近 SOTA 至 -0.000126（未反超）；节点已回收

## v12 起步指令（新节点上照此执行）

前置（新机器）：按 `docs/REPRODUCE.md` §1-§4 装环境、下数据、生成 json + metric cache。
**不需要重跑 mining** —— 8190 场景的 delta 已存于 `artifacts/v11/mining/`（2.6 GPU-hours）。

**v12 = kl_beta 扫描**（v11 归因指向的唯一「只涨不跌」knob）：

```bash
# 1. 复用已有 mining 结果重建场景表（秒级，无需 GPU）
python scripts/build_v11_scene_list.py \
    --mine-glob 'artifacts/v11/mining/mine_shard*.jsonl' \
    --out results/mining_v12/scenes_hard50.txt \
    --n-total 512 --hard-frac 0.5 --hard-repeat 2

# 2. 从 hard50_kl 续训，扫 kl 0.03/0.05/0.08（4 卡 4 臂，约 1.9h）
#    改 run_v11_hardmine_4gpu.sh：
#      INIT_CKPT=release_ckpt/v11_hard50_kl     # 从当前最佳续训
#      四臂 --kl-beta 分别 0.03 / 0.05 / 0.08 / 0.02(对照)
#    其余参数保持不变（value baseline + floor + max_kr 0.85 + st_topk tau 0.1）

# 3. gate（4 卡分 2 波，约 3.7h/波）→ 选 winner → full（约 3.7h）
EVAL_WORKERS=1 EVAL_GPUS="0 1 2 3" bash scripts/eval_v7_folds_4gpu.sh <CYCLE> gate <arms>
EVAL_WORKERS=1 EVAL_GPUS="0 1 2 3" bash scripts/eval_v7_folds_4gpu.sh <CYCLE> full <winner>
```

**判据**：full 4-shard > 0.898845 才算反超。gate 只用于筛选，**不得据 gate 或训练 reward 下结论**
（v9/v11 已两次实锤 gate 乐观；hard50 gate 最高但 full 输给 hard50_kl）。

若 kl 扫描收益饱和，转「条件化 kr」（见下方 v12 建议方向 2）。

## 不可更改的总目标

**在当前 RL idea 无大改的情况下达到 SOTA 且具备 efficiency。后续 AI 不得降低、替换或改写该目标。**

## 目标与验收

- 基准：learned SFT full-navtest PDMS = 0.89199；SOTA 上界（no-prune）= **0.898845**；SFT r0.75 = 0.898353。
- 验收1：超过 0.89199 ✅（v7 0.894798 → v10 0.898274 → v11 **0.898719**）。
- 验收2：反超 0.898845 ❌ **差 -0.000126**（已从 -0.000571 收窄到 -0.000126，量级接近噪声）。
- 验收3：keep_ratio ✅（≈0.69~0.75）/ FLOPs ✅（理论表）/ latency ❌（profiler 有 bug，见风险）。

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

### 阶段 2-3：训练 + 评估（4 臂全部 warm-start 自 v10 full_hi，只改采样分布/正则）

| arm | 配置 | kr 变化 | 训练 reward | gate(sh0+1) | full 4-shard |
|---|---|---|---|---|---|
| hard50 | 50/50 混采，kl 0.01 | 0.691→0.747 | -1.11→-0.41 | **0.899520** | 0.898360 |
| **hard50_kl ⭐** | 50/50 混采，**kl 0.02** | 0.691→0.686 | -1.11→-0.55 | 0.898909 | **0.898719** |
| hard50_lr | 50/50 混采，budget_lr 2e-4 | 0.685→0.710 | -1.03→-0.57 | 0.898425 | — |
| hard75 | 75% hard，kl 0.01 | 0.691→0.765 | -0.98→**-0.29** | 0.897437 | — |

**分 shard**：
- hard50：0.898353 / 0.900397 / **0.892993** / 0.901926
- hard50_kl：0.898818 / 0.900555 / **0.893456** / 0.902266（min shard 均为 sh2）

### 🔴 逐场景归因（对齐 N=11572）—— 本轮最重要的发现

| | mean | delta vs no-prune | 灾难场景 | **反超 no-prune 场景** |
|---|---|---|---|---|
| v10 full_hi | 0.898287 | -0.000579 | 90 | 639 |
| hard50 | 0.898359 | -0.000507 | 82 | **550（-89）** |
| **hard50_kl** | 0.898797 | **-0.000068** | 83 | **606（-33）** |

**hard50_kl 的优势不来自多救灾难场景（82 vs 83，基本相同），而来自少丢正向收益（-33 vs -89）。**

机制：`kl_beta` 0.01→0.02 收紧对 `token_net` 的约束，保住 SFT 学到的 token 排序能力；
而 KL 只作用于 token_net，`budget_net` 仍自由学习。于是 kr 几乎没抬（0.691→0.686）
却拿到最好的 full 成绩。

**⇒ 核心教训（三次验证）**：
1. hard-mining 的价值 = 让网络知道哪里危险；但**全局抬 kr 是钝器**，买到灾难场景的
   安全性，代价是丢掉正常场景的剪枝收益。
2. **正则（kl）比抬 kr 更有效** —— 保住 SFT 排序 + 让 budget 自由，是当前配方最优。
3. **训练 reward / kr 抬升与泛化明确背离**：hard75 reward 最好（-0.29）gate 最差；
   hard50_kl kr 几乎不动却 full 最好。**选 winner 只能看 gate/full。**

## 历史结论

### v10（追平 SOTA）
- winner **full_hi**：full **0.898274**（-0.000571）；kr≈0.72 → 省 19.0% 总 FLOPs / 22.3% LLM prefill。
- 方法 = value baseline（advantage = reward − V(s)）+ efficiency floor + budget-init 锚定。

### v9（教训）
- 🔴 单 knob 超参到头：gate 0.895413 但 full 回归 0.894345（< v7 0.894798）。

## v12 建议方向（基于 v11 归因，优先级降序）

v11 已把差距压到 -0.000126（噪声量级）。瓶颈是**正则强度**与**策略表达能力**，不是分布。

1. **沿 kl 方向继续扫（最低风险、性价比最高）**：hard50_kl（kl 0.02）明显优于 kl 0.01。
   直接试 **kl 0.03 / 0.05 / 0.08**，保持 50/50 混采 + warm-start 自 hard50_kl。
   依据：本轮唯一被证明「只涨不跌」的 knob，且 606 vs 639 说明正向收益还有回收空间。
2. **条件化 kr**：budget head 目前只用 mean-pooled 特征出一个标量。让 kr 显式条件于风险信号
   （baseline 子分数、障碍物密度、attention 熵），使策略能区分「该多留」与「可照常剪」，
   直击「全局抬 kr 是钝器」问题。
3. **per-token 保护**：对灾难场景做 counterfactual（`--counterfactual-k` 已实现，本轮关闭），
   定位关键少数 token 加以保护，而非整体少剪。
4. **专攻 sh2**：两个 arm 的 min shard 都是 sh2（0.8930~0.8935），其余 shard 均已超 SOTA。
   sh2 单独诊断可能直接暴露某类失败模式。

## 已完成（倒序）

- [x] 08-27 19:06 hard50_kl full 完成（**0.898719，-0.000126，历史最佳**）+ 逐场景归因
- [x] 08-27 14:28 hard50 full 完成（0.898360，-0.000485）
- [x] 08-27 GitHub 发布：代码 + `release_ckpt/`（10 个权重 58MB）+ 此前未跟踪文档
- [x] 08-27 10:46 v11 gate 完成（4 臂，hard50 gate 最高但 full 输给 hard50_kl）
- [x] 08-27 03:34 v11 训练完成（4 臂 warm-start 自 full_hi）
- [x] 08-27 01:20 v11 mining 完成（navtrain 8190，200 灾难场景占 91.7% 负 delta）
- [x] 08-26 代码：`--mine-mode` / `--scene-list` / `--init-budget-ckpt`；orchestrator 全流程
- [x] 08-26 09:29 v10 r1 full（full_hi 0.898274）
- [x] v9 r1（full 回归 0.894345）；v7 full（st_topk 0.894798）

## 待办（优先级降序）

1. [ ] **v12：kl 扫描（0.03/0.05/0.08），从 hard50_kl 续训** ← 最高性价比
2. [ ] v12 备选：条件化 kr（见建议方向 2）
3. [ ] sh2 专项诊断（唯一未超 SOTA 的 shard）
4. [ ] 修 `profile_wallclock.py`（latency 仍未验收）
5. [ ] 把 hard50_kl 的 full 结果补进 `release_ckpt/README.md`

## 决策记录

| # | 时间 | 决策 | 选择 | 理由 | reverse | 确认 |
|---|---|---|---|---|---|---|
| 1 | 08-18 | 主线方法 | A(可微Top-K)+C(delta reward) | 用户拍板 | 回退 softmax | ✅ |
| 2 | 08-18 | A 实现选型 | st_topk + gumbel 都做 | 用户拍板 | — | ✅ |
| 3 | 08-19 | winner | st_topk | min-shard 最高且超 SFT | 改 tau1 | ✅ full 确认 |
| 4 | 08-25 | 方法级改动 | value baseline + kr 上浮 | v9 单 knob 到头 | — | ✅ 追平 SOTA |
| 5 | 08-26 | v11 方向 | hard-example mining（改分布） | 灾难场景 2.4% 占 91.7% 负 delta | 回退 full_hi | ⚠️ gate 反超 full 没有 |
| 6 | 08-26 | 训练分布 | hard+normal 混采 frac 0.5 | 保 efficiency 锚 | 调 frac | ✅ hard75 更差已验证 |
| 7 | 08-27 | v11 归因 | 瓶颈=全局 kr 是钝器；**kl 正则才是有效 knob** | hard50_kl 靠「少丢正向收益」拿到 -0.000126 | — | ✅ 逐场景数据支撑 |

## 卡点 / 风险

- 🔴 **gate ≠ full（已第三次）**：v9、v11 均 gate 乐观。**gate 最高的 hard50 反而 full 输给 hard50_kl**
  （gate 0.899520 vs 0.898909，full 0.898360 vs 0.898719，排序完全反转）。任何结论必须 full 4-shard 定论。
- 🔴 **不得据训练 reward / kr 选 winner**：hard75 reward 最好 gate 最差；hard50_kl kr 不动 full 最好。
- 🔴 **latency 未验收**：`profile_wallclock.py` 已修 4 处（hydra 配置名/目录、SceneLoader 签名、
  sensor 路径改用 `$OPENSCENE_DATA_ROOT`、改直读 navtest json + 单场景容错），
  但 agent 推理仍在 `decode_token_ids_to_trajectory` 全场景失败。**efficiency 目前只有 kr + 理论 FLOPs**。
- 🔴 **`data/navsim_v2_local` 不存在**：多脚本（含 eval/train 的 `SENSOR` 变量）指向该路径，
  说明该参数在 eval 链路未被真正使用；真实 sensor 在
  `$OPENSCENE_DATA_ROOT/sensor_blobs/test/openscene-v1.1/sensor_blobs/test`。
- 🔴 **eval worker>1 有 race（H20）**：必须 `EVAL_WORKERS=1`。
- ⚠️ **本节点仅 4 张 H20**：gate 8 job 分 2 波，单波约 3.7h；full 单臂约 3.7h。
- ⚠️ **sh2 是唯一短板**：两 arm 的 sh2 均 0.893 左右，其余 shard 都已超 SOTA。

## 关键路径（⚠️ 节点已回收，以下为**仓库内**路径，本地绝对路径已失效）

- **复现全流程**：`docs/REPRODUCE.md`（每步附验证命令，已实测跑通）
- **环境**：`requirements.txt` + `env.example.sh`（cp 成 env.sh 后 source）
- **AutoVLA 侧改动**：`autovla_overlay/`（patch + 配置；AutoVLA 是学术许可的嵌套仓库，未随库分发）
- **训练好的权重**：`release_ckpt/`（10 个，58MB；最佳 = `v11_hard50_kl`）
- **mining 结果（v12 直接复用，省 2.6 GPU-hours）**：`artifacts/v11/mining/`（8190 场景）
- **评估原始 CSV**：`artifacts/v11/eval_full/`、`artifacts/v11/eval_gate/`
- mining 脚本：`scripts/run_v11_mining_4gpu.sh [n_scenes]`
- 组装场景表：`scripts/build_v11_scene_list.py --hard-frac 0.5 --hard-repeat 2`
- 训练：`scripts/run_v11_hardmine_4gpu.sh <list50> <list75> [round]`
- 评估：`EVAL_WORKERS=1 EVAL_GPUS="0 1 2 3" bash scripts/eval_v7_folds_4gpu.sh <CYCLE_ID> [gate|full] [arms...]`
- 训练核心：`scripts/train_scorer_budget_rl.py`（`--mine-mode` / `--scene-list` / `--init-budget-ckpt`）
- 无人值守流水线：`scripts/orchestrate_v11_pipeline.sh`
- 效率：`scripts/compute_flops_table.py`（可用）、`scripts/profile_wallclock.py`（**待修**）

⚠️ `scripts/*.sh` 均硬编码原节点绝对路径（`/apdcephfs/private_shayladeng/...`），
新机器需逐个改 header 前 ~15 行。详见 `docs/REPRODUCE.md` 开头三条警告。

## GitHub

- 主仓库：`Yanyeoo/RL-Drive-Pruning`（本项目，HEAD 已含全部产物）
- 前序仓库：`Yanyeoo/RL-Drive`（legacy ReCogDrive 线，已 freeze，无待提交）
- 未上传（有意）：140G 第三方预训练模型（AutoVLA/Qwen/ImpromptuVLA，官方 HF 已有）、606G 数据集
