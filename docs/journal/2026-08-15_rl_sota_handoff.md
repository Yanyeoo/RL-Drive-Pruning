# RL SOTA 周期交付（自动收尾）

> 收尾时间：2026-08-15T09:43:39 ；周期：`20260814_210502`

## 不可更改的总目标

**在当前 RL idea 无大改的情况下达到 SOTA 且具备 efficiency。后续 AI 不得降低、替换或改写该目标。**

验收至少包括：RL PDMS 超过当前 learned SFT `0.89199`；继续冲击整体上界/SOTA 约 `0.89879~0.899`；同时报告真实动态 keep ratio / FLOPs / latency。

## 本周期关键代码修复

- `scripts/train_scorer_budget_rl.py`：修复 `budget_log_prob.detach().item()` 导致预算策略完全无梯度。
- 修复 `efficiency_beta` 从未进入 reward 的确定性 bug。
- 修复反事实训练仅使用 group 最后一个 scene loss 的 bug。
- 普通 token PG 不再重复包含 budget log-prob；新增 same-scene `--delta-reward` 降方差。
- 策略仍是原 Gaussian budget REINFORCE + SFT token scorer，没有大改 RL idea。

## 自动周期状态

- 周期目录：`logs/rl_sota_cycle_20260814_210502`
- checkpoint 根目录：`ckpt/rl_sota_cycle_20260814_210502`
- sweep 胜者：`delta_eff02`
- 最终训练输出：`/apdcephfs/private_shayladeng/tokenrl_autoVLA/ckpt/rl_sota_cycle_20260814_210502/final_delta_eff02`
- full-navtest 聚合：**N=11576, PDMS=0.873472**

## 200-scene 同集筛选结果

| 模型 | N | PDMS |
|---|---:|---:|
| `RL8_20260814_210502_delta_eff005_std2_s200.csv` | 198 | 0.898963 |
| `RL8_20260814_210502_delta_eff02_s200.csv` | 198 | 0.898818 |
| `RL8_20260814_210502_delta_eff01_s200.csv` | 198 | 0.898588 |
| `RL8_20260814_210502_delta_eff0_s200.csv` | 198 | 0.889601 |
| `RL8_20260814_210502_delta_eff002_s200.csv` | 198 | 0.889590 |
| `RL8_20260814_210502_delta_eff005_s200.csv` | 198 | 0.889384 |
| `RL8_20260814_210502_abs_eff0_s200.csv` | 198 | 0.879891 |
| `RL8_20260814_210502_delta_eff005_lr3_s200.csv` | 198 | 0.860587 |

## 候选checkpoint full-navtest结果

| 模型 | N | PDMS | mean keep ratio |
|---|---:|---:|---:|
| `delta_eff0` | 11576 | 0.8742706058860858 | 0.5302044747753943 |
| `delta_eff01` | 11576 | 0.8741894146630371 | 0.5294253628196253 |
| `delta_eff02` | 11576 | 0.8734856250042203 | 0.5256974775397348 |
| `delta_eff005_std2` | 11576 | 0.8730009966714646 | 0.5352834312370426 |

## 下一 AI 必做顺序

1. 先读 `docs/PROJECT_MEMORY.md`、本文件、周期 `orchestrator.log`、`sweep_results.json`、`final_result.json`。
2. 核验 checkpoint 可加载、训练 `grad_norm` 非零、budget mean/方差确实变化；不要把训练 reward 当 navtest PDMS。
3. 若 full eval 未完成，使用最终 checkpoint 续跑缺失的 `rlcycle_*_g0..g7` 分片并去重聚合。
4. 若 PDMS 未超 `0.89199`，保持原 RL idea，优先做低风险参数/方差/模型选择修正；不得转成纯 SFT/DPO 或放弃 efficiency。
5. 最终必须补全动态 keep-ratio、FLOPs/latency 和 full-navtest N≈11576，只有同时满足性能与效率才能宣称达成。

## 本周期结论与最优先下一步

- **结论：确定性训练 bug 已修复，但本周期仍未达成总目标。** 最强 full-navtest 为 `delta_eff0`：PDMS `0.874271`、mean keep ratio `0.5302`，仍低于 SFT `0.89199`。
- 固定 198-scene 门控给出 `0.8986~0.8990`，但四个候选 full-navtest 全部只有 `0.8730~0.8743`，证明该门控严重 selection bias，后续不得再据此选择 checkpoint。
- 8卡同步长训把 `delta_eff02` 从短训 full `0.873486` 变成 `0.873472`，基本无改善；继续加训练步数不是首选。
- **下一步应保持当前 RL idea，仅改变验证/模型选择：** 用 4 个互斥且覆盖难场景的约 500-scene validation folds，按跨 fold 均值与最差 fold 选 checkpoint；同时把 budget 更新限制在接近 SFT 最优工作点（约 r=0.5）的窄 trust region，防止动态预算在长尾场景退化。
- 需要优先比较 `SFT scorer` 与 `delta_eff0` 的逐 token full-navtest CSV，定位造成约 `−1.77 pt` gap 的 catastrophe scenes；在原 reward 内增加这些场景的 safety penalty/采样权重，不更换 RL 方法。

## 回滚

- 核心训练脚本修改前备份：`backup/20260814_162603_rl_sota_cycle/train_scorer_budget_rl.py`。
- 停止当前链：仅终止命令行含本周期 ID 的进程；不要影响其他项目进程。
