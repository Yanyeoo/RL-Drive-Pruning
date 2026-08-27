# Progress Log

## 2026-08-14 — RL SOTA H20 周期启动

- 用户固化不可更改总目标：**在当前 RL idea 无大改的情况下达到 SOTA 且具备 efficiency**。
- 读取 `AGENT_RULES.md` 后完成 GPU、进程、磁盘、checkpoint、日志和训练评估链核验。
- 从训练代码确认并修复四个确定性问题：budget log-prob 被 detach 导致预算策略无梯度；`efficiency_beta` 未进入 reward；counterfactual group 仅优化最后 scene；token loss 重复包含 budget log-prob。
- 新增可选 same-scene `--delta-reward`，不改变 Gaussian budget REINFORCE 主体。
- 构建 `scripts/run_rl_sota_cycle_8gpu.sh`：8臂并行筛选 → 固定 held-out 门控 → 胜者8卡同步训练 → full-navtest 8分片评估。
- 构建 `scripts/finalize_rl_sota_cycle.sh`，并安排 2026-08-15 09:50 自动收尾与交付。
- 修改前备份：`backup/20260814_162603_rl_sota_cycle/`。

## 2026-08-15 — 周期结果与收尾

- 8臂短训 + 固定198-scene门控最高 `0.898963`，后续 full-navtest 证明该门控有严重 selection bias。
- 8卡同步长训 `delta_eff02` 完成128 steps；full N=11576 PDMS `0.873472`。
- 四个短训候选均完成full-navtest：最优 `delta_eff0` PDMS `0.874271`、mean keep ratio `0.5302`；其余为 `delta_eff01=0.874189`、`delta_eff02=0.873486`、`delta_eff005_std2=0.873001`。
- 当前总目标尚未达成；完整交付：`docs/journal/2026-08-15_rl_sota_handoff.md`。
- 09:43执行安全收尾，本周期GPU任务均退出，8卡显存归零。
