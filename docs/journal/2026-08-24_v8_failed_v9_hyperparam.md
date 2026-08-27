# 2026-08-24 v8 规模消融失败 → v9 超参微调（回 512 甜点位冲 SOTA）

## 接手时状态（08-24 20:38）

- 所有 8×H20 空闲，v8 周期已停 4 天无人接管。
- v8 规模消融（CYCLE_ID=20260819_scale，st_topk 超参不变，只动数据量/步数）：

| arm | scenes | ep | 状态 | gate mean | vs SFT(0.89199) |
|---|---|---|---|---|---|
| A st_topk_s512 | 512 | 1 | ✅ 训完（checkpoint.pt 在，**从未 eval**） | 待补跑 | — |
| B st_topk_s2000 | 2000 | 1 | ✅ 训完 + gate | **0.88955** | **−0.0024 ❌** |
| C st_topk_s4000 | 4000 | 1 | ✅ 训完 + gate | **0.89113** | **−0.0009 ❌** |
| D st_topk_s4000_e3 | 4000 | 3 | 🔴 707/750 步中断（无 checkpoint.pt，仅 ckpt_step700） | 未 eval | 死路 |

## 关键结论（v8 实锤）

- **加数据量 512→2000/4000 有害**：s2000/s4000 均跌破 SFT 0.89199，远差 v7 winner（gate 0.894656）。
- **数据多样性是毒药**：v7 根因判断「训练不足 + 数据多样性不足」→ 加数据反噬。
- **512 scenes / 64 steps 是唯一甜点位**。规模这条路彻底关闭。
- 因此冲 SOTA 只能走「甜点位内单 knob 微调」。

## v9 计划（CYCLE_ID=20260824_v9r1，4×H20，~38min 训练）

回到 v7 winner（st_topk tau=0.1 lr=3e-5 pg_w=1.0 kl=0.01 ep=1 group=8）单 knob 变化：

| arm | knob | 假设 |
|---|---|---|
| st_topk_ep2 | num-epochs 1→2（128 步，同 512 scenes） | 训练更充分（非加数据） |
| st_topk_tau005 | selection-tau 0.1→0.05 | st_topk sigmoid 更锐利，梯度更干净 |
| st_topk_pgw2 | selection-pg-weight 1.0→2.0 | token 选择梯度更强 |
| st_topk_kl005 | kl-beta 0.01→0.05 | token_net 更强锚定 SFT 防漂移 |

评估协议不变：gate（shard0+shard1，8 卡并行）快筛 → full（4-shard）定论 winner。

## 时间账（8 卡，deadline 08-25 11:00）

- Round1 训练 ~0.6h → gate（8 卡）~2h → 分析 → Round2 训练 ~0.6h → gate ~2h → full ~2h ≈ 7.2h。留足余量可跑 3 轮。

## 已启动（08-24 20:41）

1. Round1 训练：`run_v9_hyperparam_sweep_4gpu.sh r1 0`（GPUs 0-3）。
2. v8 s512 复现 eval：`eval_v7_folds_4gpu.sh 20260819_scale gate st_topk_s512`（GPUs 4-5，验证 pipeline 一致性）。
