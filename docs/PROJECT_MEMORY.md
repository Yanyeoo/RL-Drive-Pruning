# DriveToken / RL-Drive-Pruning 项目记忆与交付文档

> 写给下一个 AI / 协作者。最后更新: 2026-08-06 17:33。
> ICLR 2027 ddl: 2026-09-25。目标: 追求 SOTA 更高 PDMS。

---

## 一、项目概述

**目标**：训练一个 token scorer + budget head，在 Qwen2.5-VL-3B VLA 推理时动态剪枝 vision tokens，在保持驾驶质量的前提下减少计算量。

**核心方法演进**：
- Stage 1 (SFT): LambdaRank 训练 scorer，对 720 个 vision tokens 打分排序
- Stage 2 (RL): REINFORCE 训练 budget head（Gaussian policy 决定 kr） + token_net（Top-K selection）

**关键数据**：
- 训练: `data/navtrain_nocot/` (19225 scenes)
- 测试: `data/navtest_nocot/` (11596 scenes, 4 shard)
- VLA: Qwen2.5-VL-3B, checkpoint at `models/AutoVLA/AutoVLA_PDMS_89.ckpt`
- Vision tokens: N=720 (3 cameras × ~240 tokens/cam)
- Python: `/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python` (3.9.23)
- GPU: 4×H20, 每周 320 卡时

---

## 二、所有实验结果汇总

| 方法 | PDMS | 场景数 | vs SFT(0.89008) | 说明 |
|---|---|---|---|---|
| no-prune (r=1.0) | **0.89886** | — | +0.00878 | 理论上界 |
| SFT scorer r=0.5 | **0.89008** | — | 0 | 基线 |
| SFT scorer r=0.355 | 0.85575 | — | -0.03433 | matched-compute baseline |
| v3 Budget RL (eff_beta=0.15, proxy reward) | 0.85429 | 2949(sh0) | -0.03579 | AAAI投稿版本 |
| v4 Budget RL (eff_beta=0.05, proxy reward) | 0.86094 | 11576(全量) | -0.02914 | 降低效率惩罚 |
| v5 True PDMS RL (无eff_bonus) | 0.84850 | 11576(全量) | -0.04158 | True PDMS reward |
| v5 BEST (ckpt_best, sh0) | 0.87667 | 2949(sh0) | -0.01341 | v5最优checkpoint |
| **v6 Per-Token CF REINFORCE** | **训练中** | — | — | 当前运行 |

**退化轨迹**：v3(0.87) → v4(0.86) → v5(0.85) → Scene-level RL 天花板 ~0.87

---

## 三、当前运行状态 (v6)

### 训练
- **版本**: v6 Per-Token Counterfactual REINFORCE
- **PIDs**: 157318, 157321, 157324, 157326
- **启动时间**: 2026-08-06 16:24
- **进度**: step 35, ~94s/step
- **ckpt**: `ckpt/s3_token_scorer_budget_rl_v6_20260806_162439_sh{0..3}/`
- **日志**: `logs/v6_train/train_sh{0..3}.log`
- **配置**: counterfactual_k=4, group_size=8, 1 epoch, efficiency_beta=0, driving_scale=3.0, safety_beta=0.03
- **R**: mean=2.28 (≈PDMS 0.76), kr: 0.55, grad: 74%非零

### Auto-chain
- **PID**: 157920
- **脚本**: `scripts/auto_chain_v6.sh`
- **行为**: 轮询训练PIDs，完成后自动 eval→消融→7B nuScenes→汇总

### 预计时间线
```
~22:00  训练完成 (600 steps)
~01:30  v6 全量 eval 完成
~05:00  消融完成
~18:00  GPU回收
```

---

## 四、关键文件索引

### 训练脚本
| 文件 | 用途 |
|---|---|
| `scripts/train_scorer_budget_rl.py` | **主训练脚本** (v6 per-token counterfactual) |
| `scripts/launch_v6_train.sh` | v6 4卡启动 |
| `scripts/auto_chain_v6.sh` | v6 无人值守全链 |

### ICLR 计划
| 文件 | 内容 |
|---|---|
| `iclr/PLAN.md` | ICLR 2027 转投计划 (方法创新+实验+时间线) |
| `aaai/PLAN.md` | AAAI 2027 补充材料计划 |

### Handoff / Journal
| 文件 | 内容 |
|---|---|
| `docs/journal/2026-08-06_v6_handoff.md` | v6 本次handoff |
| `docs/journal/2026-08-05_v5_handoff.md` | v5 handoff |
| `docs/journal/2026-08-04_handoff.md` | v4 handoff |
| `docs/journal/HANDOFF_2026-07-27_SAFE_HTPO_UNATTENDED.md` | 操作规则 (重要!) |

### 备份
| 目录 | 内容 |
|---|---|
| `backup/20260806_v6_launch/` | v6 代码改动前备份 |
| `backup/20260805_v5_launch/` | v5 代码备份 |
| `backup/20260803_prefactcheck/` | factcheck 前备份 |

### 核心代码
| 文件 | 说明 |
|---|---|
| `code/rldrive/scoring/token_scorer_budget.py` | TokenScorerWithBudget 模型定义 |
| `code/third_party/AutoVLA/models/utils/score.py` | PDM_Reward / rl_pdm_score |
| `code/rldrive/agents/token_prune_patch.py` | Variant-A (attn mask) prune |
| `code/rldrive/agents/token_prune_patch_varB.py` | Variant-B (physical drop) prune |

---

## 五、方法版本说明

### v3-v5: Scene-Level Budget RL
- **机制**: 一个 scene 一个 reward → 所有 token 共享同一个 advantage
- **问题**: 无法区分哪些 token 重要哪些不重要
- **结果**: PDMS 持续退化，天花板 ~0.87

### v6: Per-Token Counterfactual REINFORCE (当前)
- **机制**: 
  - 每个 scene: baseline forward(R_base) + pruned forward(R_pruned) + K次counterfactual
  - 对每个采样token: swap out → re-evaluate PDMS → A_i = R_pruned - R_{-i}
  - token_net 用 per-token advantage 更新
  - budget_net 保持 Gaussian REINFORCE
- **计算量**: 每scene 2+K=6次VLA forward (vs v5的2次)
- **关键参数**: --counterfactual-k 4, --group-size 8

### ICLR PLAN 后续方向
1. **§2.1**: 7B ViT特征跨模型迁移 (scorer emb_dim: 2048→1280)
2. **§2.2**: Per-token counterfactual REINFORCE (v6, 进行中)
3. **§3.1**: 完整消融矩阵 + multi-seed

---

## 六、操作规则 (来自 HANDOFF_2026-07-27)

1. **每个决策前先做事实核验** — 从代码/产物/日志验证，不凭记忆
2. **起进程前检查遗留进程** — `ps aux | grep python` 避免双开
3. **关键artifact操作前备份** — `cp -a`，不动原文件
4. **任何偏离PROMPT/design doc的决策当场写journal** — 附理由+reverse指令
5. **实时更新todo，完成一项立即mark**

---

## 七、GPU 18:00回收前检查清单

明天下午检查:
- [ ] `results/raw/SOTAV6_R1_FINAL_sh*.csv` — v6 全量 PDMS
- [ ] `results/raw/SOTAV6_R1_BEST_sh0.csv` — v6 best ckpt
- [ ] `results/raw/SOTAV6_R1_SAFENET_sh0.csv` — v6 + safety_net
- [ ] `results/raw/SOTAV3_R1_FINAL_sh0.csv` — v3 对比
- [ ] `logs/auto_chain_v6/final_report.json` — 汇总报告
- [ ] `ckpt/s3_token_scorer_budget_rl_v6_20260806_162439_sh*/` — v6模型

### 结果解读
- v6 PDMS > 0.892: ✅ ICLR核心实验成立
- v6 PDMS ∈ [0.88, 0.892]: ⚠️ 接近，需进一步优化
- v6 PDMS < 0.88: ❌ per-token方向也未奏效，需根本性方法改变

---

## 八、Reverse指令

```bash
# Kill v6训练
kill 157318 157321 157324 157326
# Kill chain
kill 157920
# 清理v6产物
rm -rf ckpt/s3_token_scorer_budget_rl_v6_20260806_162439_*/
rm -rf logs/v6_train/
rm -rf results/raw/SOTAV6_*
```

---

## 九、GitHub

- **Repo**: `https://github.com/Yanyeoo/RL-Drive-Pruning`
- **Remote**: origin (已配置token)
- **当前分支**: main
- **未提交改动**: `scripts/train_scorer_budget_rl.py`, `autovla_with_token_prune.py`, 等
