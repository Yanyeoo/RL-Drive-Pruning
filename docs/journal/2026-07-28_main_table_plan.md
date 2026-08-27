# 主表结构 & 执行计划 — 2026-07-28（对标 07-21，更新为全 drop + 实测数字）

> 全部方法均 **Variant B true token drop**（`+agent.prune_variant=drop`）。
> Baseline 报 **raw PDMS**；**(ours)** 报 **+fallback**（denylist 命中场景→no-prune 分，
> 768 tokens；已核验精确复现 0.9045 headline）。
> 数字来源：`results/raw/tokenprune_S3_full/`（drop CSV）实测聚合，非记忆。

---

## 一、Table 1 设计（最终）

```
Table 1: Performance comparison on NAVSIM (AutoVLA-3B, closed-loop).
All methods use true token removal (Variant B).
```

列：**Method | Tokens↓ | PDMS↑ | NC↑ | EP↑ | Rel.(%) | FLOPs↓**
分组：按 retention（540/360/180 tokens）+ Dynamic。
- NC = no_at_fault_collisions；EP = ego_progress；Rel.% = PDMS / 0.89879。
- FLOPs↓（LLM prefill）：r=0.75→16.9%；r=0.5→33.6%；r=0.25→49.9%（§12.6 Pareto）。

---

## 二、当前实测值 & 缺口（2026-07-28 21:50 事实核验）

Baseline no-prune = `MT_attn_L12_r10` = **0.89879** (N=11576, drop)。

### Retain 540 (r=0.75, ↓25%)
| Method | PDMS | 覆盖 | 状态 |
|---|---|---|---|
| FastV | ❌ | 现有仅 **mask** | **需跑 drop** |
| Random | ❌ | — | 需跑 drop |
| PruMerge | ❌ | — | 需跑 drop |
| SparseVLM | 0.89908 | full | ✅ 现成 (`MT_sparsevlm_text_r075`) |
| SFT Scorer (ours) | **0.89462** | full +fb | ✅ (`MT_varBsafe_scorer_r075`) |
| RL Scorer (ours) | 0.8945 | sh0/1/2 → **补 sh3 中** | 🔄 A 阶段 |

### Retain 360 (r=0.5, ↓50%)
| Method | PDMS | 覆盖 | 状态 |
|---|---|---|---|
| FastV | ❌ | 现有仅 mask (0.833) | 需跑 drop |
| Random | 0.86349 | full | ✅ 现成 (`MT_random_r05`) |
| PruMerge | ❌ | — | 需跑 drop |
| SparseVLM | ❌ | — | 需跑 drop |
| SFT Scorer (ours) | **0.90452** | full +fb | ✅ **headline** (`MT_varBsafe_scorer_r05`) |
| RL Scorer (ours) | 0.89279 | full +fb | ✅ (`MT_rl_shaped_r05`) |

### Retain 180 (r=0.25, ↓75%)
| Method | PDMS | 覆盖 | 状态 |
|---|---|---|---|
| FastV | ❌ | — | 需跑 drop |
| Random | ❌ | — | 需跑 drop |
| PruMerge | ❌ | — | 需跑 drop |
| SparseVLM | ❌ | — | 需跑 drop |
| SFT Scorer (ours) | ❌ | — | 需跑 (`MT_sft_varB_drop_r025`) |
| RL Scorer (ours) | 0.8264 | full +fb | ✅ (`MT_rl_shaped_r025`) |

### Dynamic
| Method | PDMS | 覆盖 | 状态 |
|---|---|---|---|
| SFT + τ-cut (ours) | **0.8949** | full +fb (MSE kr060) | ✅ (`TC_mse_tau_kr060`) |
| RL + τ-cut (ours) | 0.7572 | sh0 → **补 sh1/2/3 中** | 🔄 A 阶段 |
| Budget RL (ours) | 0.8701 | full +fb | ✅ (`MT_budget_rl_dynamic`) |

> 实时草表：`results/table1_draft.md`（每完成一格自动重生成）。

---

## 三、关键发现 / 与 07-21 的偏差

1. **RL Scorer 全线弱于 SFT**（drop full-navtest 实测）：RL r=0.25=0.826 / r=0.5=0.893
   vs SFT +fb r=0.5=0.905。与 `key_results.md §13.4` 用户裁定（保留 RL 主线、不重写）一致。
2. **FastV 现有 CSV 是 mask 口径**（`run_fastv_baseline.sh` 未写 drop），全 drop 主表下 FastV 三格必须重跑。
3. **+fallback 已核验**：`MT_varBsafe_scorer_r05` raw 0.87253 → +fallback **0.90452**（768 命中），
   精确复现 headline。ours 各行统一用此 post-hoc 口径。
4. **PruMerge 不进主表**？→ 本次按用户 07-28 最新指示 **重新纳入**（`prumerge_cls` selector 已实现，
   §appendix 归类作废）。

---

## 四、需要补跑的实验（全 drop）

分片：sh0=2950/sh1=2797/sh2=2964/sh3=2869。单分片 ~3.3h；shard0 ~1h。

### A 阶段（进行中，GPU0-3 并行，21:22 起，ETA ~00:45）
- RL Scorer r=0.75 → sh3
- RL + τ-cut kr060 → sh1/sh2/sh3

### B 阶段（shard0 占位，A 后自动启）— 11 格
| selector | ratio | exp |
|---|---|---|
| fastv_l2 | 0.25/0.5/0.75 | MT_fastv_l2_drop_r{025,05,075}_sh0 |
| random | 0.25/0.75 | MT_random_drop_r{025,075}_sh0 |
| prumerge_cls | 0.25/0.5/0.75 | MT_prumerge_cls_drop_r{025,05,075}_sh0 |
| sparsevlm_text | 0.25/0.5 | MT_sparsevlm_text_drop_r{025,05}_sh0 |
| scorer (ours+fb) | 0.25 | MT_sft_varB_drop_r025_sh0 |

（random r=0.5 与 sparsevlm r=0.75 已有 full drop，不重跑。）

### C 阶段（best-effort 7B，B 后）
- ImpromptuVLA-7B + nuScenes zero-shot，用已训好的 `ckpt/s3_token_scorer_7b`。
- 之前 image_processor blocker 已在 7/23 修复（finetune 目录 `preprocessor_config.json` 含
  `image_processor_type`）。脚本先 smoke，通过则 4 卡并行跑 r=1.0/0.75/0.5/0.25 → step2 metrics。
- 失败即写报告、不阻塞、不改协议。报告：`results/impromptu7b/BESTEFFORT_REPORT.md`。

### B2 阶段（剩余时间，把 B 的 shard0 升级到 full 4 分片）
- 优先级 r=0.5 → r=0.75 → r=0.25，sh1/2/3，deadline 硬停。

---

## 五、执行 / 无人值守

- 编排脚本：`scripts/run_maintable_overnight.sh`（A 等待 → B → C → B2 → finalize）。
- 硬停：**2026-07-29 13:45**（14:00 回收前 15min）。每格完成即重生成 `results/table1_draft.md`，
  中途被杀也留有最新草表。
- 幂等：已存在 CSV 自动 SKIP；单实例锁 `logs/maintable_overnight.lock`；急停 `touch STOP_MAINTABLE`。
- 表生成器：`scripts/gen_table1_draft.py`（baseline raw / ours +fallback / 覆盖标注）。

*记录时间：2026-07-28 21:52。数据均经 CSV 行数与聚合实测核验。*
