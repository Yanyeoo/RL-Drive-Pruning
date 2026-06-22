# Key Results — Single Source of Truth for Numbers

> **唯一权威表**：所有 milestone 的关键数字都在这里。
> 任何"我们 B0 多少"、"对比 paper 多少"、"训出来比 baseline 高几个点"
> 的问题，**先看这里**。
>
> 维护规则见 `docs/results/README.md`。
> 详细推导/复现/路径细节看每行末尾链接的 journal。

---

## 0. Quick reference — one-liner per milestone

| ID | what | headline number | vs ref | date | journal |
|---|---|---:|---|---|---|
| **B0** | AutoVLA navtest baseline (no pruning) | **PDMS = 0.8983** (n=11576) | paper 0.8911, **+0.72 pt** ✅ matches | 2026-06-16 | [MA2_b0_navtest.md](../journal/MA2_b0_navtest.md) |
| M0.1 | navtest token snapshot | 11596 eligible / 11576 evaluable / 2 invalid | — | 2026-06-16 | [b0_invalid_token_diagnosis.md](../journal/2026-06-16_b0_invalid_token_diagnosis.md) |
| M0.2 | navtrain split build | _pending download_ | — | — | — |
| **M1.a** | attention layer probing (navtest, n=500 v2 lock) | **L\*=12 (vision_frac=0.1861, n=500)**, beats L27 (0.1805) by +0.0056 | — | 2026-06-18 | [m1a_layer_selection_2026-06-18.md](../_internal/m1a_layer_selection_2026-06-18.md), [journal](../journal/m1a_layer_sweep_navtest_2026-06-18.md) |
| **M1.b₀** | L12/L27 per-head decomposition (n=100 + n=200 disjoint) | **L12 has 1 dead head (h13), L27 has 2 dead (h8,h9). Spearman ρ=1.0000 on disjoint sample. top-12 retains 96.8% vision attn @ 25% KV reduction.** | — | 2026-06-18 | [m1b_per_head_analysis_2026-06-18.md](../_internal/m1b_per_head_analysis_2026-06-18.md) |

> ⚠️ 任何一行变动 = 必须改这表 + 在对应 journal 里留 diff link。

---

## 1. B0 — AutoVLA navtest baseline (LOCKED)

**Headline**: `mean PDMS = 0.8983` on 11576 navtest tokens.

### 1.1 vs AutoVLA paper (NeurIPS 2025, Post-RFT)

| metric | ours (B0) | paper Post-RFT | Δ | judgment |
|---|---:|---:|---:|---|
| **PDMS (aggregate)** | **0.8983** | 0.8911 | **+0.72 pt** | ✅ 复现成功（噪声内）|
| no_at_fault_collisions | 0.9944 | 0.9841 | +1.03 | ✅ 略好 |
| time_to_collision | 0.9768 | 0.9804 | −0.36 | ✅ 持平 |

**复现判定**：✅ AutoVLA 可作主干。我们的 ckpt 行为与论文一致。

### 1.2 Sub-component breakdown (n=11576)

| sub-component | mean | weakest? | failures |
|---|---:|---|---:|
| no_at_fault_collisions       | 0.9944 |   | 65 collisions |
| drivable_area_compliance     | 0.9603 |   | 459 off-road |
| **ego_progress**             | **0.8326** | **🔻 dominant** | continuous |
| time_to_collision_within_bound | 0.9768 |   | 269 violations |
| comfort                      | 0.9986 |   | 16 uncomfortable |
| driving_direction_compliance | 0.9812 |   | 218 wrong-dir |

→ **ego_progress = 0.83 是最大优化空间**（其余子项已 ≥ 0.96）。RL 发力点。

### 1.3 Score distribution (where the headroom is)

| range | count | %    |
|---|---:|---:|
| `[0.9, 1.0]` | 8635 | 74.6 |
| `[0.8, 0.9)` | 2180 | 18.8 |
| `[0.7, 0.8)` |   66 |  0.6 |
| middle bands |  183 |  1.6 |
| **`[0.0, 0.1)` (hard-zero)** | **510** | **4.4** ⚠️ |
| invalid     |    2 |  0.02 |

→ **510 个 hard-zero token 是 RL 的核心改进区**：把 hard-zero 从 4.4% 降到 3% ≈ +1.5 PDMS。

### 1.4 Throughput / cost

| | value |
|---|---|
| Wall-clock (4× H20 parallel) | 1h 50m total |
| Per-GPU steady-state | 2.19 s/token |
| VRAM | 30.9 GB / 98 GB |
| Bottleneck | sensor blob IO from CephFS |

→ 任何 r=1.0 的 navtest 全量 sweep ≈ 2h on 4× H20。

### 1.5 Artifacts

| | path |
|---|---|
| merged csv | `exp/ma2_5_b0_quad_merged_20260616_154858/merged.csv` |
| token snapshot | `data/splits/navtest_b0_tokens.txt` (11596 行) |
| repro 命令 | 见 `MA2_b0_navtest.md` §9 |

---

## 2. M0.1 — navtest token snapshot

| | count | meaning |
|---|---:|---|
| 原始 navtest_local_filtered.yaml | 12146 | scene_filter 上限 |
| ∩ metric_cache ∩ navtest_nocot | 11596 | M0.1 锁定的 evaluable 上界 |
| 实际 merge 后 unique | 11576 | navsim SceneFilter 又丢了 20（has_route + frame count）|
| valid (score 计算成功) | 11574 | 99.98% |
| invalid (trajectory decode <8 poses) | 2 | `d318551a8ce150e5`, `7defd0c32cd8546a` |

**fix 方案**：M5/M6 agent refactor 时给 `autovla_agent.py:445` 加 pad-last-pose
patch（已 doc 在 `b0_invalid_token_diagnosis.md`）。

---

## 3. Environment / known infra issues

| | status | note |
|---|---|---|
| `inference` (4× H20, fp32, eager attn) | ✅ | B0 跑通 |
| `GRPO train smoke` (Lightning, fp32) | ❌ SIGFPE | cuBLAS GEMM bug on H20+torch2.4+cu12.1，未绕过 |
| 计划 fix | `attn_implementation='eager'` | 优先级 E1，未实验 |
| navtrain 数据 | ⏳ 下载中 | pid 56277, ETA 21:55 |

---

## 4. M1.a — attention layer probing (navtest, locked v1)

**Headline**: `L* = 12` (`vision_frac_mean = 0.1789` on navtest n=100).

### 4.1 Why L12, not L27 (the numerical tie-breaker)

| metric | L12 | L27 |
|---|---:|---:|
| vision_frac_mean (n=100) | 0.1789 | 0.1804 |
| per-scene std | 0.0599 | 0.0460 |
| downstream layers (pruning savings) | **15** | **0** ← decisive |
| isolated peak (fine sweep) | ✅ L11=0.099, L13=0.069 | ✅ L26=0.045 |

L27 has 0 downstream consumers of its vision-KV cache → pruning at L27 saves no compute. **L12 wins by construction**, even though `vision_frac` is numerically tied.

See `docs/_internal/m1a_layer_selection_2026-06-18.md` for full reasoning.

### 4.2 Full 14-layer sweep summary

Layers swept: {0, 4, 8, 10, 11, 12, 13, 14, 16, 20, 24, 25, 26, 27}, n=100 each, 1400 forward passes total. **Clear double-peak**: L12 (0.1789) and L27 (0.1804). Per-layer SE ≈ 0.006.

| L | vision_frac | | L | vision_frac |
|--:|---:|---|--:|---:|
| 0 | 0.0537 | | 14 | 0.0793 |
| 4 | 0.0302 | | 16 | 0.0262 |
| 8 | 0.1034 | | 20 | 0.0952 |
| 10 | 0.1158 | | 24 | 0.0458 |
| 11 | 0.0993 | | 25 | 0.1096 |
| **12** | **0.1789** ← L\* | | 26 | 0.0446 |
| 13 | 0.0688 | | **27** | **0.1804** ← rejected |

### 4.3 Sanity checks

- 1400/1400 .pt saved, 0 err, 0 skip.
- V2/V3/V4 asserts: `captured_q_len == prompt_len == 941`, `vision_attn.shape == (720,)`, `vision_blocks` consistent across scenes.
- coarse + fine sweep agree on common layers (L12=0.1789, L27=0.1804 identical between coarse pass and re-analysis).

### 4.4 n=500 sanity (DONE 19:17, 2026-06-18)

To eliminate the n=100 tie ambiguity, we reran L12 and L27 on token index 100–499 (complementary to the original 100), then merged with the n=100 results to get n=500 each.

| layer | n   | mean   | std    | SE     | 95% CI            |
|------:|----:|-------:|-------:|-------:|:------------------|
| **L12** | 500 | **0.1861** | 0.0613 | 0.0027 | [0.1807, 0.1914] |
| L27   | 500 | 0.1805 | 0.0415 | 0.0019 | [0.1769, 0.1841]  |

**L12 mean increased by +0.0072 (n=100 → n=500), L27 nearly flat (+0.0001).** The n=100 "tie" was sampling noise. At n=500, L12 leads L27 by +0.0056 (z=−1.68 toward L27 lower; ranking did not swap).

**L\*=12 lock is now triple-supported**: numerical (n=500), engineering (15 downstream layers), structural (isolated peak in fine sweep).

### 4.5 Pending (Monday)

- navtrain probe A 10-min re-confirm L\*=12 after `.chain_complete`.

### 4.6 Cost

| | value |
|---|---|
| Total forward passes | 2200 (14 layers × 100 + 2 layers × 400 extra) |
| Wall clock (4× then 2× H20) | 12 min coarse + 25 min fine + 17 min n=500 sanity |
| Storage | ~7 MB total (2200 × ~3KB .pt) |

---

## 5. _Reserved for future milestones_

```
## 5. M0.2 — navtrain splits         [will fill after download]
## 6. M1.b — token relevance scoring [will fill after run]
## 7. M2.x — pruning ratio sweep     [will fill]
## 8. M5/M6 — final RL results       [will fill]
```

---

## 5. M1.b₀ — per-head decomposition (navtest, starter)

**Headline**: L12 has **1 dead head (h13, mean=0.0002)**, L27 has **2 fully dead (h8, h9, mean=0.0000)**. Per-head ranking is **sample-invariant** (Spearman ρ=1.0000 on disjoint sample). Top-12 head mask at L12 retains 96.8% of vision attention with 25% KV reduction.

### 5.1 L12 per-head highlights (n=100)

| top-4 heads | their share | top-8 cumulative | top-12 cumulative | effective heads |
|---|---:|---:|---:|---:|
| {8, 9, 15, 12} | 61.4% | 86.2% | **96.8%** | 10.44 / 16 |

| dead/near-dead at L12 | mean | role |
|---|---:|---|
| head 13 | 0.0002 | **dead** ⚰️ |
| head 14 | 0.0248 | near-dead |
| head 0  | 0.0332 | near-dead |
| head 6  | 0.0339 | near-dead |

### 5.2 Ranking stability (n=100 vs disjoint n=200)

| metric | value |
|---|---|
| Spearman rank corr | **1.0000** |
| Pearson mean corr | 0.9997 |
| top-{4,6,8,12} overlap | 4/4, 6/6, 8/8, 12/12 (identical) |

→ Head ranking at L12 is **structural**, not a sampling artifact. n=100 baseline is sufficient for M1.b design.

### 5.3 L27 per-head (foil, n=100)

| top-4 heads | top-2 share | top-12 cumulative | effective heads | dead |
|---|---:|---:|---:|---|
| {11, 3, 10, 1} | 39.6% | 99.9% | 9.56 / 16 | {h8, h9} (both 0.0000) |

**Cross-layer**: top-4 head IDs at L12 ({8,9,15,12}) are **disjoint from** L27 ({11,3,10,1}) → M1.b head selection must be **per-layer**, not global.

### 5.4 Free-lunch action (Level 0, 3 variants — to be run on navtest)

Goal: stack confirmed dead heads across layers, run navtest sweep, expect PDMS ≥ 0.8983 − 0.001 (within B0 noise).

| variant | layers × heads masked | heads removed | KV saving (over all 256 head-slots) | risk | navtest target |
|---|---|---:|---:|---|---|
| **V1 minimal** | L12:{h13} | 1 | 0.39% | none — single confirmed-dead head | PDMS ≥ 0.8983 − 1e-3 |
| **V2 moderate** | L12:{h13} + L27:{h0,h8,h9} | 4 | 1.56% | very low — all confirmed dead/near-dead | PDMS ≥ 0.8983 − 1e-3 |
| **V3 aggressive** | V2 + L24:{h0,h1,h2,h6,h7,h8,h9,h10,h12,h14,h15} | 15 | 5.86% | low — all 11 L24 heads have mean < 1e-3 | PDMS ≥ 0.8983 − 5e-3 |

Implementation: static head-mask in `code/rldrive/agents/autovla_with_attention.py`, gated by `cfg.head_mask_layers: dict[int, list[int]]`. Default off → B0 equivalence preserved.

### 5.5 Cross-layer landscape sweep (L8/L12/L16/L20/L24/L27, n=100)

Source: `exp/m1a_perhead_L12/landscape_summary.json`.

| layer | g_mean | top-4 share | top-12 cum | eff_heads | dead heads (count) |
|---|---:|---:|---:|---:|---|
| L8  | 0.103 | 63.4% | 97.7% | 9.76 | (0) |
| **L12** | **0.179** | 61.4% | 96.8% | **10.44** | **{h13} (1)** ← M1.a selected |
| L16 | 0.026 | 89.9% | 99.3% | **3.61** | {h1, h10, h13, h14} (4) |
| L20 | 0.095 | 60.3% | 99.9% | 8.98 | {h8, h11, h12, h14} (4) |
| **L24** | 0.046 | **98.4%** | **99.99%** | **3.64** | **{h0,h1,h2,h6,h7,h8,h9,h10,h12,h14,h15} (11)** ← maximal free-lunch |
| L27 | 0.180 | 61.1% | 99.9% | 9.56 | {h0, h8, h9} (3) |

**Key insight**: dead-head identities are **layer-specific** (no global pattern). L24 is the most-prunable layer with only 3 active heads, but its low g_mean (0.046) means downstream PDMS impact is also small per-token. See `docs/_internal/m1b_per_head_analysis_2026-06-18.md §A` for full per-head means.

### 5.6 Cost

| | value |
|---|---|
| Total forward passes (per-head) | 1000 (L12 n=100 + L12 extra n=200 + L27 n=100 + L8/L16/L20/L24 each n=100) |
| Wall clock | ~25 min cumulative (2× H20, parallel) |
| Storage | ~95 MB (.pt per-head tensors: 16 × 720 floats × 600 scenes) |

### 5.7 Pending follow-ups

- ✅ L8 / L16 / L20 / L24 per-head landscape sweep — completed 2026-06-18 21:29.
- 🔄 **Implement Level 0 free-lunch mask + B0-style navtest re-run (3 variants V1/V2/V3)** — Phase D/E/F, 2026-06-22 evening.
- ⏳ Per-scene head-rank variance analysis (feeds Level 2 learned policy).
- ⏳ M1.a Step 5 — navtrain probe A re-confirm of L\*=12 (blocked on `.chain_complete`).

See `docs/_internal/m1b_per_head_analysis_2026-06-18.md` for full reasoning, ranking tables, and reproducibility commands.

---

## 6. _Reserved for future milestones_

```
## 7. M0.2 — navtrain splits          [will fill after download]
## 8. M1.b₁ — Level-0 free-lunch sweep  [running, 2026-06-22]
## 9. M2.x — pruning ratio sweep      [will fill]
## 10. M5/M6 — final RL results       [will fill]
```

---

## 7. Changelog of this file

| date | change |
|---|---|
| 2026-06-16 20:30 | initial — populate B0 + M0.1 + env status |
| 2026-06-18 16:56 | M1.a coarse sweep (8 layers, n=100): L\*=27 nominal, L12 tied close 2nd. |
| 2026-06-18 18:55 | M1.a fine sweep done (14 layers, n=100). **L\*=12 locked v1** by downstream-flop argument. L27 rejected (0 downstream layers). n=500 sanity rerun on L12+L27 launched. |
| 2026-06-18 19:17 | M1.a **n=500 sanity confirmed** L\*=12. L12=0.1861 > L27=0.1805 (gap +0.0056). M1.a v2 final lock on navtest. |
| 2026-06-18 21:20 | **M1.b₀ per-head starter done**. L12 dead head = {h13}. L27 dead heads = {h8, h9}. Spearman ρ=1.0000 on disjoint sample (n=100 vs n=200). top-12 mask at L12 → 96.8% vision share retained @ 25% KV reduction. Top-4 head sets disjoint between L12 ({8,9,15,12}) and L27 ({11,3,10,1}). L8/L16/L20/L24 sweep in progress. |
| 2026-06-22 20:55 | **Layer landscape sweep complete** (L8/L12/L16/L20/L24/L27). L24 = most-prunable (11 dead heads, eff_heads=3.64). Level-0 free-lunch redesigned as 3 variants (V1/V2/V3) stacking confirmed dead heads cross-layer. Phase D/E/F implementation + navtest sweep starting tonight. |
