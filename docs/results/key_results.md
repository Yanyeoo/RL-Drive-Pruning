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
| M1.a | attention probing | _not started_ | — | — | — |

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

## 4. _Reserved for future milestones_

```
## 4. M0.2 — navtrain splits         [will fill after download]
## 5. M1.a — attention probing       [will fill after run]
## 6. M1.b — token relevance scoring [will fill after run]
## 7. M2.x — pruning ratio sweep     [will fill]
## 8. M5/M6 — final RL results       [will fill]
```

---

## 5. Changelog of this file

| date | change |
|---|---|
| 2026-06-16 20:30 | initial — populate B0 + M0.1 + env status |
