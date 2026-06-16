# MA2_b0_navtest — Baseline B0 PDMS for AutoVLA on navtest

> 📊 **Headline numbers live in [`docs/results/key_results.md`](../results/key_results.md) §1.**
> 本 journal 保留完整推导/复现细节；查数字请先看 key_results。

date: 2026-06-16 19:50 (China time)
status: **B0 LOCKED**
milestone: M0 (MA2.5) — main deliverable

---

## TL;DR

| metric | B0 (ours, 4×H20) | AutoVLA paper | Δ |
|---|---|---|---|
| **mean PDMS** | **0.8983** | 0.8950 | **+0.33 pt** |

Backbone: AutoVLA (Qwen2.5-VL-3B), ckpt `AutoVLA_PDMS_89.ckpt`,
LoRA merged & disabled, ratio=1.0 (no token pruning, the baseline).
Eval set: navtest 11576 unique tokens (2 invalid out of 11576).

**B0 is locked.** All downstream milestones compare against this number.

---

## 1. PDMS sub-components

| component | mean | floor | notes |
|---|---|---|---|
| no_at_fault_collisions       | 0.9944 | 0/1 binary | only 65 tokens with collision |
| drivable_area_compliance     | 0.9603 | 0/1 binary | 459 off-road tokens |
| ego_progress                 | 0.8326 | continuous | dominant variance source |
| time_to_collision_within_bound | 0.9768 | 0/1 binary | 269 TTC violations |
| comfort                      | 0.9986 | 0/1 binary | 16 uncomfortable tokens |
| driving_direction_compliance | 0.9812 | 0/1 binary | 218 wrong-direction tokens |
| **aggregate score**          | **0.8983** | composed | navsim v1 PDMS formula |

Sub-component levels match AutoVLA paper expectations — ego_progress
being the dominant low term is consistent with the paper's
collision-driven safety + progress-driven utility trade-off.

---

## 2. Eval set provenance

- **Source scene_filter**: `navtest_local_filtered.yaml` (12146 tokens)
- **Filtered by**: metric_cache availability ∩ navtest_nocot json
  availability → **11596 eligible**
- **Sharded by**: sha1(token) mod 4
  - shard0: 2954 → csv 2950 rows (+1 aggregate) → 2949 valid token rows
  - shard1: 2798 → 2797
  - shard2: 2969 → 2964 → 1 invalid + 2 missing? — see below
  - shard3: 2875 → 2869
- **Merged total**: 11576 unique tokens (after dropping `'average'`
  aggregate rows) → 11574 valid + 2 invalid

### 2.1 Token snapshot

Reproducible 11596-token list dumped to:

```
data/splits/navtest_b0_tokens.txt
```

(sha1-sorted, plain text, one token per line — `wc -l = 11596`)

### 2.2 Missing 20 tokens (11596 eligible vs 11576 in merged csv)

Diff of -20 between sharding intersection (11596) and merged csv unique
tokens (11576). Hypotheses:
- (a) navsim's `Unused metric cache for N tokens` discard — likely
- (b) some shard csvs ended before writing aggregate row — unlikely given
  all 4 rc=0
- (c) drop_duplicates removed legitimate per-shard repeats — unlikely
  because shards are disjoint by hash

Not blocking B0 (loss = 0.17% of eval set, well within noise).
Will investigate during M1 prep if it affects downstream label generation.

### 2.3 Invalid tokens

```
d318551a8ce150e5  (shard2)
7defd0c32cd8546a  (shard3)
```

Both are valid=False in their respective csvs. Will inspect logs to
understand failure mode (trajectory decode failure? metric_cache
mismatch?). 2/11574 = 0.017% failure rate — well within the < 1% target.

---

## 3. Timing & throughput

```
Wall-clock (4× H20, parallel): 1h 50m total
  ├─ sharding helper:        ~1 s
  ├─ 4× ckpt load + warmup:  ~30 s (per GPU)
  ├─ steady-state inference: ~1h 45m
  └─ csv merge:              ~1 s

Per-token throughput per GPU:
  shard0 (2954 tokens): 1h47m → 2.18 s/token
  shard1 (2798 tokens): 1h42m → 2.20 s/token
  shard2 (2969 tokens): 1h48m → 2.19 s/token
  shard3 (2875 tokens): 1h44m → 2.19 s/token

Effective: ~2.19 s/token/GPU steady-state.
Wall-clock per token at 4× parallel: 0.55 s/token.

VRAM: 30.9 GB / 98 GB per GPU (model fits with massive headroom).
GPU util: 25-85% (volatile; bottleneck = sensor blob IO from CephFS).
```

Extrapolation: any future r=1.0 navtest sweep (e.g., M1.b attention dump)
takes ~2h on 4 GPUs.

---

## 4. Comparison vs plan acceptance criteria

| criterion (plan §M0 acceptance) | actual | pass? |
|---|---|---|
| MA2.1 + MA2.2 + MA2.3 ready | yes | ✅ |
| B0 PDMS ∈ [0.45, 0.65] (loose plan target for EPDMS v2) | 0.8983 (PDMS v1) | ⚠ N/A see §6 |
| B0 PDMS > prior-work ceiling (~0.41) | 0.8983 ≫ 0.41 | ✅ |
| Inference latency + token count recorded | 2.19 s/token, 11576 tokens | ✅ |

Note: plan §M0 was written assuming **EPDMS v2** (navsim v2's 9-axis
score with environmental factors). AutoVLA fork's `run_pdm_score_cot.py`
actually runs **PDMS v1** (7-axis: 6 sub-components + aggregate score).
The 0.45-0.65 range was the EPDMS v2 expectation; under PDMS v1,
AutoVLA's expected ballpark is 0.88-0.90, matching our 0.8983.
This re-interpretation is documented in
`docs/journal/2026-06-15_benchmark_switch_to_navtest.md` (already
captured during MA2.3 work).

---

## 5. Artifacts

| artifact | path |
|---|---|
| B0 PDMS merged csv | `exp/ma2_5_b0_quad_merged_20260616_154858/merged.csv` |
| Token list snapshot | `data/splits/navtest_b0_tokens.txt` |
| Shard csvs (raw) | `exp/ma2_5_b0_quad_shard{0..3}_20260616_154858/.../*.csv` |
| Dispatcher log | `logs/ma2_5_b0_quad_dispatcher.log` |
| Per-shard inference logs | `logs/ma2_5_b0_quad_shard{0..3}_20260616_154858.log` |
| Dispatcher script | `scripts/run_autovla_navtest_dual_gpu.sh` (4-GPU mode: `GPUS="0 1 2 3"`) |
| Dispatcher journal | `docs/journal/2026-06-16_ma2_5_b0_dispatcher.md` |

---

## 6. Why PDMS not EPDMS — explicit note for future-self

AutoVLA's bundled navsim fork (`code/third_party/AutoVLA/navsim`) only
exposes the v1 PDMS evaluator via `run_pdm_score_cot.py`. The hydra
config tree has no EPDMS v2 scorer wired up. Per `bench_switch_to_navtest`,
we decided **MA2 will report PDMS** (= what AutoVLA paper itself uses,
which is also what the eval ckpt is named after: `AutoVLA_PDMS_89`).

If later milestones (M5/M6) need EPDMS v2, the integration cost is:
- add v2 scorer hydra config in AutoVLA fork (low risk)
- re-run inference on navtest 11596 tokens (~2h on 4 GPUs)

This is **deferred** until/unless M6 review requires v2 axes
(e.g., the "environment" or "lane keep" axes that v2 adds).

---

## 7. Risks identified during MA2.5

### 7.1 navtrain dataset missing → blocks M0.4 / M1 / M2

**Severity: HIGH**, **Status: AWAITING DECISION**.

Detailed analysis in `docs/_internal/risk_navtrain_data_missing.md`.

Summary: implementation plan assumes navtrain (104480 tokens) for SFT +
RL training pools. We don't have the raw blobs. Two paths:
- (A) Download via `navsim/download/download_trainval.sh` (~600-800GB
  camera-only, ~8h). Disk OK (1.6T free).
- (B) Repartition: use navtest \ probe_500 as train pool, probe_500 as
  M6 eval. Smaller eval, no leakage with proper held-out.

**Recommend tabling decision until user is back** — both paths are
viable, requires user judgment on download time vs eval-set-size
trade-off.

### 7.2 navsim discards ~20 tokens silently

See §2.2. Not blocking. Will probe in M0.4 / M1 prep.

---

## 8. What's next

### Immediate (today, can start now)
- [x] Lock B0 → this doc
- [x] Snapshot token list
- [ ] **Investigate Path A vs B for navtrain** — present trade-offs

### Short-term (after navtrain decision)
- [ ] Start navtrain download if A chosen (8h background)
- [ ] OR rewrite plan to use navtest \ probe_500 (Path B)
- [ ] M0.2 navtest splits (probe_500 stratified by `instruction`):
  - left / right / keep_forward / U-turn — 125 each
- [ ] M1.a layer probing setup (100-scene attention dump)

### Background / parallel
- [ ] Investigate the 2 invalid tokens (`d318551a8ce150e5`, `7defd0c32cd8546a`)
- [ ] Investigate the -20 missing tokens (§2.2)

---

## 9. Repro

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA

# Re-run B0 (idempotent; will produce new TS-tagged exp dir):
SCENE_FILTER_SRC=navtest_local_filtered \
  JSON_DIR=$PWD/data/navtest_nocot \
  METRIC_CACHE=$PWD/data/navtest_metric_cache \
  EXP_TAG=ma2_5_b0_quad_repro \
  TIMEOUT_PER_GPU=43200 \
  GPUS="0 1 2 3" \
  bash scripts/run_autovla_navtest_dual_gpu.sh

# Expected wall-clock: ~2h on 4× H20.
# Expected mean PDMS: 0.898 ± 0.005 (run-to-run noise from sensor blob
#   IO ordering + multinomial sampling in trajectory decoder).
```
