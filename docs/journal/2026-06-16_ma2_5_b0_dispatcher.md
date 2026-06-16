# MA2.5 — Quad-GPU navtest dispatcher + B0 PDMS run

date: 2026-06-16 (afternoon)
status: dispatcher built, smoke verified, full run launched

## 0. Context

MA2.3 single-GPU inference smoke already passed earlier today
(see `2026-06-16_ma2_3_smoke_pass.md` — 5/5 tokens, mean PDMS=0.965).

Goal of this session:
1. Build a multi-GPU dispatcher that shards `navtest` (11596 eligible tokens)
   across N GPUs, runs inference in parallel, and merges per-shard csvs.
2. Validate the dispatcher on `navtest_smoke5` (cheap).
3. Launch full-run on 4× H20 → B0 PDMS number (M0 deliverable).

## 1. Dispatcher design

File: `scripts/run_autovla_navtest_dual_gpu.sh`
(filename preserved for backward compat; supports N>2 via `GPUS=` env var)

### Pipeline
```
src scene_filter ─┐
metric_cache      ├── intersection ──> eligible tokens
json corpus       ─┘
                       │
                       ▼
       sha1(token) mod N ──> N scene_filter yamls + N split yamls
                       │
                       ▼ (parallel)
   GPU 0  : run_pdm_score_cot.py train_test_split=<shard0>
   GPU 1  : run_pdm_score_cot.py train_test_split=<shard1>
   ...
   GPU N-1: run_pdm_score_cot.py train_test_split=<shardN-1>
                       │
                       ▼
       pandas concat ─ drop 'token=average' aggregate rows ─ drop_duplicates(token)
                       │
                       ▼
          merged.csv  +  mean PDMS  +  6 sub-metrics
```

### Key knobs
| env var            | default                       | role                                     |
|--------------------|-------------------------------|------------------------------------------|
| `SCENE_FILTER_SRC` | `navtest_local_filtered`      | scene_filter yaml name (smoke: `navtest_smoke5`) |
| `JSON_DIR`         | `data/navtest_nocot`          | MA2.1 json corpus                        |
| `METRIC_CACHE`     | `data/navtest_metric_cache`   | MA2.2 metric_cache root                  |
| `GPUS`             | `"0 1"`                       | space-separated device indices (4-GPU: `"0 1 2 3"`) |
| `TIMEOUT_PER_GPU`  | `86400`                       | seconds (full-run uses 43200=12h)        |
| `EXP_TAG`          | `ma2_dual`                    | prefix for `experiment_name`             |

## 2. Bugs hit during wrapper validation

### B1 — `pathlib` topology error in cache glob

```python
# wrong
cached = {p.parent.name for p in cache.glob('*/unknown/*')}
# right
cached = {p.name for p in cache.glob('*/unknown/*') if p.is_dir()}
```

metric_cache layout: `<log>/unknown/<token_hash>/`, so `p` is the leaf
`<token_hash>` dir. `p.parent.name` resolves to `"unknown"` for every match,
collapsing the set to size 1 and emptying the intersection.

Discovery: first smoke run reported `cached=1` even though `find` showed
~11596 token directories. Direct `Path.glob` repro confirmed.

### B2 — Hydra cannot resolve `navsim.agents.autovla_agent.AutoVLAAgent`

Root cause: the single-GPU smoke script did `cd ${NAVSIM_ROOT}` and exported
`PYTHONPATH="${NAVSIM_ROOT}:${AUTOVLA_ROOT}:..."`. The dispatcher initially
did neither, so the autovla agent module was not on the path.

Fix:
- `export PYTHONPATH="${NAVSIM_ROOT}:${AUTOVLA_ROOT}:${PYTHONPATH:-}"` at
  dispatcher top.
- Each parallel shard launches in a subshell that `cd ${NAVSIM_ROOT}` and
  `exec timeout ... python ...`, so cwd matches the single-GPU smoke
  and `$!` still captures the actual python PID.

### B3 — `token='average'` aggregate row inflates `drop_duplicates` count

Each navsim per-shard csv ends with one extra row (`token='average'`).
Two shards merging would produce one duplicate aggregate (sometimes dedup'ed
to a phantom 6th "token"). Fix: drop `token=='average'` before
`drop_duplicates`.

## 3. Validation runs

### 3.1 Dual-GPU smoke (5 tokens, GPU 0 + 1)

```
[shard] src=5 cached=11596 json=225 eligible=5 nshard=2
[shard0] tokens=3   [shard1] tokens=2
[merge] total unique tokens = 5
[merge] valid = 5  invalid = 0
[merge] mean score (valid) = 0.9647
   no_at_fault_collisions       = 1.0000
   drivable_area_compliance     = 1.0000
   ego_progress                 = 0.9153 (note: shard0 mean=0.9188 alone)
   time_to_collision_within_bound = 1.0000
   comfort                      = 1.0000
   driving_direction_compliance = 1.0000
```

Wall-clock per shard (incl. ckpt load): ~32-35s for 3 tokens
→ ~3-5s/token after warm-up; matches single-GPU smoke.

**Sanity check vs single-GPU smoke**: identical 5/5 valid, same mean 0.9647.
Sharding is content-preserving. ✅

### 3.2 Quad-GPU smoke (5 tokens, GPU 0..3)

```
[shard] src=5 cached=11596 json=225 eligible=5 nshard=4
[shard0] tokens=1  [shard1] tokens=2  [shard2] tokens=2  [shard3] tokens=0
shard0..2 rc=0;  shard3 rc=1 (empty input → hydra exits non-zero)
[merge] total unique tokens = 5
[merge] mean score (valid) = 0.9647
```

**Known noise**: when N > #eligible_tokens, the last shard receives 0 tokens
and exits rc=1. For the full-run this is impossible (11596/4≈2899 per shard)
so we accept this as a smoke-only artifact and don't gate on `all-rc==0`.

## 4. MA2.5 full-run launch (in progress as of writing)

```bash
SCENE_FILTER_SRC=navtest_local_filtered \
JSON_DIR=/apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtest_nocot \
METRIC_CACHE=/apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtest_metric_cache \
EXP_TAG=ma2_5_b0_quad \
TIMEOUT_PER_GPU=43200 \
GPUS="0 1 2 3" \
bash scripts/run_autovla_navtest_dual_gpu.sh
```

Background pid: see `logs/ma2_5_b0_quad_dispatcher.log`.

### Sharding result
```
src=12146  cached=11596  json=11596  eligible=11596  nshard=4
shard0: 2954    shard1: 2798    shard2: 2969    shard3: 2875
```

12146 vs 11596: scene_filter ships 12146 tokens but we only have
metric_cache + json for 11596 (550 missing from cache — pre-existing,
not blocking; these are the 4.5% holes left by earlier MA1.2 pipeline).

### Observed throughput (first 90s of steady-state)
```
shard0 scenario 20/2949 at +33s after ckpt-load done
shard3 scenario 20/2868 at +33s
GPU 0..3: 30.8GB each, util ~47%
```

→ steady-state **~2.2s/token/GPU** (matches MA2.3 plan's "~3-4s/token"
estimate; faster because batch overhead amortized across 2900 tokens).

ETA: max(2954, 2969) × 2.2s ≈ **108 min ≈ 1.8h wall-clock** for B0 number.
(Significantly under the plan's 6.5h estimate, which assumed 2 GPUs +
slower per-token.)

## 5. Pass criteria for MA2.5 (M0 deliverable)

- [ ] All 4 shards finish rc=0
- [ ] Merged unique tokens >= 11500 (allow ~100 invalid)
- [ ] At least 1 csv per shard
- [ ] `mean score` exists and is in [0, 1]
- [ ] Number gets recorded against AutoVLA paper's PDMS=89.5

## 6. Files changed in this session

```
M scripts/run_autovla_navtest_dual_gpu.sh    # B1+B2+B3 fixes; N-shard refactor
A docs/journal/2026-06-16_ma2_5_b0_dispatcher.md  # this file
```

(Repo-tracked patch unaffected — dispatcher lives entirely in our wrapper
scripts, no upstream AutoVLA edits.)

## 7. Next after B0 lands

- Record B0 PDMS (and per-sub-metric) into `docs/_internal/handoffs/`.
- Compare against AutoVLA paper's 89.5: if off by > 2 pts, debug;
  if within ±1 pt, lock B0 and proceed to **M1 (token-pruning v0)**.
- Generate `eligible_tokens.txt` snapshot so later experiments use the
  exact same 11596 for apples-to-apples comparison.
