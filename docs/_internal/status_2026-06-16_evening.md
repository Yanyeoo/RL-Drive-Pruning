# Status snapshot — 2026-06-16 19:15

Self-driven session checkpoint. Update on each major event.

## What's running right now

| pid | task | log | etime |
|----:|------|-----|------:|
| 51594 | navtrain background downloader | `logs/navtrain_download.log` | ~10 min |

Current download state: metadata.tgz @ ~80% / 5.2 GB, ~9 MB/s.
After metadata: 8 × ~54 GB tgz (current 1-4, history 1-4).
Expected total wall time: **~16-18 h** at observed bandwidth.

## Completed this session (after B0 lock)

### 1. Diagnosed B0's 2 invalid + 20 missing tokens
File: `docs/journal/2026-06-16_b0_invalid_token_diagnosis.md`

- 2 invalid (`d318551a8ce150e5`, `7defd0c32cd8546a`) — same root cause:
  `autovla.predict()` returned <8 poses → `Trajectory.__post_init__`
  assertion. **Pure model decoding edge case, not a pipeline bug.**
  Fix deferred to M5/M6 agent refactor (pad-last-pose patch in
  `autovla_agent.py:445`).
- -20 gap — `navsim`'s built-in `SceneFilter` (has_route +
  num_frames) drops 20 of the 11596 eligible tokens during log load.
  Standard behavior. Canonical B0 evaluable set is 11576.

### 2. Wrote navtrain download pipeline
- `scripts/download_navtrain_robust.sh` — robust wrapper around the
  vanilla AutoVLA `download_navtrain.sh`: `set -euo pipefail`,
  `wget -c`, md5 verify, install-sentinels for resumability.
- `scripts/install_navtrain.sh` — atomic mv from staging to
  `${OPENSCENE_DATA_ROOT}/{navsim_logs,sensor_blobs}/trainval`,
  refuses to overwrite non-empty targets.
- Staging: `/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain`
- Live target: `/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/` (same ceph mount → atomic mv)
- Risk doc: `docs/_internal/risk_navtrain_data_missing.md`

### 3. Wrote M0.2 + sanity scaffold (ready to run when data lands)
- `scripts/check_navtrain_sanity.py` — end-to-end sanity check:
  SceneLoader, agent_input load, driving_command distribution sample.
- `scripts/build_m02_splits.py` — produces:
  - `data/splits/probe_A.txt`  (100 tokens, stratified by driving_command, 25/class)
  - `data/splits/train_pool.txt` (90% of navtrain \\ probe_A)
  - `data/splits/val_pool.txt` (10% of navtrain \\ probe_A)
  - `data/splits/m02_split_stats.txt` (sidecar with class dist)
  - Deterministic via `--seed`.

## When download finishes (planned next steps, autopilot)

1. Run `scripts/install_navtrain.sh` → atomic mv to live root.
2. Run `scripts/check_navtrain_sanity.py` → validate end-to-end load.
3. Run `scripts/build_m02_splits.py` → produces M0.2 split files.
4. Cross-check split sizes vs `splits.md` (navtrain ≈ 100k+ scenes).
5. Update `MA2_b0_navtest.md` §7 "navtrain risk" to "resolved".
6. Move on to M1.a (layer probing) — but pause first: it needs
   attention dumps from autovla_agent inference. Will plan how to
   instrument before running.

## Open / parked

- M5/M6 fix for 2 invalid token pad-last-pose patch (deferred, doc'd).
- M1.b + M0.4 merge strategy (planned, not started — needs attention
  hook design in autovla_agent.forward).

## Reproducibility breadcrumbs

- B0 token snapshot: `data/splits/navtest_b0_tokens.txt` (11596 lines)
- B0 merged csv:
  `exp/ma2_5_b0_quad_merged_20260616_154858/merged.csv` (11577 lines = header + 11576 row)
- env: `/apdcephfs/private_shayladeng/miniconda3/envs/{autovla,navsim}/bin/python`
- ckpt: AutoVLA_PDMS_89.ckpt (Lightning full-FT, not LoRA)
