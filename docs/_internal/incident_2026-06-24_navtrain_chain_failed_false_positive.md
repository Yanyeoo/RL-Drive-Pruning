# Incident — 2026-06-24 `.chain_failed` is a false positive; navtrain data is intact

> **Severity**: critical (operational, blocked M0.4 / M1.a Step 5 / RL main line for ~1.5 days)
> **Severity (data)**: zero
> **Detected**: 2026-06-24 ~17:00 by user-driven re-verification
> **Originally written**: 2026-06-23 03:12:18 by `install_navtrain.sh` rc=141 → `post_dl_chain.sh` wrote `.chain_failed`
> **Related incidents**: `incident_2026-06-23_overnight_sighup.md` (same chain, operational angle)
> **Related risk**: `risk_navtrain_data_missing.md` (now resolved — navtrain IS on disk)

---

## TL;DR for the next AI

1. **`.chain_failed` lied.** rc=141 was SIGPIPE from a `ls ... | head -3` sanity-check line in `install_navtrain.sh`, after **all real install work had already succeeded**. The sentinel has been flipped to `.chain_complete` and a copy of the original `.chain_failed` preserved as `.chain_failed.false_positive`.
2. **navtrain data is fully on disk** at `$OPENSCENE_DATA_ROOT` = `/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/`:
   - `navsim_logs/trainval/`     — 1310 `.pkl`, 14 GB
   - `sensor_blobs/trainval/`    — 1192 scene dirs, 443 GB
3. **Hard verification PASSED:** navsim's standard `SceneLoader` enumerated the full `navtrain.yaml` filter — `declared = 103288` tokens / `built = 103288` tokens / `diff = 0`. ✅
4. **DO NOT panic about "sample scene cam load fails".** navtrain sensor data is **sparse key-frames by design** (9 jpgs per scene per camera, not 1:1 with log frames). `SensorConfig.build_all_sensors()` over a 15-frame window is the wrong API for spot-checking; the actual training pipeline does not trigger this failure mode. This is documented at §3 below.
5. **Unblocked tasks (this is what you should be doing now):**
   - **M0.4** navtrain r=1.0 baseline (was blocked by fake `.chain_failed`)
   - **M1.a Step 5** navtrain probe A (verify L\*=12 holds on navtrain)
   - Plan dependency: `M1.b → M1.c → M2 → M5` all gated on the above.

---

## 1. What actually happened during the chain

Reconstructed from `logs/post_dl_chain.log` and `scripts/install_navtrain.sh`:

| Time | Step | Result |
|---|---|---|
| 2026-06-23 02:38 | last rsync (history_3) complete | ✅ |
| 02:47–02:56 | history_4 rsync | ✅ |
| 03:00:05 | all 8 `navtrain_{current,history}_*.tgz` downloaded, **all md5 OK** | ✅ |
| 03:03:48 | `.download_complete` sentinel written | ✅ |
| 03:04:39 | `post_dl_chain.sh` triggered `install_navtrain.sh` (1st attempt) | — |
| 03:08:43 | logs install reaches "navsim_logs/trainval installed: 14G" then `install_navtrain.sh` exits **rc=1** | ⚠️ misleading |
| 03:08:44 | `post_dl_chain.sh` retried (2nd attempt) | — |
| 03:08:44–03:12:18 | sensor_blobs mv (443G) completes | ✅ |
| 03:12:18 | `install_navtrain.sh` exits **rc=141** at sanity-check `ls ... | head -3` | ⚠️ false positive |
| 03:12:18 | `post_dl_chain.sh` wrote `.chain_failed` (`step=install_navtrain.sh rc=141`) | ❌ wrong call |
| 03:14:05 | `overnight_watch.sh` saw all PIDs dead → FATAL (separate SIGHUP issue) | — |

### Why rc=1 on 1st attempt (the easier bug)

`install_navtrain.sh:38` does:
```bash
[[ -d "${STAGE_LOGS}" ]] || { log "FATAL: ${STAGE_LOGS} missing"; exit 1; }
```
where `STAGE_LOGS = ${STAGING}/trainval_navsim_logs/trainval`.

`download_navtrain_robust.sh` ran with **two concurrent instances** (user manual takeover of history_2 mid-flight, see incident_2026-06-23). One instance had already `mv`'d `STAGE_LOGS` to `LIVE_LOGS` by the time the second instance's chain reached preflight → directory genuinely gone → `set -euo pipefail` → rc=1. The retry succeeded because logs were already in place and the script's idempotent branch skipped them.

### Why rc=141 on 2nd attempt (the real false positive)

`install_navtrain.sh:81–82`:
```bash
ls "${LIVE_LOGS}"  | head -3
ls "${LIVE_BLOBS}" | head -3
```
- `set -o pipefail` is on.
- `head -3` closes its stdin after 3 lines.
- `ls` keeps writing → kernel sends SIGPIPE → `ls` exits 141.
- `pipefail` propagates 141 as the pipeline exit status.
- `set -e` terminates the script.

**This is a textbook bash-strict-mode gotcha.** Nothing in `LIVE_LOGS` / `LIVE_BLOBS` is wrong; the script literally just printed the first 3 filenames successfully and then died on the IO closure.

---

## 2. How the data was hard-verified (2026-06-24)

### 2.1 On-disk layout

```
$OPENSCENE_DATA_ROOT/navsim_logs/trainval/       1310 *.pkl   14 G
$OPENSCENE_DATA_ROOT/sensor_blobs/trainval/      1192 dirs    443 G
$OPENSCENE_DATA_ROOT/_staging_navtrain/          (empty dirs + sentinels only, no leftover data)
```

- `1310 logs vs 1192 sensor dirs` is **navsim by design**: `metadata_trainval.tgz` ships the full trainval log metadata, while `navtrain_{current,history}_*.tgz` ship only the navtrain subset of sensor blobs (445 GB per [splits.md](../../code/third_party/navsim/docs/splits.md): "subset and filter of trainval"). The 118 extra logs cover navtest and unused trainval segments.
- `intersection(logs, blobs) = 1192`, `blobs - logs = 0` → no orphan blobs, no rsync corruption.

### 2.2 Standard SceneLoader enumeration

```python
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
import hydra
from omegaconf import OmegaConf

cfg = OmegaConf.load(".../scene_filter/navtrain.yaml")
scene_filter = hydra.utils.instantiate(cfg)
loader = SceneLoader(
    data_path=LOG_ROOT,
    sensor_blobs_path=SENSOR_ROOT,
    scene_filter=scene_filter,
    sensor_config=SensorConfig.build_no_sensors(),
)
assert len(loader.tokens) == 103288   # PASS, diff=0 from declared
```

- `navtrain.yaml`: 103288 tokens × 167 log_names declared.
- (Earlier session note quoting "104480 tokens" was a misread of `grep -c "^  - '"` which counts both `tokens` and `log_names` blocks; the true token count is 103288.)
- Enumeration takes ~120 s on this filesystem and walks every scene dir; if any sensor dir were missing the loader would prune that token, so `diff=0` is a strong existence proof.

---

## 3. The "sample scene cam load fails" red herring

If you write a one-off verification like:
```python
loader = SceneLoader(..., sensor_config=SensorConfig.build_all_sensors())
scene = loader.get_scene_from_token(loader.tokens[0])
```
**you will get `FileNotFoundError: .../CAM_F0/<some_hash>.jpg`. This is not a data problem.**

Why:

- The log `.pkl` (e.g. `2021.05.12.19.36.12_veh-35_00005_00204.pkl`) contains **398 dense log entries**, each referencing image hash names that exist somewhere in the full openscene trainval ~2.1 TB tree.
- Our `sensor_blobs/trainval/<scene>/CAM_*/` directories contain **9 jpgs per camera** — these are the navtrain "key frames" only. navtrain trades sensor density for storage (2.1 TB → 443 GB).
- `SensorConfig.build_all_sensors()` over a `4 history + 1 current + 10 future` window asks for sensor data on **15** frames, of which **typically 1–2** are key frames on disk; the rest are log-only frames whose jpg hashes were never shipped in the navtrain tgz set.

**The correct training-time API** uses `SensorConfig` with `*_indices=[]` for the camera fields you don't need, or uses the navtrain-aware sensor configs that AutoVLA's training pipeline already wires up (search `code/third_party/AutoVLA` for `sensor_config:` in training yamls). The chain itself does not need `build_all_sensors()` to declare success — token enumeration with `build_no_sensors()` is the standard "is this dataset usable" check.

If you want a *visual* spot-check, do this instead:
```python
import os, glob
scene_dir = f"{SENSOR_ROOT}/{loader.tokens[0]_to_scene_name}"
for cam in ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2"]:
    files = sorted(glob.glob(f"{scene_dir}/{cam}/*.jpg"))
    print(cam, len(files), files[0] if files else "—")
# Expect: 9 jpgs per cam, all 8 cams + MergedPointCloud present
```

---

## 4. Remediation applied (2026-06-24)

1. **Sentinel flipped:**
   - `$OPENSCENE_DATA_ROOT/_staging_navtrain/.chain_failed` → renamed to `.chain_failed.false_positive` (forensic copy)
   - Wrote `$OPENSCENE_DATA_ROOT/_staging_navtrain/.chain_complete` with a one-line provenance note.
2. **`install_navtrain.sh` patched** at lines 81–82 to swallow SIGPIPE on the sanity-check `ls | head`:
   ```bash
   { ls "${LIVE_LOGS}"  || true; } | head -n 3 || true
   { ls "${LIVE_BLOBS}" || true; } | head -n 3 || true
   ```
   Equivalent fixes (any one works): wrap in subshell with pipefail off, or use `printf '%s\n' "${LIVE_LOGS}"/*  | head -n 3`. The chosen form keeps `set -euo pipefail` intact for the rest of the script.

(No change to `post_dl_chain.sh`; it correctly trusts `install_navtrain.sh`'s exit code. The bug was inside the inner script.)

---

## 5. Lessons / guardrails for the next AI

- **rc=141 inside a `set -euo pipefail` bash script is SIGPIPE 95% of the time.** Before declaring "chain failed", grep the failing script for `| head`, `| grep -q`, `| awk 'NR==1'` patterns near the reported line.
- **Trust on-disk evidence over sentinels.** Sentinels are written by scripts that have their own bugs; the actual data has none. The check order should be:
  1. on-disk byte count vs spec (443 GB ≈ navsim docs' 445 GB)
  2. md5 records in download log
  3. `SceneLoader` enumerate count vs `scene_filter.tokens` declared count
  4. *only then* trust/distrust the sentinel
- **Don't use `SensorConfig.build_all_sensors()` as a "smoke test" for navtrain.** It will always fail by design. Use `build_no_sensors()` for dataset existence checks; use the actual training yaml's sensor_config for end-to-end checks.
- **Concurrent instances of `download_navtrain_robust.sh`** (e.g. user manual takeover + the chain's own retry) can cause harmless rc=1 in `install_navtrain.sh:38` preflight when one instance has already moved the staging dir. This is OK as long as the retry path succeeds. Consider adding `mkdir -p` + lockfile in a future cleanup pass.

---

## 6. What to do next (unblocked)

In priority order (from RESUME_MONDAY.md, now actually executable):

1. **M0.4** — navtrain r=1.0 baseline EPDMS. Run `m02_splits.sh` if not yet, then the baseline inference. ETA: hours.
2. **M1.a Step 5** — navtrain probe A at `L*=12`. Acceptance: `vision_frac_mean ∈ [0.15, 0.22]`. Template in `RESUME_MONDAY.md` §"probe A on navtrain". If pass → M1.a is fully delivered, proceed to M1.b.
3. **M1.b** — full navtrain attention extraction (1× full inference) at L=12 for the M1.c data pool.

V4 isolation (mentioned in user's earlier note) is **not** on this critical path; defer until M0.4 + Step 5 are green.

---

## 7. Addendum (2026-06-24 20:10) — "missing-images scan" is the same trap

After the chain was unblocked, a follow-up session hit a `FileNotFoundError` on
a single jpg in the M1.a Step 5 nocot prebuild and reacted by writing a full
"per-token window scan" (`scripts/scan_navtrain_missing_images.py` +
`scripts/scan_navtrain_window.py`) that, for every target token in
`navtrain.yaml`, required all 8 cams × 14 frames (= 112 jpgs) inside the
`[-4, +10]` window to exist on disk. It reported **81 % unusable / 18.6 %
"clean" = 19 225 tokens**.

**That is the same anti-pattern as §3.** The 14-frame window contains only
1–2 key-frames per camera by design; the other 12–13 frames' jpg hashes are
intentionally not in the navtrain tgz set. The scan was therefore measuring
how many windows happen to coincide with key-frame frames, not how many
tokens are usable.

How we know the scan was wrong:

- **§2.2 still holds**: `SceneLoader(..., build_no_sensors())` enumerates all
  103 288 tokens with diff=0.
- **End-to-end test**: a freshly sampled 100-token list straight from
  `navtrain.yaml`'s `tokens:` block, fed through `nocot_sample_generation.py`
  and then `run_attention_probe.py`, produced **ok=100, skip=0, err=0**
  (`vision_frac_mean=0.1693`, journal `2026-06-24_m1a_step5_navtrain_probeA_pass.md`).
- The original missing-jpg that triggered the scan came from a token list
  reused from a navtest probe — that token was **not** in `navtrain.yaml` at
  all (it lived in a log whose frames navtrain intentionally did not ship).

**Decisions**:

- The scan products (`exp/m1a_navtrain_probeA_setup/navtrain_window_*`) are
  kept on disk as forensic evidence of the false alarm. **Do not** import them
  as a token filter for M1.b / MA2.x. The token pool for those steps is the
  full 103 288.
- `RESUME_MONDAY.md` has been updated with a "do not repeat this scan" note in
  the M1.a Step 5 PASS block.

**Add to §5 guardrails (for the next AI)**:

- **Do not write a "missing images scan" for navtrain that requires N×8 jpgs
  to exist per window.** That is `build_all_sensors()` in disguise and will
  always report ~80 % unusable. If you want to spot-check a token, run it
  through the actual nocot pipeline (or `SceneLoader.get_scene_from_token`
  with the training-time `sensor_config`) — if it succeeds the token is
  usable; if it fails the token is genuinely broken (extremely rare in
  navtrain.yaml itself).

