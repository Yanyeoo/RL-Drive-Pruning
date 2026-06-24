# 2026-06-24  `.chain_failed` was a false positive — navtrain unblocked

**Bottom line:** navtrain dataset has been on disk and complete since
2026-06-23 03:12. The `.chain_failed` sentinel was wrong. M0.4 / M1.a Step 5
were blocked for ~1.5 days by a bash SIGPIPE bug in a sanity-check line.

## What I verified

- `OPENSCENE_DATA_ROOT/navsim_logs/trainval/`  = 1310 `.pkl`, 14 G
- `OPENSCENE_DATA_ROOT/sensor_blobs/trainval/` = 1192 scene dirs, 443 G
  (matches navsim's documented 445 G)
- All 8 `navtrain_{current,history}_*.tgz` md5 OK (per download log)
- `SceneLoader` with `navtrain.yaml` filter enumerates **103288 / 103288** tokens
  (declared == built, diff = 0)

## Root cause of the false `.chain_failed`

`install_navtrain.sh:81–82` ran `ls "${LIVE_LOGS}" | head -3` under
`set -euo pipefail`. `head` closes the pipe after 3 lines; `ls` keeps writing,
gets SIGPIPE, exits 141; `pipefail` propagates 141; `set -e` exits the script.
By that point all install work was already done.

Earlier in the same chain there was also a real rc=1 from
`install_navtrain.sh:38` preflight, caused by **two concurrent instances** of
`download_navtrain_robust.sh` racing on the staging dir (one had already
moved `STAGE_LOGS` to `LIVE_LOGS` when the other reached preflight). The
chain's retry path handled this correctly — logs were already in place, so
mv was skipped and sensor_blobs install proceeded normally.

## Fixes applied

1. `_staging_navtrain/.chain_failed` → renamed to `.chain_failed.false_positive`
2. Wrote `_staging_navtrain/.chain_complete` with full provenance
3. Patched `scripts/install_navtrain.sh:81-82` to swallow SIGPIPE on the
   sanity-check `ls | head` (kept `set -euo pipefail` for the rest of the
   script intact)
4. Wrote `docs/_internal/incident_2026-06-24_navtrain_chain_failed_false_positive.md`
   with full timeline, evidence chain, and guardrails for the next AI

## Don't get fooled again

If you (next AI) write a quick `SceneLoader(..., SensorConfig.build_all_sensors())`
+ `get_scene_from_token(...)` smoke test and see `FileNotFoundError` on a
camera jpg — that is **not** a data problem. navtrain sensors are sparse
key-frames by design (9 jpgs per cam per scene). Use `build_no_sensors()`
for dataset existence checks, or use the actual training pipeline's
`sensor_config` for end-to-end checks. See incident doc §3.

## Next

Unblocked: **M0.4 navtrain r=1.0 baseline** → **M1.a Step 5 navtrain probe A**
(template in `RESUME_MONDAY.md`). Then M1.b → M1.c → M2 → M5.
