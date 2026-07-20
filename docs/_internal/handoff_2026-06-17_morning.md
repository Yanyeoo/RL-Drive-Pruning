# Morning resume @ 2026-06-17 17:05

> Written when picking up from the 2026-06-16 22:00 recycle.
> Pair-doc to: `RESUME_TOMORROW.md` (the plan) + this file (what actually happened).

## TL;DR

- Three commands in `RESUME_TOMORROW.md` worked **after one fix**:
  the new VM instance does **not have `aria2c` pre-installed**, so the
  download script `set -e`-died on `aria2c: command not found` the first time.
- Fixed via `yum install -y aria2` (TencentOS Server 3.2, RHEL-like; root
  available). aria2 1.35.0 installed cleanly.
- After install, restarted both background procs. Currently:
  - `download_navtrain_robust.sh` (pid 3738) → `aria2c` (pid 3754),
    pulling `navtrain_current_1.tgz` at **~100 MB/s × 16 conns**, ETA ~8min
    for split #1.
  - `post_dl_chain.sh` (pid 3774) polling `.download_complete`.
- Expected `.chain_complete` by ~noon (8 tgz × ~10min + extract/rsync +
  install + sanity + m02_splits build ≈ 4-5h).

## What I touched

1. `yum install -y aria2` — system-wide, persists in this VM but
   **may need redoing on next recycle**. If aria2c persists to a
   user-writeable location next time, it would be nicer to vendor it
   under `/apdcephfs/.../bin/`. Not urgent.
2. Killed the orphan first-attempt `post_dl_chain.sh` (pid 3173)
   that was polling but had no producer because the download had died.
3. Re-launched both background procs per `RESUME_TOMORROW.md` §TL;DR.
   No script edits.

## What I did NOT touch

- No edits to `scripts/download_navtrain_robust.sh` or
  `scripts/post_dl_chain.sh`. The aria2c-only design is correct; the
  fix is environmental.
- No data-side cleanup. Staging layout (`current_split_1/`,
  `trainval_navsim_logs/`, `trainval_sensor_blobs/`) is exactly as
  RESUME_TOMORROW predicted; the script's idempotency handles it:
  - step 0 (logs) skipped ✅
  - step 1 (tgz #1) re-downloading + will re-extract + rsync
    (~5min loss to redo the incomplete rsync — already accepted in
    RESUME_TOMORROW §"Why we can resume safely").

## Suggestion for `RESUME_TOMORROW.md` next time

Add a 4th pre-flight check before the three TL;DR commands:

```bash
command -v aria2c >/dev/null || { echo "MISSING aria2c — run: yum install -y aria2"; exit 1; }
```

So future-me does not waste 5min finding out the hard way.

## During-download plan (4-5h window)

Per `RESUME_TOMORROW.md` §"chain 跑完后下一步" and user's onboarding
message item 2: draft the M1.a attention hook code per the design in
`m1_attention_hook_design.md`. Pure model-side, does not touch
navtrain. Target: a reviewable code sketch at the autovla.py
`vlm.generate()` call site (~line 522-527).

## Status anchor

| item | state |
|---|---|
| B0 PDMS = 0.8983 | locked, see `docs/results/key_results.md` §1 |
| navtrain download | in progress (this morning resume) |
| chain (install + sanity + m02_splits) | armed (post_dl_chain.sh polling) |
| M1.a code | not started — drafting today during download |

— shayla
