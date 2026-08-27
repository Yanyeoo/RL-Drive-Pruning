# Scope pivot — baseline-only + stop taucut (2026-07-28 22:10)

## User rulings (this session)
- Keep my plan (shard0 draft -> upgrade, +fallback for ours, keep 7B). Runbook = reference only.
- Do NOT run a real safety_net four-shard; post-hoc +fallback 0.9045 is accepted as headline.
- ours fixed-ratio rows are NOT the headline; do not chase completing them.
- Stop rl_taucut; keep the already-running rl_shaped_r075_sh3 (near-free full row).
- GPU focus = baseline four methods only (FastV/Random/PruMerge/SparseVLM), all drop.

## Verified facts before acting
- Live pdm jobs were: rl_taucut_kr060 sh1(3712/13) sh2(3706/07) sh3(3710/11) + rl_shaped_r075_sh3(3708/09).
- rl_shaped job is in its own PGID 3708, independent of backfill PGID 3607 -> stopping backfill won't kill it.
- ours coverage: SFT r=0.5 full 0.87253 (+fb 0.90452), SFT r=0.75 full, RL r=0.25/0.5 full,
  RL r=0.75 sh0/1/2 (sh3 running), SFT-tau MSE kr060 full 0.89395, Budget RL full.
  Missing (intentionally NOT filled): SFT r=0.25, RL-taucut sh1/2/3.

## Actions
1. `kill -TERM` the 6 taucut PIDs (3706/07/10/11/12/13). rl_shaped_r075_sh3 kept alive.
2. Backed up `run_maintable_overnight.sh` + `backfill_maintable_gpu4_7.sh` to
   `backups/20260728_220446_taucut_stop/`.
3. Edited `run_maintable_overnight.sh`:
   - WAIT_A no longer blocks on taucut CSVs; keeps rl_shaped_r075_sh3 only.
   - Added `gpu_wait_idle()` gate so baseline jobs never double-book GPU0 (busy with RL job).
   - Phase B = baseline four methods shard0 (dropped SFT-ours r=0.25).
   - Phase B2 = upgrade those baseline cells to full 4-shard (r0.5 -> r0.75 -> r0.25).
   - Phase C (7B best-effort) unchanged.
4. Killed the OLD orchestrator (pid 13472, was WAIT_A, no pdm children); cleared its lock.
   Launched NEW single orchestrator (pid 15133) -> PHASE_B. Only one pdm dispatcher now
   (backfill pid 3615 only waits on the kept rl_shaped job, then exits; it will not
   relaunch taucut because its round-robin already passed those queue indices).

## Reverse instruction
- `touch STOP_MAINTABLE` to stop baseline dispatch gracefully.
- Restore scripts from `backups/20260728_220446_taucut_stop/` if needed.
- To resume taucut later: rerun the four `MT_rl_taucut_kr060_sh{0,1,2,3}` shards with
  selector=scorer_taucut, keep_ratio=0.5, RL ckpt_best, +agent.tau=-0.1668 +agent.tau_min_keep=36.
