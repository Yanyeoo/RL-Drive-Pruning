# Decision proposal — run M1.a attention probe on navtest, NOT navtrain

Date: 2026-06-17 18:15
Status: PROPOSAL (user AFK; flagged for confirmation tomorrow)
Author: AI agent autonomous session

## TL;DR

`implementation_plan.md` says M1.a should run the attention layer-probe on
"navtrain probe set A (100 scenes)". Today's reality:
- navtrain download is far slower than the original RESUME estimate
  (~13 h instead of ~6 h, mostly ceph-fuse tar/rsync IO);
  `.chain_complete` will not land before tomorrow morning.
- navtest data + scene_filters + metric_cache are already on disk
  (`data/navtest_nocot`, `data/navtest_metric_cache`).
- The attention signal we want — `attn[L, q=last_instr, k=vision_tokens]`
  — is a function of (model weights, image+prompt). It does NOT depend
  on whether the scene is train/val/test.

If we accept this, M1.a can START TONIGHT on a small navtest split
(e.g. existing `navtest_smoke5`) instead of waiting for navtrain.

## What does this NOT change?

- M0.2 (navtrain ingest) is still required, just no longer on M1.a's
  critical path. Chain watcher keeps running.
- M1.b (scorer SFT) still needs navtrain because that's where the
  ranking-distill targets come from at scale.
- B0 PDMS=89.83 baseline remains the only locked number.

## What does change?

- M1.a probe set becomes `navtest_smoke5` (≤5 scenes) for the first
  smoke verification (V2/V3/V4 in attention_capture.py TODOs), then
  scale to a 100-scene navtest split if smoke is clean.
- Layer-probe selection (find L* in 0..27) runs on these navtest scenes.
- Final L* gets re-confirmed on navtrain probe A once chain_complete.
  That re-confirm is a cheap sanity check, not a hard gate.

## Risks

1. **Domain shift train↔test for attention pattern**
   Plausible but small: AutoVLA was trained with the same prompt
   format on navtrain, evaluated on navtest, and got near-paper PDMS.
   So the model's vision-text attention pattern shouldn't be wildly
   different between the two splits. Quantifiable later (the
   re-confirm step).
2. **navtest_smoke5 too small for layer-probe**
   Yes — smoke is just for V2/V3/V4 sanity. Real layer-probe needs
   ~100 scenes. We have `navtest_local_filtered` (1340 scenes used
   for B0); we can hash-shard out a 100-scene probe (mirror what
   `run_autovla_navtest_dual_gpu.sh` already does for navtest).
3. **Compute waste**
   None — we'd run identical inference passes either way; only the
   scene token list differs.

## What I'm doing while waiting for confirmation

Continuing pure-code work that is reusable under EITHER decision:
- E3 (unit tests) ✅ DONE
- E4 (prereqs note) writing this file
- writing the M1.a runner shell with knobs that accept BOTH
  navtest and navtrain scene_filter names — same script either way

I will NOT launch any GPU inference run until user confirms the pivot
or re-affirms the original navtrain-first plan tomorrow morning.

## Question for the user

Do we:
  (a) START M1.a smoke on navtest tonight (my recommendation), then
      re-confirm L* on navtrain probe A once chain_complete; or
  (b) WAIT until navtrain chain_complete and run M1.a on navtrain
      probe A as the original plan said?

Default if no answer: (b). I will not touch the GPU until told.
