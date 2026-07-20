# HANDOFF — 2026-06-18 16:00, session death imminent (17:00)

**TO: next AI agent**
**FROM: previous AI session (will die at 17:00, no memory carry-over)**
**USER context: 4×H20 will shrink to 2×H20 after 17:00.**

---

## ⚠ READ THIS FIRST (3 critical things that will bite you)

1. **DO NOT touch the navtrain background processes.** They are healthy and self-recovering. List below.
2. **DO NOT trust `du`/`ls` to judge rsync progress.** ceph-fuse rsync looks stalled for 30+ min while it does metadata stat. The previous session falsely diagnosed a "stall" earlier today and wasted hours on a misguided takeover. See `incident_2026-06-18_false_stall_diagnosis.md`.
3. **The user's #1 priority right now is "don't waste H20 GPUs."** GPUs have been idle all day waiting for navtrain. **You should pivot to M1.a on navtest immediately** — this was already decided by the user in this session (see "USER DECISION" below).

---

## USER DECISION (locked in this session, 15:56–16:00)

User confirmed: **pivot M1.a to navtest, do NOT wait for navtrain.**

This is exactly proposal (a) from `decision_proposal_2026-06-17_m1a_on_navtest.md`:
- navtest data is fully on disk (`data/navtest_nocot`, `data/navtest_metric_cache`, `sensor_blobs/test`).
- Attention signal `attn[L, q=last_instr, k=vision_tokens]` does NOT depend on train/test split. AutoVLA was trained on navtrain, evaluated on navtest with near-paper PDMS — so vision-text attention pattern is essentially the same on both.
- navtrain rsync (background) is irrelevant to M1.a critical path. Final L* will get a 10-min sanity re-confirm on navtrain probe A once `.chain_complete` lands (estimated tomorrow morning).

User's exact words: "不要浪费4卡H20" (now 2-card H20 after 17:00, but the principle stands).

**Net: launch M1.a smoke on navtest_smoke5 ASAP. Don't re-litigate this decision.**

---

## Background processes — current state (verified 16:00)

| PID | What | Status | Action |
|---|---|---|---|
| 3738 | `download_navtrain_robust.sh` (main script) | Doing rsync of `history_split_1/` → `trainval/` since 13:17. Expected to finish ~17:30–18:30. | LEAVE IT |
| 3774 | `post_dl_chain.sh` (chain watcher polling for `.download_complete`) | Polling, ~21h+. | LEAVE IT |
| 139364/366/367/369 | `rsync -a history_split_2/ → trainval/` (user-confirmed takeover, started 14:55) | Running, slow (metadata-bound on ceph-fuse). | LEAVE IT |

**Do not kill any of these.** They will eventually converge to `.download_complete` → chain triggers `install_navtrain.sh` (same-disk `mv` = seconds) → `m02_splits.sh` → done.

**If they're still running tomorrow:** that's expected. Don't intervene unless they actively error.

---

## navtrain status (DON'T re-verify by running rsync diagnostics)

- ✅ All 8 tgz downloaded from S3 (current 1-4 + history 1-4)
- ✅ current 1-4: fully installed (sentinels `.navtrain_current_*.installed` exist)
- 🟡 history_1: main script rsyncing right now (~5h ETA, in progress since 13:17)
- 🟡 history_2: takeover rsync running (started 14:55, will eventually touch `.navtrain_history_2.tgz.installed`)
- ⏳ history_3, history_4: staging dirs exist (`_staging_navtrain/history_split_3`, `_staging_navtrain/history_split_4`, ~35GB each). Main script will rsync them in sequence after history_1 finishes. They will be slow (5h+ each on ceph-fuse) but **this is fine, GPUs don't need them for M1.a**.

**Sentinels to watch (don't poll aggressively, every 30min is plenty):**
```
ls /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain/trainval_sensor_blobs/.navtrain_*.installed
ls /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain/trainval_sensor_blobs/.download_complete
```

---

## Disk: 890G/2T used. Safe (peak was 945G earlier, has been declining).

---

## M1.a — what to do (priority order)

### Step 1: pre-flight (5 min) — DO THIS FIRST

Verify these without launching anything:

```bash
# 1. smoke filter exists
find /apdcephfs/private_shayladeng -maxdepth 6 -name "*smoke5*"

# 2. ckpt + conda env
ls /apdcephfs/private_shayladeng/ckpt/
/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python -c "import torch; print(torch.cuda.device_count())"

# 3. probe runner sanity
head -60 /apdcephfs/private_shayladeng/tokenrl_autoVLA/code/rldrive/scoring/run_attention_probe.py

# 4. GPU free?
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

If anything is missing, **stop and ask the user** — do not improvise (this session's previous AI improvised twice and broke things).

### Step 2: smoke (≤5 scenes, ~10 min) — Step 1 must pass first

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
bash scripts/run_m1a_attention_probe.sh \
    --scene-filter navtest_smoke5 \
    --save-dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer14_smoke \
    --layer-idx 14 \
    --max-scenes 5 \
    2>&1 | tee /apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/m1a_smoke_$(date +%Y%m%d_%H%M).log
```

Expected output: V2/V3/V4 sanity asserts pass, attention tensor `[B, H, T_q, T_k]` shape correct, saved tensors land in `save-dir`.

If smoke passes → Step 3. If fails → diagnose (`run_attention_probe.py` and `attention_capture.py` in `code/rldrive/scoring/`).

### Step 3: 100-scene navtest layer sweep (2 cards parallel, ~2-3h)

After smoke is clean. See `m1a_prereqs.md` for the 100-scene hash-shard recipe (mirror what `run_autovla_navtest_dual_gpu.sh` does).

Layer sweep: layer-idx in {0, 4, 8, 12, 14, 16, 20, 24, 27}. **Don't sweep all 28 — too expensive.** Pick L* = argmax(vision-attention fraction).

### Step 4: write L* decision doc + commit

Once L* selected, write `docs/_internal/m1a_layer_selection_2026-06-1X.md` with: scene count, layer scores, chosen L*, comparison plot if cheap.

### Step 5: re-confirm on navtrain probe A (after `.chain_complete`)

10-min sanity. If L* shifts by ≤2 layers, accept. If shifts more, escalate to user.

---

## Hard rules for next AI

1. **No GPU launches before pre-flight passes.**
2. **No killing background processes.**
3. **No "improvements" to navtrain download/rsync.** It's running, leave it.
4. **No new tar/rsync/mv operations on `_staging_navtrain/`.** Previous AI broke staging twice today.
5. **If unsure, STOP and ask user.** User explicitly preferred conservative behavior throughout today's session.

---

## Files the next AI should read in order

1. This file (`handoff_2026-06-18_session_death.md`)
2. `decision_proposal_2026-06-17_m1a_on_navtest.md` — the M1.a-on-navtest rationale
3. `m1a_prereqs.md` — what's needed before launching
4. `m1_attention_hook_design.md` — what V2/V3/V4 asserts mean
5. `incident_2026-06-18_false_stall_diagnosis.md` — don't repeat my mistake
6. `implementation_plan.md` (project root) — overall roadmap, M0/M1/M2 milestones

---

## Open questions parked for user

- (Tomorrow) Confirm L* re-check on navtrain probe A or skip if shift is small?
- (No urgency) `history_takeover_v2_h2.log` exists; previous AI's failed v1 takeover (history 1/2/3/4) is documented as `history_takeover_{1..4}.log`. Safe to ignore, kept for forensics.

---

End of handoff. Good luck.
