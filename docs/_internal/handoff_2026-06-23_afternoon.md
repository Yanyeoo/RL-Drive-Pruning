# Handoff — 2026-06-23 afternoon (for the human PI)

> **Read in 60 seconds.** Skim §1 → §2 → §6. Everything else is reference.
>
> **Authored**: 2026-06-22 21:18 by overnight session, after ~3 h of
> unattended work between user departure and target return time.

---

## 1. TL;DR — what to do when you walk in

**Step 1 (5 sec)**: confirm nothing is on fire
```bash
ps -p 15315 -o pid,etime,cmd                     # overnight watcher should be alive
tail -5 logs/overnight_status.log               # last poll <5 min ago
```

**Step 2 (10 sec)**: read §2 below to pick the next action. There is **one**
recommended action.

**Step 3 (one command)**: launch Phase E2 correctness gate. See §6.

---

## 2. State summary (single source of truth)

| Track | State | Tag / commit | Next action owner |
|---|---|---|---|
| M0 / B0 baseline | ✅ done, PDMS = 89.83 | `key_results.md` §5.1 | — |
| M1.a layer probe (L\*=12) | ✅ done | commit `3d27cc6` | — |
| M1.b₀ per-head landscape | ✅ done, 11 dead heads in L24, 1 in L12 | commit `3d27cc6` | — |
| M1.b Phase C–D (code + spec) | ✅ done, 5 unit tests green | tag `m1b_phaseD_complete` (`1e47e01`) | — |
| **M1.b Phase E2 (5-token gate)** | ⏸ ready-to-launch | script `scripts/run_m1b_phase_e2_gate.sh` | **YOU** (one command, §6) |
| M1.b Phase F (full sweep 4 × 4 h) | ⏸ blocked on E2 PASS | script `scripts/run_m1b_freelunch_sweep.sh` | YOU after E2 passes |
| M1.c LambdaRank scorer (spec) | ✅ spec written | `docs/specs/m1c_lambda_rank_scorer_spec.md` | YOU review |
| navtrain download | 🔁 in progress, rsync `history_split_3 → trainval` ~13 MB/s | watcher PID 15315 | autonomous |
| M0.2 / Phase J (navtrain ready) | ⏸ blocked on download | watcher will detect, will **not** auto-start | YOU when ready |

**No fires. No corrupt state. Everything below is forward progress.**

---

## 3. What I did while you were out

Three new commits, two new specs, one new module, two new launch scripts.

### 3.1 Code (committed and tagged)

- `code/rldrive/agents/head_mask_patch.py` — context manager that zeroes the
  per-head slice of `self_attn.o_proj`'s input via a `forward_pre_hook`.
  Mathematically equivalent to per-head ablation (proof in
  `m1b_freelunch_spec.md` §4.4). 5 unit tests, all exact-equality green.
- `code/rldrive/agents/autovla_with_attention.py` — wired
  `head_mask_layers` + `head_mask_verbose` into the agent ctor; uses
  `ExitStack` to compose with the existing attention-capture context.
- `code/rldrive/configs/agent/autovla_with_attention.yaml` — null defaults
  preserve B0 bit-identity when no mask is given.

### 3.2 Specs (authored *before* code, the rigorous way)

- `docs/specs/m1b_freelunch_spec.md` — 11 §, locks Phase F contract.
  Variants V0 (sanity bit-identity), V1 (1 dead head in L12), V2 (V1 + 11
  dead heads in L24, 12 total), V3 (V2 + light marginal heads, ~18 total).
- `docs/specs/m1c_lambda_rank_scorer_spec.md` — **new**, the methodology
  upgrade after M1.b. 1 M-param MLP scorer trained pairwise on per-token
  attention statistics (LambdaRank). Reuses M1.b V0 outputs as zero-cost
  training data. **This is the source of the paper's main Pareto figure.**

### 3.3 Internal docs (appended, not rewritten)

- `docs/_internal/m1b_per_head_analysis_2026-06-18.md` § A — cross-layer
  landscape sweep table + variant rationale + sanity-reproduction script.
- `docs/results/key_results.md` §§5.4–5.7 — variant table + cost + pending
  follow-ups; 2026-06-22 changelog row.

### 3.4 Launch scripts (idempotent, env-driven, no hidden state)

- `scripts/run_m1b_freelunch_sweep.sh` — Phase F dispatcher. `VARIANTS=...`
  env var picks subset. Writes per-variant `manifest.json`, tags pre/post.
  **Failure of one variant does not abort the chain.** `DRY_RUN=1`
  prints the plan without executing.
- `scripts/run_m1b_phase_e2_gate.sh` — **the one command you run when you
  return.** 5-token V0 + V1 smoke. Auto-diffs V0 vs B0 reference and
  greps V1 logs for the head_mask first-fire signal. Writes
  `docs/_internal/m1b_phaseE2_gate.md` with PASS/FAIL verdict and the
  follow-up command for Phase F.

---

## 4. Why these design choices (for your review)

I had to make several independent technical decisions without you. Here are
the four most consequential, each with the alternative I rejected and why.

### 4.1 Head masking mechanism — `o_proj` pre-hook

| Option | Verdict |
|---|---|
| Edit `q_proj` / `k_proj` to zero rows | ✗ depends on transformers internal naming; fragile |
| Add a `mask` tensor multiplied inside attention | ✗ requires forking the VLM forward; loses upstream patch compatibility |
| **`forward_pre_hook` on `o_proj`** | **✓** mathematically equivalent (head h contribution = `o_proj(concat(...))[:, h*d:(h+1)*d]·W_o[h*d:(h+1)*d, :]`; zeroing input slice = zeroing output contribution); transformers-version-agnostic; reversible |

Proof reproduced in `m1b_freelunch_spec.md` §4.4. Unit tests confirm
bit-identity when mask = empty.

### 4.2 Variants V1/V2/V3 stacking, not interleaving

Reason: cumulative variants give a monotone dose–response curve in the paper
table. Reviewers immediately read "more heads removed → at most this much
PDMS lost". Non-stacked variants would require pairwise comparisons.

### 4.3 LambdaRank instead of regression / RL for the learned scorer

`m1c_lambda_rank_scorer_spec.md` §1 (the table) and Appendix A explain in
full. Short version: regression needs ground-truth ΔPDMS labels which cost
4 h per head; RL has prohibitive sample complexity for a 4-h-per-step
environment. Pairwise ranking with attention-magnitude as weak label
tolerates noisy supervision and trains in minutes.

### 4.4 Full 11576 tokens for Phase F (no early stopping at 100)

You might have suggested running 100 tokens first to save time. I deliberately
locked the full set because:

1. The headline iso-compute claim is on full navtest. Partial-set numbers
   never appear in the paper.
2. 4 h per variant is small overhead given the multi-day human review
   loop. Buying a half-day by skipping a re-run would be a bad trade.
3. Cherry-picking risk: a 100-token subset might accidentally exclude the
   scenes where dead heads matter.

If you disagree, override with `SCENE_FILTER=...` in the sweep env. Spec
permits it.

---

## 5. Background processes (do not kill)

| PID | What | ETA |
|---|---|---|
| 15315 | `overnight_watch.sh` polling every 300 s | runs until navtrain ready or 36000 s ceiling (~10 h from 20:51) |
| 4900, 4939 | `post_dl_chain.sh` (parent + nohup wrapper) | runs until rsync done |
| 4941 | `download_navtrain_robust.sh` | runs until all history splits rsync'd |
| 10602–10604 | `rsync history_split_3/ → trainval/` | ETA ≈ 5–6 h at observed 13 MB/s |

After history_3 finishes, history_4 (174 GB) will rsync next. Total ETA
~12 h. Watcher will log it but **will not auto-launch any M0.2 / Phase J
run**. That decision is reserved for you.

---

## 6. The one command to run when you walk in

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA

# Phase E2 correctness gate — V0 sanity + V1 head_mask first-fire
# 5 tokens, ~3 minutes wall clock.
B0_REF_CSV=<path to your last B0 per-token PDMS csv> \
bash scripts/run_m1b_phase_e2_gate.sh
```

Open `docs/_internal/m1b_phaseE2_gate.md` after it returns. The file
contains the verdict and the follow-up command for Phase F.

### 6.1 Decision tree

```
                          Phase E2 gate
                        /              \
                  PASS                  FAIL
                   |                      |
              Launch Phase F          Read gate doc
              (4 variants serial,     §"Failure mode"
               ~16 h)                  table, decide:
                   |                   - retry with new seed?
              After F:                 - inspect log first?
              read main_table.csv      - rollback to tag
              + pareto_curve.pdf       `pre_m1b_phase_d`?
                   |
              Approve M1.c?
                   |
              Phase G (data
              extraction, ~10 min)
```

### 6.2 Hard failure rollback

If Phase E2 returns garbage (V0 not bit-identical to B0), the M1.b code is
quarantined by the tag `pre_m1b_phase_d`:

```bash
git stash
git checkout pre_m1b_phase_d
```

This drops you back to a known-good state from before any M1.b code change.

---

## 7. Open questions for you (asynchronous, low-priority)

These are *not blocking* — Phase E2 + F can proceed without your answer.

1. **Do you want the V3 variant in Phase F at all?** Spec §3 lists it as
   "light marginal heads"; I picked the threshold conservatively. If you'd
   rather skip V3 to save 4 h, run `VARIANTS="V0 V1 V2"
   bash scripts/run_m1b_freelunch_sweep.sh` after E2.
2. **For M1.c (paper's main Pareto figure)**, do you prefer per-layer
   top-k mask materialisation (my default, safer) or global top-k (more
   aggressive)? Spec §6.1 makes per-layer the default; flip the boolean
   if you disagree.
3. **Push to remote?** I have not pushed any commits. If you want the
   git tags off-machine:
   ```bash
   git push origin main pre_m1b_phase_d m1b_phaseD_complete
   ```

---

## 8. Pointers — full evidence trail

- Top-of-funnel goal: `results/README.md`
- Latest headline numbers: `docs/results/key_results.md`
- Per-head landscape evidence: `docs/_internal/m1b_per_head_analysis_2026-06-18.md`
- Phase F contract: `docs/specs/m1b_freelunch_spec.md`
- Phase G/H/I contract (after Phase F): `docs/specs/m1c_lambda_rank_scorer_spec.md`
- Last 3 incident logs: `docs/_internal/incident_*.md`
- Phase E2 gate logic source: `scripts/run_m1b_phase_e2_gate.sh`
- Phase F sweep logic source: `scripts/run_m1b_freelunch_sweep.sh`

Tag → commit map:
- `pre_m1b_phase_d` = `3d27cc6` (Phase C done, before head_mask code)
- `m1b_phaseD_complete` = `1e47e01` (Phase D done, code + spec, tests green)

---

## 9. What is *not* done (honest gap list)

- **Phase F not started**. Spec locked, script ready. You launch.
- **Phase G/H/I not started**. Spec locked, no code yet (depends on V0 outputs from Phase F).
- **navtrain not finished**. M0.2 / Phase J cannot run.
- **No remote push**. Tags are local-only.
- **No README updated in `results/`**. Pareto curve doesn't exist yet, so
  the file still says "TODO" for the figure. That's correct — it will be
  filled by Phase F + I.

---

End of handoff. Estimated read time 4 min. One command to act.
