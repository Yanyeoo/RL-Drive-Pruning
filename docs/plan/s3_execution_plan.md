# S3 Execution Plan — 168 GPU-h/week, ≤24h windows, memory-loss-safe

> **Created**: 2026-07-06 15:xx. **Status**: PROPOSAL (awaiting user go/adjust).
> **Operating constraints (user, 2026-07-06)**:
> - Budget: **168 H20 card-hours / week**.
> - A single window (session) is **≤ 24h wall-clock**.
> - **Memory is lost between windows** → every window MUST be self-contained,
>   resumable from checkpoints, and end by writing a fresh handoff.
> **State at plan time**: S1 executor ✅ (lossless). S2 gate ✅ PASS. S3 Slice-1
> (SFT token scorer, `ckpt/s3_token_scorer/`) ✅ PASS — beats attn_L12 at r≤0.5,
> r=0.5 PDMS 0.8953 > r=1.0 0.8914; scorer-oracle 0.9167 @ 30% keep (+2.15pt
> headroom for a budget policy). See `key_results.md §10`.

---

## 0. Cost model (measured this cycle)
`card_hours = N_scenes × sec/scene ÷ 3600`. Measured: 2-pass eval **4.5 s**,
1-pass dump / r=1.0 **2.4 s**, GRPO rollout generate **~3 s**.

| job | N | card-h |
|---|---|---|
| eval 1 arm, full navtest | 11574 | 14.5 |
| eval 1 arm, shard0 (dev) | 2947 | 3.7 |
| eval 1 arm, 1500-subset (dev) | 1493 | 1.9 |
| navtrain feature/r-sweep 1 arm (2-pass) | 2500 | 3.1 |
| navtrain full feature dump (1-pass) | 19225 | 12.8 |
| GRPO one pass, K=8 | 2000 | ~13.3 |

**Windows/week**: assume 4×H20/window ⇒ ~20h wall ≈ 80 card-h/window ⇒ **~2
windows/week** to stay ≤168. (Scripts auto-detect card count at boot.)

---

## 1. Window-safety infrastructure (BUILD FIRST — the memory-loss linchpin)

Every window follows this protocol; the driver automates it.

1. **Boot** (read-only, 5 min): read `docs/journal/HANDOFF_latest.md` → reconstruct
   state. `nvidia-smi` + `pgrep` guards (idle cards? residual procs? double-run?).
2. **Backup**: `cp -a` key artifacts → `backups/cycle_start_<ts>/`.
3. **Driver** `scripts/run_s3_window.sh` (nohup) runs a task queue where each task is:
   - **idempotent** (`SKIP_DONE=1`, per-scene CSV / per-step ckpt already-done skip),
   - **pgrep-guarded** (no `run_pdm_score_cot` double-run),
   - bounded by `DEADLINE_H` (auto-stop ~1h before the 24h edge) + `STOP_S3` sentinel.
4. **Auto-finalize** (driver on exit, even on deadline/stop): aggregate → append to
   `key_results.md`; write `docs/journal/HANDOFF_<date>_next.md` (exact next commands
   + state); `cp -a` → `backups/cycle_close_<ts>/`.
5. **GRPO resumability** (critical): checkpoint every `CKPT_EVERY` steps to
   `ckpt/<run>/step_XXXXX.pt` + `train_state.json` {step, optimizer, RNG,
   per-scene EPDMS baselines, best-val}. Driver auto-detects latest → resumes.
   ⇒ a killed window + wiped memory loses at most `CKPT_EVERY` steps.

**Deliverable of infra**: `scripts/run_s3_window.sh`, a task-queue format, a
GRPO `Trainer` with checkpoint/resume, and the `HANDOFF_latest.md` convention.

---

## 2. Milestones — value-first, fact-gated (do cheap+decisive before expensive)

> **UPDATE 2026-07-06 (user decision A)**: Phase A ran → **claim② NEGATIVE**
> (key_results §11; 6 configs incl. full 720-score profile all < fixed r=0.5).
> **Phase B (budget GRPO) and Phase C (scorer GRPO) are DROPPED/deprioritized.**
> Method fixed = dynamic token scorer + **fixed r=0.5**. Remaining work = **Phase D
> (M6)**: full-navtest main table (running) + FastV baselines + ablations + figures.
> claim② = honest negative/ablation. See design_decisions Revision 2026-07-06.

### Phase A — Budget Policy (Stage B) via oracle-SFT  [claim② adaptive>fixed]  ≈ **15 card-h**
Cheapest high-value step. Proves the core novelty lever (per-scene budget).
- **A1** r-sweep oracle labels: PDMS at r∈{.25,.5,.75} on **navtrain 2500** using
  the SFT scorer as selector (r=1.0 = 1-pass reuse). Dump `scene_ctx` (nav_cmd/
  ego_speed/driving_instr) during the runs. ≈ 11 card-h.
- **A2** ε-scan (Q4.2.b) → ε\*; build 4-class budget labels (post-proc, free).
- **A3** Budget Policy SFT: MLP(scene_ctx)→4-class CE (minutes, free).
- **A4** Eval **adaptive** (scorer + budget policy) on shard0; compare vs best
  fixed-r (reuse existing scorer fixed-r CSVs) + vs scorer-oracle ceiling. ≈ 3.7 card-h.
- **GATE**: adaptive PDMS − best-fixed ≥ ~0.5pt ⇒ claim② holds; else analyze
  (class-imbalance: 83% scenes want r=0.25 → policy must learn *which* scenes
  need more, this is the real test).

### Phase A' — Harden current results on full navtest (fold into M6)  ≈ **43 card-h**
SFT scorer at r∈{.25,.5,.75} on **full navtest** → definitive claim①/③ numbers
(removes the 1500-subset caveat). Do lazily / as part of M6, not on the critical path.

### Phase B — Budget Policy GRPO (M5, R3 piecewise-Pareto reward)  ≈ **~65 card-h**
Only if A's SFT leaves a gap to oracle. α/β scan on probe-100 (Q4.3.d) → full
GRPO on 2000 scenes, K=8, ~4 passes, checkpointed. Reward per design Q4.3.b/c/e.

### Phase C — Scorer GRPO (M2, pure EPDMS-advantage)  ≈ **~55 card-h**  [optional]
Lower priority: SFT scorer already beats attn_L12. Do only if scorer quality is
shown to be the bottleneck (not budget). Checkpointed.

### Phase D — M6 final: main table + Pareto + ablations  ≈ **~100 card-h**
Full navtest: Ours-full (scorer+budget), Ours-fixed, scorer-fixed, attn_L12,
random, FastV, FastV-selector-at-input, at key ratios; Pareto sweep; A1–A8/A10
ablations (mostly reuse). Spread over ~2 windows.

---

## 3. Week schedule (each window ≤24h, ~2/week, ≤168 card-h/week)

| Week | Window | Content | card-h |
|---|---|---|---|
| 1 | **W1** | Build §1 infra + **Phase A** (Budget Policy SFT) → claim② result | ~20 |
| 1 | W2 | **Phase B** (Budget Policy GRPO), checkpointed | ~65 |
| 2 | W3 | finish B; decide C by A-vs-oracle gap; α/β analysis | ~55 |
| 2 | W4 | **Phase D** M6 full-navtest main table + Pareto (part 1) | ~80 |
| 3 | W5 | M6 ablations + Phase A' hardening + figures | ~80 |

Re-plan after W1 with real numbers (GATE-driven; skip B/C if A already captures
most headroom).

---

## 4. First window (W1) recipe — self-contained
1. Boot checks (nvidia-smi/pgrep) + backup.
2. Build: `scripts/run_s3_window.sh` (driver), GRPO trainer skeleton w/ ckpt/resume,
   `HANDOFF_latest.md` convention.
3. Phase A jobs (queued, resumable): r-sweep oracle (navtrain 2500) → ε-scan →
   Budget Policy SFT → adaptive eval on shard0.
4. Finalize: `key_results.md` new §11; `HANDOFF_2026-…_W2.md`; backup.

**Decisions needed from user before W1**:
- Q1: approve this phase order (A→B→D, C optional)? 
- Q2: train scorer/budget on the current 4000-scene subset, or expand navtrain
  first (+scene_ctx interaction feature)?  [proposal: subset now, expand only if needed]
- Q3: confirm 4×H20 per window (scripts adapt to fewer).
