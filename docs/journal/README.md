# `docs/journal/` — daily journals

Free-form daily progress notes. One file per day:

```
docs/journal/YYYY-MM-DD.md
```

Each entry typically captures: what was done, what was learned, what
broke, what's next, and any open questions to discuss.

## Conventions

- **One file per day, named `YYYY-MM-DD.md`**. Multiple topics within a day
  are merged into the same daily file under top-level `## N. <topic>` sections,
  with a "时间索引" (timeline) table at the top.
- Original prose of each topic is preserved verbatim under its section; only a
  TOC + timeline + key-findings preamble is added on top.
- **Exception**: `MA2_b0_navtest.md` is kept at the journal root as a permanent
  per-milestone reference (already cross-referenced from `docs/results/key_results.md`).
  Its content is also integrated into `2026-06-16.md` for daily lookup.

## Files

| File | Contents |
|---|---|
| `2026-06-15.md` | benchmark switch to navtest / MA2 integration map / MA2.3 GRPO smoke debug |
| `2026-06-16.md` | MA2.3 smoke PASS / MA2.5 dispatcher / B0 LOCKED 0.8983 / B0 invalid-token diagnosis / aria2c speedup |
| `2026-06-18.md` | M1.a layer sweep navtest (L\*=27 / L12 tie) |
| `2026-06-24.md` | `.chain_failed` false positive / M1.a Step 5 navtrain probe A PASS (L\*=12) / M1.b₂ Stage 1+2 |
| `2026-06-25.md` | M1.b₂ Stage 3 DONE (19,225 attention dump) / 全日进度报告 / per-scene rank-variance |
| `2026-06-26.md` | B1.0 acceptance 收尾 / C1 V0 dryrun on navtrain_probe100 / V4 spec → REVIEW / T6 Phase 2 §10 open questions |
| `MA2_b0_navtest.md` | (preserved) MA2.b₀ navtest baseline 0.8983 LOCKED |
