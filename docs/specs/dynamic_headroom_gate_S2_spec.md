# S2 Spec — Token-Pruning Headroom Gate (fixed-r Pareto + per-scene oracle)

> **Status**: DRAFT v1 (2026-07-03) — for user review **before running**.
> **Depends on**: S1 Variant A (attention-mask pruning executor) passing its r=1.0 lossless gate.
> **Purpose**: This is the **decisive gate** for the whole dynamic-pruning direction. It answers, with NO training, two make-or-break questions:
>   1. **Headroom**: does token pruning preserve PDMS at iso-compute (r=0.5)? (Q1-B 底线)
>   2. **Scene-adaptivity**: does the *optimal* keep-ratio vary across scenes? (claim② premise)
> If both fail, dynamic budget cannot beat a fixed ratio → report to user, do NOT build S3.

---

## 1. Design

### 1.1 Selector (no training)
Per-scene token importance score `s ∈ R^N` = **L*=12 attention** row (query=last instruction token, key=vision tokens), head-averaged — captured live in pass-1 via the existing `patch_attention_capture(vlm, layer_idx=12, ...)`. This is the *same* signal M1.a selected as best-aligned, and is the honest "attention selector" that S3's trained scorer must later beat.

Rationale for live capture (not the navtrain dump): the dump is **navtrain**; S2 evaluates on **navtest**, so importance must be computed on-the-fly per navtest scene. 2-pass (capture → prune → generate) via `AutoVLAWithTokenPruneAgent`.

### 1.2 Arms (all on the SAME navtest set, 4-shard, n≈11574, matching §6 protocol)

| arm | selector | keep_ratio r | # runs | purpose |
|---|---|---|---|---|
| **P-attn** | attn_L12 | 0.25, 0.5, 0.75 | 3 | fixed-r Pareto with a *good* selector |
| **P-rand** | random (seed-fixed) | 0.5 | 1 | lower bound; is the selector doing anything? (= design baseline #2) |
| ref | — | 1.0 | 0 | reuse B0=0.8983 (r=1.0 lossless proven in S1) |

(4 GPU-runs total. r=1.0 reused. Random only at r=0.5 to bound the iso-compute column.)

### 1.3 Per-scene oracle (the scene-adaptivity test) — POST-PROCESSING, no extra GPU
For each scene, we have PDMS at r∈{0.25,0.5,0.75,1.0} (P-attn arms + ref). Compute per scene:
- `r*_ε = min{ r : PDMS_r ≥ max_r PDMS_r − ε }` for ε∈{0.005,0.01,0.02} (the B3 oracle rule, reused from Q4.2).
- **Distribution of r\*** across scenes = the key output. If ≥~20–30% of scenes have r\*<1.0 AND r\* is not collapsed to a single value → scene-adaptive headroom EXISTS.
- Also report: oracle-EPDMS (pick best r per scene) − fixed-best-r EPDMS = the **ceiling gain** a perfect budget policy could capture. This is the number that justifies S3.

---

## 2. Metrics reported
- Per arm: PDMS (4-shard weighted mean) + 4 sub-metrics (collision/comfort/progress/ttc), n_valid.
- **Pareto table**: PDMS vs r (0.25/0.5/0.75/1.0) for P-attn; mark the iso-compute r=0.5 point.
- P-attn(r=0.5) vs P-rand(r=0.5): selector gain at iso-compute.
- Oracle: r\* histogram per ε; oracle-EPDMS; ceiling gain over best fixed-r.

## 3. Decision criteria (the GATE)

| outcome | condition | action |
|---|---|---|
| **PASS → build S3** | P-attn(r=0.5) drop ≤ ~0.5 PDMS pt (Q1-B) **AND** oracle ceiling gain over best fixed-r ≥ ~0.5 pt **AND** r\* not single-valued (real scene variance) | proceed to S3 (scorer+budget+GRPO) |
| **PARTIAL** | headroom OK but r\* nearly constant (no scene variance) | dynamic budget adds little → pivot to "fixed-ratio + better selector" story (scorer still worth it, budget policy not); revisit with user |
| **FAIL → stop** | P-attn(r=0.5) drops ≫0.5 pt, i.e. token pruning itself hurts | token pruning has no headroom on AutoVLA/navtest (R-D-3 fires) → report to user; dynamic goal not viable on this backbone |

## 4. Compute plan
- Reuse the exact sweep infra (`run_m1b_phaseF_2gpu.sh` pattern) with `agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent` and `+agent.keep_ratio=` / `+agent.selector=`.
- 4 runs × 4-shard on 2× H20. 2-pass ⇒ ~1.5–2× the per-scene time of a plain sweep; budget ~ 4 × (3.5h × 1.8) ≈ **~25 GPU-h** ≈ fits ~2 unattended windows on 2 cards. (Refine after a 5-scene smoke gives real s/scene.)
- **Scheduling**: queued AFTER the in-flight L0K4 landscape run (user's choice q2). Can reuse a driver-style unattended loop with a fresh deadline once user approves running.

## 5. Deliverables of S2
- `results/raw/M1b_tokenprune_{attn,rand}_r*_*` (same aggregate.json/manifest.json schema).
- `docs/results/` Pareto + oracle-r\* figures; a new key_results section (§8 "Token-pruning headroom").
- A go/no-go note in journal keyed to §3 criteria.

## 6. Resolved decisions (user review 2026-07-03)

1. **r grid = {0.25, 0.5, 0.75}** (+1.0 reused). No finer grid. ✅ locked.
2. **ε (oracle tolerance)**: post-processing only in S2 (NOT training, no extra GPU). **Primary ε=0.01**, report r\* histogram also at {0.005, 0.02} for robustness. Formal ε\* scan (Q4.2.b) deferred to S3 label-building. ✅ locked.
3. **4th selector (FastV-at-input / ViT-L2-norm): NOT in the gate.** Rationale: the gate is a cheap go/no-go answerable by `attn_L12 + random + oracle`; extra selectors are M6 main-table baselines that (a) are needed anyway *iff* the gate passes, and (b) each costs a full pass-2 generate run. Running them now would burn GPU before the direction is proven. → **deferred to post-gate / M6.** ✅ locked.
4. **S3-justification bar**: oracle ceiling gain over best fixed-r **≥ ~0.5 PDMS pt** (mirrors Q1-B ε-budget). Revisit if borderline. (pending final confirm, treated as working threshold.)

**⇒ Final gate = exactly 4 GPU-runs**: `P-attn` at r∈{0.25,0.5,0.75} + `P-rand` at r=0.5. r=1.0 reused (B0). Oracle r\* is post-processing over these + B0.