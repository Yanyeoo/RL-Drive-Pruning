# M1.b₂ Phase 2 — Learned Per-Scene Head-Gating Policy

**Status**: DESIGN (not yet implemented)
**Author**: AI agent session 2026-06-25 evening
**Predecessors**: M1.b₀ (per-head structure), M1.b₁ (Level-0 free-lunch on navtest), M1.b₂ Phase 1 (per-head attention dump on navtrain)
**Inputs ready**: `exp/m1b2_navtrain_full_alllayers/*.pt` — 19,225 tensors of shape `(28, 16, 720)` (per-layer × per-head × vision-token attention, fp32)
**Owner / Reviewer**: hand to user before any GPU run

---

## §0. TL;DR

M1.b₁ proved a **static** L12:{h13} mask is free-lunch on navtest (Δ PDMS = −0.0004 @ 0.39% KV saving). M1.b₂ Phase 2 asks: **can a learned policy that gates heads per-scene push the Pareto frontier further** (e.g. 1–5% KV saving with ΔPDMS still within noise)?

Input is the `(N=19225, L=28, H=16, V=720)` attention tensor we just produced + the pretokenized scene metadata in `data/navtrain_nocot/<scene_id>.json`. Output is a policy `π(scene_metadata) → mask ∈ {0,1}^(L×H)`.

Three architectures considered (linear / MLP / per-layer factorized), trained via either **(a) supervised pseudo-labels from attention rank** or **(b) RL with PDMS as reward**, with a **mixed warm-start** as the recommended path.

**Acceptance gate**: at least match M1.b₁ V1 (Δ PDMS ≥ −0.001) while delivering **> 0.39% KV saving on average** (ideally 1–2% with no per-scene catastrophic regression).

**Compute budget**: ~2 GPU-h supervised prototype + 4–8 GPU-h RL fine-tune + 1 GPU-h eval per checkpoint. Fits inside one 14h window with margin.

---

## §1. Motivation

### 1.1 What we have

| Milestone | Result | Reference |
|---|---|---|
| **M1.a** | `L* = 12` selected (highest vision_frac, n=500) | `m1a_layer_selection_2026-06-18.md` |
| **M1.b₀** | Per-head dead-head structure exists at L12, L24, L27 (structural, not sampling noise) | `m1b_per_head_analysis_2026-06-18.md` |
| **M1.b₁ V1** | Static L12:{h13} mask → PDMS 0.8981 vs B0 0.8983 (**Δ = −0.0004**, free-lunch ✅) on full navtest n≈11574 | `key_results.md §6` |
| **M1.b₂ Phase 1** | Per-head attention dumped on full **navtrain_avail19k** (19,225 tokens × (28, 16, 720)), 3h16m wall, 0 OOM | `2026-06-25_m1b2_stage3_done.md` |

### 1.2 Why per-scene gating

M1.b₁ uses a **single global mask**, identified from population-level rank statistics. Two structural observations motivate Phase 2:

1. **Head-importance is scene-dependent** — M1.b₀ §3 already showed that top-4 heads at L12 are not top-4 at L27 (cross-layer disjoint). It is plausible (untested) that within L12, the top-K heads also shift across scenes (e.g. crowded urban vs empty highway).
2. **V2 cliff at L27** (M1.b₁) — masking L27:{h11,h3,h10,h1} costs 4.4 pp PDMS. A per-scene policy could skip the L27 mask on scenes that need it and apply it on the (majority) of scenes that don't, recovering the saving without the cliff.

If head-importance were uniformly distributed across scenes, a learned policy collapses to the M1.b₁ static optimum and we lose nothing. If it varies, we win.

### 1.3 Why now

- Phase 1 just produced the **only** unblocking input (19,225 per-head attention tensors).
- Without Phase 2 we cannot beat M1.b₁ on the saving/quality Pareto.
- Eval is cheap (~2.5 s/scene mirror of Phase 1; n=11574 navtest in ~8 GPU-h on 4× H20).

---

## §2. Problem Formulation

### 2.1 Spaces

| Symbol | Meaning | Shape |
|---|---|---|
| `s` | Scene (= one log_id) | — |
| `x(s)` | Scene representation (TBD in §2.2) | `(D_s,)` |
| `A(s)` | Per-head attention on vision tokens | `(L=28, H=16, V=720)` fp32 |
| `m` | Head mask | `{0,1}^(L=28, H=16)` |
| `r(s, m)` | PDMS reward when running AutoVLA on scene `s` with mask `m` | scalar ∈ [0, 1] |
| `π_θ(m | x(s))` | Learned policy | — |

### 2.2 Scene representation — candidates

We must decide **what `x(s)` is**, because it determines what information the policy can condition on. Three candidates:

| ID | `x(s)` | Pros | Cons |
|---|---|---|---|
| **R1** | Prompt embedding = AutoVLA `hidden_states[L*=12]` mean-pooled over vision tokens, shape `(D=4096,)` | Free (already inside our probe), reuses the layer we trust | Requires running AutoVLA forward to get it — fine for offline; need cache |
| **R2** | Per-head attention statistics: `A(s)` mean over `V`, shape `(L*H=448,)` | Cheapest, comes from our cached `.pt` directly | Risks circularity: we're using attention-of-itself to predict mask-of-itself |
| **R3** | Scene metadata embedding from `data/navtrain_nocot/<id>.json` (prompt text, ego state, etc.) | Independent of model internals; aligns with potential real-time deployment (no forward needed) | Most expensive feature engineering; lowest signal density |

**Recommendation: start with R1.** R2 is the cheapest but circular. R3 is the cleanest research story but adds engineering scope. R1 is the standard "use the LM's own representation" baseline — if R1 fails we have a clean negative result that informs whether to try R3.

> **Open Q1** for reviewer: confirm R1 (mean-pooled L12 hidden state of vision tokens) is the right starting representation. Alternative: use the `<answer>` token's hidden state instead of mean-pool.

### 2.3 Action space — sparsity constraint

Policy outputs `m ∈ {0,1}^(L×H)`. We must bound the action space, else `m = 1` (mask nothing) is always optimal w.r.t. PDMS.

Three options:

| ID | Constraint | Effective branch factor |
|---|---|---|
| **C1** | Fixed cardinality `‖1 − m‖_0 = k` (mask exactly `k` heads per scene) | `C(448, k)`, e.g. k=4 → 1.7e9 |
| **C2** | KV-saving ≥ τ (continuous threshold on `Σ (1-m_l,h) · cost_l,h`) | larger |
| **C3** | Per-layer cardinality cap: at most `k_l` heads masked per layer | smaller |

**Recommendation: C1 with k = 4**, mirroring the V2 head count in M1.b₁ (which was 4 heads at L27). This anchors comparison to a known M1.b₁ point and makes the Pareto plot direct.

### 2.4 Objective

```
maximize  E_s [ r(s, m) ]  subject to  m ~ π_θ(· | x(s)),  ‖1−m‖_0 = k
```

In M1.b₁ V1 language: hold KV saving at the V1 level (= 1 head) or push to V2 level (4 heads) and **beat** static masking on PDMS.

---

## §3. Data

### 3.1 Asset inventory

| Asset | Size | Path | Status |
|---|---|---|---|
| Per-head attention | 19,225 × `(28, 16, 720)` fp32 = ~24 GB | `exp/m1b2_navtrain_full_alllayers/*.pt` | ✅ Phase 1 (today) |
| Scene metadata (prompt + ego) | ~19,225 JSON | `data/navtrain_nocot/<id>.json` | ✅ from M0 |
| Trajectory denylist (8 scenes) | 8 ids | `exp/m1b2_navtrain_full_alllayers/_stage3_trajectory_err_tokens.txt` | ✅ Phase 1 |
| AutoVLA hidden state at L12 (for R1) | not cached | — | ⏳ **need to dump separately** |

**Action**: if we go R1, we need a small probe extension that saves `hidden_states[12].mean(dim=vision_tokens)` per scene. Estimate: same loop as Phase 1, ~3 GPU-h on 4× H20. Or we can fuse into the same forward pass next time. **For prototype**, dump on a 2k-scene subset first.

### 3.2 Splits

By **log_id**, not by scene_id, to prevent leakage (a single log gives many scenes with near-identical visuals).

- Read `log_id` from each `<scene>.json`.
- Hash `log_id` → 3-way split: train 70% / val 15% / test 15%.
- Drop the 8 denylist scenes from all splits.
- Net: ~13,452 / ~2,882 / ~2,883 scenes (approx, exact after dedupe).

Persist the split as `splits/m1b2_phase2_split_v1.json` (log_id-keyed) so all downstream runs use identical partition.

### 3.3 Pseudo-labels (for supervised warm-start, §5.1)

For each scene `s`, compute the **head importance score** as
```
score(l, h, s) = mean over V of A(s)[l, h, :]
```
Then per-layer top-K is preserved, bottom-K is the candidate mask. Stored as `(N, L, H)` ranks tensor, ~7 GB int16, generated once on CPU.

> **Updated 2026-06-25 18:30 from rank-variance prior** (`docs/journal/2026-06-25_m1b2_rank_variance.md`):
> Naive "bottom-k by score" pseudo-labels carry **two** risks revealed by 19,225-scene analysis:
> (1) For layers with low top-1 entropy (e.g. L0–L2, L5, L8, L16, L22, L24 < 10% normalized entropy) the bottom-K set is **scene-invariant** — supervised label is degenerate, will collapse to static. (2) For L24 specifically, 7 of the 11 heads M1.b₁ V3 masks have **bot-K freq < 50%** on navtrain — naive labels there are noisy.
> **Refinement**: per-layer, train only when top-1 entropy > 1.0 bit (= 25% normalized). For low-entropy layers, use a frozen M1.b₁-style static mask as policy prior. For L24 specifically, use bot-K freq > 0.9 to filter the candidate pool (= 3 heads: h7, h9, h10), then let policy choose among them.

### 3.4 Per-scene rank-variance prior (NEW 2026-06-25 18:30)

Completed independent analysis of the 19,225 tensors before Phase 2 training; results in `exp/m1b2_rank_variance/`. Key takeaways for §4 / §6 below:

| Tier | Layers | Phase 2 implication |
|---|---|---|
| High top-1 entropy (> 40% norm) | L7, L9, L12, L13, L20, L21 | **Primary target** — biggest headroom for per-scene gating |
| Moderate (25–40%) | L3, L10, L11, L14, L18, L19, L23, L25, L27 | Secondary target |
| Low (10–25%) | L4, L6, L15, L17, L26 | Skip in v0, revisit later |
| Static (< 10%) | L0–L2, L5, L8, L16, L22, L24 | Use static mask, no policy |

V1 (L12:h13) is provably 100% bot-K on 19,225 scenes → static V1 is **already optimal** on this layer's bottom. Phase 2 must claim its wins elsewhere (L20, L21, L24-with-pool, L13).

---

## §4. Policy Architecture — three candidates

All three map `x(s) ∈ R^{D_s}` → logits `z ∈ R^{L × H}`, then apply Gumbel-top-K (k=4) to get differentiable mask.

| ID | Form | Params | Risk |
|---|---|---|---|
| **P1 Linear** | `z = W · x + b`, `W ∈ R^{(L·H) × D_s}` (= 448 × 4096 = 1.8 M) | 1.8 M | Underfits if relation is non-linear |
| **P2 Small MLP** | 2-layer: `D_s → 512 → 448`, GELU | 2.3 M | OK default |
| **P3 Per-layer factorized** | `z_l = U_l · (V_l · x)`, decouples layers, share-bottom | 0.5 M | Best regularizer if structure is per-layer |

**Recommendation: train P1 and P2 in parallel as baselines, treat P3 as ablation.** All three are tiny — fit comfortably on 1 GPU.

### Differentiable top-K

Use **Gumbel-softmax with k-hot relaxation** (Plöger et al.) with temperature `τ` annealed 1.0 → 0.1 over training. At inference, take hard top-K.

---

## §5. Training Objective

### 5.1 Supervised pseudo-label loss (warm-start)

Pseudo-target: per scene, the **bottom-k heads by `score(l, h, s)`** (from §3.3) become the "should-mask" set. Loss:

```
L_sup = BCE( σ(z), 1 − m_pseudo )
```

This is **not** the real objective (PDMS) but it teaches the policy to track attention rank, which we already know is a strong (free-lunch confirmed at V1) signal. Convergence: minutes on 1 GPU.

### 5.2 RL fine-tune

After warm-start, fine-tune with **REINFORCE + baseline**:

```
∇θ J = E_s [ (r(s, m) − b(s)) · ∇θ log π_θ(m|x(s)) ]
```

- `r(s, m)` = PDMS of AutoVLA(s) under mask `m`, computed by reusing the M1.b₁ scoring pipeline (`scripts/run_m1b_freelunch_sweep.sh` adapted to take a per-scene mask file).
- Baseline `b(s)` = mean reward across 4 sampled masks per scene (anti-variance).
- 1 step = 64 scenes × 4 samples = 256 forwards ≈ 11 min on 4× H20.
- Target: 50–100 steps = ~10–20 GPU-h.

**Reward shaping (optional)**: `r = PDMS + λ · (k − ‖1−m‖_0)` if we relax C1 to "at most k".

### 5.3 Mixed (recommended)

Pipeline:
1. Train P1/P2 supervised for 5 epochs (~30 min on 1 GPU) → `θ_0`.
2. From `θ_0`, RL fine-tune 50 steps (~10 GPU-h on 4 GPU).
3. Evaluate against M1.b₁ V1 and V2.

---

## §6. Acceptance Gates

The Phase 2 prototype **passes** iff **all** hold on the navtrain test split (n ≈ 2,883):

| Gate | Threshold | Justification |
|---|---|---|
| **G1** Mean PDMS ≥ M1.b₁ V1 − 0.001 (= 0.8971) | strict | Cannot regress below static free-lunch |
| **G2** KV saving ≥ 0.39% (V1 level) | strict | At minimum match the static saving |
| **G3** No per-scene PDMS drop > 0.05 (vs B0 on same scene) | strict | Prevent catastrophic per-scene regression |
| **G4** Policy entropy > 0.5 nats (averaged) | soft | Ensure it's actually doing per-scene gating, not collapsing |
| **G5** Variance of selected heads across scenes > variance threshold | soft | Same — confirm not collapsing to static |
| **G6** (STRETCH, added 2026-06-25 18:30) Match V1 (Δ ≤ −0.001) AND strictly beat V3 on L24 (Δ vs V3 ≥ +0.005) at ≥ 5% KV saving | stretch | Targets the L24 brittleness revealed by rank-variance prior — V3 masks 7 non-bot-K heads, learned policy should fix this |

**Stretch goal**: Mean PDMS ≥ V1 PDMS = 0.8981 with KV saving ≥ 2.0% (i.e. dominate V1 strictly).

A separate **navtest eval** (not navtrain) is required after Phase 2 internal acceptance, to confirm cross-distribution generalization. Compute: ~8 GPU-h.

---

## §7. Compute Budget

| Stage | Resource | Wall (est.) |
|---|---|---|
| §3.1 Extend probe to dump `hidden_states[12]` (R1 feature) | 4× H20 | ~3 h |
| §3.3 Compute pseudo-labels (CPU) | CPU | ~10 min |
| §5.1 Supervised warm-start (P1 + P2) | 1× H20 | ~30 min |
| §5.2 RL fine-tune (best of P1/P2) | 4× H20 | ~10 h |
| §6 navtest eval | 4× H20 | ~8 h |
| **Total to first PASS** | | **~22 GPU-h wall** ≈ 5.5 h on 4 GPU |

Fits in **2 sessions** of 14h windows.

---

## §8. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **R1**: prompt embedding (R1 representation) does not carry enough information to predict head importance | **HIGH** | If P1+P2 supervised loss does not converge below random, fall back to R3 (metadata) or per-layer-only gating (28 actions instead of 448). **Mitigation strengthened 2026-06-25 18:30**: per rank-variance prior, train only on high-entropy layers first (L7, L9, L12, L13, L20, L21) — best SNR. |
| **R2**: differentiable top-K (Gumbel) does not transfer to hard top-K at inference | medium | Train with temperature schedule + add hard-top-K consistency loss in last epoch |
| **R3**: RL variance too high to learn in 50 steps | medium | Increase samples-per-scene from 4 → 8; or stay supervised-only and accept the gap |
| **R4**: log_id-based split leaves test set too small | low | Fallback: scene-id random split with strict log_id-disjoint check |
| **R5**: 8 denylist scenes have downstream effect | low | Already excluded in §3.2 |
| **R6**: PDMS reward noise (~±0.005 per scene observed in M1.b₁) drowns signal | medium | Aggregate reward over k=8 scene mini-batches; long-horizon return |
| **R7**: Phase 2 policy collapses to M1.b₁ static mask | "good news, kind of" | If collapses but still passes G1/G2, write up as "static is near-optimal on navtrain" — informative null |

---

## §9. Milestones

| ID | Deliverable | Gate | ETA from start |
|---|---|---|---|
| **M2.0** | Hidden-state feature dump on 2k subset | tensors saved | +3 h |
| **M2.1** | Supervised warm-start P1+P2 trained, val loss curves logged | val BCE < 0.5 | +4 h |
| **M2.2** | RL fine-tune complete on best supervised init | passes G1-G3 on navtrain val | +14 h |
| **M2.3** | Full navtrain test eval | passes G1-G5 | +22 h |
| **M2.4** | navtest eval (cross-distribution) | passes G1-G3 on navtest | +30 h |
| **M2.5** | Locked: results + ablations in `key_results.md §8` | journal entry | +32 h |

---

## §10. Open Questions for Reviewer

> **RESOLVED 2026-06-26 12:35** — User delegated decision authority to AI session. Decisions below; rationale in §10.x. Any of these can be revisited after v0 numbers come in.

| # | Question | **Decision** | One-line rationale |
|---|---|---|---|
| Q1 | Scene representation | **R1** (L12 hidden, mean-pooled over vision tokens) — **REVISED 2026-06-26 14:05 → R1' (see §10.y)** | L12 is where h13 is reliably bot-1 across all 19,225 navtrain tokens (M1.b₁ §3); same layer the policy is meant to gate → no representation-target mismatch. R2/R3 (L24 / multi-layer concat) are ablation candidates, not v0. |
| Q2 | Sparsity constraint | **C1, k = 4** | Matches V3 mask cardinality (3+1=4 heads): keeps action space directly comparable to existing V0/V3/V4 baselines so per-scene PDMS deltas are interpretable on day 1. Larger k (= 8, 12) defers to ablation. |
| Q3 | Architecture | **P1 (linear) + P2 (1-hidden-layer MLP) trained in parallel; P3 (per-layer factorized) deferred** | P1 = honest baseline / sanity floor; P2 = capacity headroom. Running both in the same v0 sweep costs +30% wall but gives a first capacity / sample-efficiency comparison for free. P3 introduces extra hyperparameters (per-layer K) that can't be tuned with 5 GPU-h. |
| Q4 | Training | **Supervised-only v0 first (pseudo-labels from §3.3)**; RL fine-tune deferred to v1 once supervised v0 clears G1+G3 | (a) Pseudo-labels from per-scene bot-K rank are essentially free; (b) supervised loss is convex in P1, near-convex in P2 → debuggable; (c) RL adds reward shaping + variance / entropy regularizer questions that we can't answer without first knowing how well a static policy can score. |
| Q5 | Acceptance gate G3 | **Keep G3 = "no per-scene PDMS drop > 0.05" as soft target**; promote/demote after v0 actuals | 0.05 ≈ 2× the noise floor of single-token PDMS perturbation we observed in M1.b₁ navtest (typical inter-seed jitter ≈ 0.02–0.03). Strictly enforcing 0.05 might tank v0 entirely on rare hard scenes, so we **log violations but don't gate v0 on it**; v1 RL must respect it. |
| Q6 | Compute budget | **Supervised-only v0 ≈ 5 GPU-h** first; RL v1 ≈ 22 GPU-h conditional on v0 G1+G3 PASS | Smallest credible learning signal. If v0 P1 and P2 both fail to beat the static V4 mask in expectation across navtrain probe → kill the whole per-scene gating thread before investing 22 GPU-h. |

### 10.x Open issues these decisions DO NOT resolve

- **Q4 follow-up**: pseudo-label noise. §3.3 builds labels from rank-variance bot-K which is itself derived from the very heads we're trying to gate → label leakage risk. Mitigation in v0: hold out 10% of navtrain tokens never seen during pseudo-label generation, evaluate generalization there. **Owner: AI; due before v0 launch.**
- **Q5 follow-up**: per-scene metric variance baseline. We've never measured re-run jitter on the same scene with the same mask. v0 sweep will run V4 twice (different seed / shuffle) on the navtrain probe → empirical noise floor, then revisit G3 threshold. **Owner: AI; piggy-back on v0 sweep.**
- **Q6 follow-up**: data scaling. 5 GPU-h budget assumes 1 epoch over the 19,225 navtrain tokens. If supervised loss hasn't plateaued, we'll have to either (a) extend to 2 epochs at +5 GPU-h, or (b) accept early-stopped P2. Decision threshold: if val loss is still dropping > 5%/epoch at end of epoch 1, extend.

### 10.y R1 revision (2026-06-26 14:05) — hidden state not in dump

**Discovered while staging v0**: the 19,225 `.pt` dumps under `exp/m1b2_navtrain_full_alllayers/` only contain `per_layer_vision_attn` of shape `(28, 16, 720)` (per-layer × per-head × vision-token attention weights). They do **not** carry the L12 hidden state that §10 Q1 R1 assumed. Re-dumping hidden state would cost ~10 GPU-h, blowing the v0 5 GPU-h budget (Q6).

**Decision (AI, user-delegated)**: substitute **R1'** for v0 only.

| | Original R1 | **R1' (v0)** |
|---|---|---|
| Feature | L12 hidden state, mean-pooled over vision tokens, ∈ R^d_model | L12 per-head attention mass, ∈ R^16: `per_layer_vision_attn[12].mean(dim=-1)` |
| Source | needs re-dump (10 GPU-h) | already on disk (24.9 GB, 19,225 files) |
| Dimensionality | ~3072 (Llama-2-7B / Qwen-class) | **16** |
| Causal alignment | ✅ same layer the policy gates | ✅ same layer, but feature = aggregate of the gated quantity itself |
| Leakage risk | low | **elevated** (label = bot-K rank of the same `attn[12]`; feature = mean of the same `attn[12]`. P1 with ≥1 trivial weight pattern can recover bot-K from feature.) |

**Mitigation** (mandatory for R1' v0 to be informative):

1. **Triviality probe**: before training, fit a *closed-form* baseline `y = bot-K(rank(x))` from x = R1' alone. If this rule-based baseline already achieves ≥ 95% top-4 accuracy on holdout, then **the v0 setup is degenerate** — P1/P2 will just learn this rank-of-mean function and tell us nothing about generalization. Report this baseline number explicitly; if ≥ 95%, stop and pivot to plan B (re-dump hidden) or plan C (hand-crafted feature, §10 alt).
2. **OOD generalization gate**: holdout = 10% navtrain (G_v0_1) **plus** the entire navtest 11,576-token set as OOD probe (G_v0_3). If P1/P2 holdout top-4 acc is high but navtest top-4 acc is significantly lower (Δ > 0.1 absolute), R1' is overfitting to navtrain attention statistics. Promote R1 (re-dump hidden) for v1.
3. **Honest naming**: the v0 results table will tag R1' explicitly so we don't quietly conflate it with the original R1 in cross-doc references.

**v0 acceptance gates (revised under R1')**:

- G_v0_0 (sanity): closed-form `bot-K(rank(x))` baseline reported. If ≥ 95% holdout top-4 acc → degenerate, ABORT v0.
- G_v0_1: P1 + P2 each clear > 90% top-4 acc on 10% navtrain holdout AND beat the closed-form baseline by ≥ 2 points.
- G_v0_2: P2 - P1 holdout top-4 acc gap reported (capacity question).
- G_v0_3 (OOD): both P1, P2 top-4 acc on full navtest reported. Pass if degradation < 0.1 absolute vs holdout.

If G_v0_0 fails (degenerate) → v0 ABORT, write up plan B / C decision in §10.z.
If G_v0_1 fails (can't even fit) → kill per-scene gating thread (Q6 contingency).
If G_v0_3 fails (overfits) → schedule R1 re-dump for v1, freeze R1' results as ablation.

**Status**: R1' adopted for v0. R1 (hidden re-dump) deferred to v1 conditional on G_v0_3 outcome.

**Addendum 2026-06-26 14:10 — navtest attn dump unavailable for G_v0_3**:
Discovered while staging dataset that `exp/m1b2_navtest_*` does **not** exist — only navtrain (19,225) and the smoke probe100 carry `per_layer_vision_attn` dumps. Dumping navtest 11,576 tokens at the same schema would cost ~3 GPU-h and breach the v0 5 GPU-h ceiling.

**Mitigation for v0**: replace G_v0_3 (navtest OOD) with **G_v0_3'** = inner-holdout double-split. Use a hash-stable 80 / 10 / 10 split of the 19,225 navtrain tokens:
- 80% train (~15,380)
- 10% in-distribution holdout (~1,920) → G_v0_1
- 10% **shifted** holdout, sampled by hash of *full token id* (different bucket from G_v0_1) → G_v0_3': measures generalization across same-distribution-but-unseen tokens. This is **weaker** than navtest OOD but is the only zero-cost option today.

If both G_v0_1 and G_v0_3' pass, v0 is *promoted to v1 candidacy*, but **navtest OOD remains a v1 prerequisite** (will be paid for during v1 RL: navtest attn can be dumped piggy-back on V4 navtest full eval, ~3 GPU-h, already in §4.1 budget).

If G_v0_1 ≫ G_v0_3' (Δ > 0.1 absolute), then even within-navtrain generalization is shaky → strong evidence for R1' triviality / overfitting, and we MUST re-dump hidden for v1 regardless of v0 absolute numbers.

### 10.z R1' is mathematically degenerate → pivot to R1'' (cross-layer transfer)

**Discovered 2026-06-26 14:18 in 50-file smoke**: with R1' = `attn[12].mean(-1) ∈ R^16` as feature and y = `bot-K(rank(attn[12].mean(-1)))`, the closed-form rule `pred = bot-K(rank(x))` achieves **100.0% self-consistency on training data, and identically 100.0% on any holdout**, because **y is a deterministic function of x by construction**. There is no learnable signal. P1/P2 can at best memorize the rank-of-x function; they cannot generalize because there is no generalization gap to traverse. The §10.y G_v0_0 "triviality probe" is not just a risk — it is the definitional outcome.

**Pivot decision (AI, user-delegated 2026-06-26 14:20)**: replace R1' with **R1''**.

| | R1' (rejected) | **R1'' (v0)** |
|---|---|---|
| Feature x | `attn[12].mean(-1) ∈ R^16` | **concat of `attn[L].mean(-1)` for L ∈ {0, 4, 8, 16, 20, 24}, ∈ R^96** |
| Label y | bot-4 of `attn[12].mean(-1)` (same vector!) | bot-4 of `attn[12].mean(-1)` — **never appears in x** |
| Trivial closed-form | 100% (degenerate) | unknown, expected ≪ 100% (real prediction problem) |
| Excluded layers | n/a | L11, L12, L13 (avoid neighbour leakage from same-attn-block residual stream) |
| Layer choice rationale | n/a | uniform 6 samples across the 28-layer stack (early L0 / mid-shallow L4 / mid L8,16 / deep L20,24); covers shallow/mid/deep regimes |

**Reframed v0 research question**: *Is the per-scene bot-K mask at L12 predictable from attention statistics at other layers?*

- **YES** (P1/P2 ≫ const baseline) → attention patterns are cross-layer correlated → in v1 we can use any cheap layer as scene representation, OR we can train a multi-layer joint mask policy
- **NO** (P1/P2 ≈ const baseline) → attention patterns are layer-localized → strong evidence that *correctly* gating L12 requires actually looking at L12's hidden state (or L12 itself, which we exclude here as degenerate). Forces R1 (re-dump hidden, ~10 GPU-h) for v1 with high prior.

**Revised acceptance gates (v0 under R1'')**:

- **B0 (const)**: predict the dataset-wide top-4 most-frequent L12 bot-K head ids (≈ {h13, h14, h6, h2} per current rank_variance.json). Multi-label per-head F1 = baseline.
- **B1 (closed-form)**: `pred = bot-K(rank(x_concat))` — return the 4 head idxs (in R^96 / 6-layer space) with smallest mean. Reported but **not** a meaningful baseline (different action space); listed for completeness.
- **G_v0_0**: B0 const baseline F1 reported. Not a gate — defines the floor.
- **G_v0_1**: P1 holdout per-head F1 > B0 + 0.05. P2 same. Both must clear.
- **G_v0_2**: report P2 − P1 holdout F1 gap (capacity probe).
- **G_v0_3'**: P1 / P2 shifted-holdout per-head F1 within 0.05 of in-distribution holdout (generalization, not OOD-true).
- **OUTCOME → v1 path**:
  - all gates PASS → v1 does R1''-feature RL (cheap), navtest dump piggy-back, layer-set ablation
  - G_v0_1 fails → v1 must re-dump L12 hidden (R1 original); kill cross-layer transfer thread
  - G_v0_3' fails → R1'' overfits navtrain attention statistics → re-evaluate after navtest dump

Compute remains within the 5 GPU-h v0 ceiling: dataset build is CPU-only ~70 s; P1+P2 train on 19k × R^96 fits trivially in 1 GPU × 10 min each (P1 < 100 K params, P2 < 10 K params). Bulk of the budget remains unused — held in reserve for layer-set ablation if §10.z gates pass.

---

## §11. Non-goals (explicit)

- **Not** training AutoVLA itself. Policy operates on frozen AutoVLA.
- **Not** changing M1.a layer choice (L* = 12 stays).
- **Not** chasing absolute SOTA on navtest — we measure Δ vs B0/V1 only.
- **Not** building a serving system. Offline analysis only.
- **Not** redumping per-head attention. Phase 1 output is canonical.

---

## §12. Changelog

| Date | Author | Note |
|---|---|---|
| 2026-06-25 18:18 | AI agent | Initial draft, awaiting reviewer feedback on §10 |
| 2026-06-25 18:30 | AI agent | Patched §3.3 / §3.4 / §6 (added G6 stretch) / §8 R1 from rank-variance prior on 19,225 scenes. Findings: V1 is 100% bot-K validated; V3 L24 mask has 7/11 non-bot-K heads (brittleness hypothesis); 6 high-entropy layers identified for Phase 2 primary target. |
| 2026-06-26 12:35 | AI agent | §10 Q1–Q6 RESOLVED (user-delegated). Decisions: R1 / C1 k=4 / P1+P2 parallel, P3 deferred / supervised-only v0 first / G3 logged-not-gated for v0 / 5 GPU-h v0 budget. §10.x added 3 follow-up issues (pseudo-label leakage, per-scene jitter baseline, data scaling). Phase 2 v0 prototype unblocked pending only the next GPU window. |
| 2026-06-26 14:05 | AI agent | §10 Q1 **R1 → R1' revision**. Discovered hidden state was never dumped (only `per_layer_vision_attn` is on disk). To stay within 5 GPU-h v0 budget, swap R1 → R1' = `attn[12].mean(-1) ∈ R^16`. New §10.y documents elevated leakage risk + 4 acceptance gates G_v0_0..3, including a closed-form-rank baseline as triviality probe. Original R1 (hidden re-dump, ~10 GPU-h) deferred to v1 conditional on G_v0_3. |
| 2026-06-26 14:20 | AI agent | **§10.z R1' → R1'' pivot**. Smoke test exposed that R1' is mathematically degenerate (y is a deterministic function of x → no learnable signal). Pivot to R1'' = concat `attn[L].mean(-1)` for L ∈ {0,4,8,16,20,24}, ∈ R^96, predicting L12 bot-4 (never in x). Reframes v0 as a real cross-layer-transfer hypothesis test. v0 outcome now informs v1 path: cross-layer correlated → cheap representation; layer-localized → must re-dump hidden. Compute still well within 5 GPU-h ceiling. |

---

## Appendix A. References

- `docs/results/key_results.md` §4 (M1.a layer selection), §5 (M1.b₀ per-head), §6 (M1.b₁ V0/V1/V2/V3 sweep), §7 (M1.b₂ Stage 3)
- `docs/journal/2026-06-25_m1b2_stage3_done.md` (Phase 1 production run journal)
- `docs/journal/2026-06-24_m1b2_stage1_2_full_journey.md` (Phase 1 dev journey)
- `docs/specs/m1b_freelunch_spec.md` (variant + acceptance spec for M1.b₁, scoring pipeline reusable for Phase 2 RL reward)
- `docs/specs/m1c_lambda_rank_scorer_spec.md` (sibling track: learned per-token head ranker — overlapping but separate)
