# M1B2 Phase-2 V0 — Probe Results & Hypothesis Verdicts (2026-06-26)

**Scope.** Two cheap, fast experiments executed in parallel with the V4 4-shard PDMS sweep
(still running at time of writing). Both experiments were specified in
`docs/_internal/m1b2_phase2_design_2026-06-25.md` §10 (R1' / R1'').

**TL;DR.**
1. **R1'' hypothesis is dead.** P1 (linear 96→16) and P2 (MLP 96→64→16) probing
   `L{0,4,8,16,20,24}` attention → L12 bot-K head selection **cannot beat a
   const-top-K baseline** to 4 decimals. Cross-layer attention has no
   learnable signal pointing to "which L12 heads are bot".
2. **V4's L24-mask {7,9,10} is independently validated by a pure-frequency scan.**
   L24 bot-K frequencies on 19,225 tokens: **h9 = 100.0%, h10 = 98.9%, h7 = 93.7%**.
   V4's choice (made earlier via rank-variance explorer probes) matches the
   single-shot bot-K frequency ranking exactly.
3. **L12 is the most token-dependent of the 7 sampled layers** (EM=0.30 const
   baseline vs 0.85–1.00 on L0/L4/L8/L16/L20), which mechanistically explains
   why R1'' failed: the predictor X side comes from layers whose own attention
   patterns barely vary across tokens, so X has no signal to map to L12's
   genuine variability.

Status of the four parallel V0 tasks:

| ID | Task                            | Status     | Wall    |
|----|---------------------------------|------------|---------|
| A  | dataset_build (R1'')            | DONE       | ~13 min |
| B  | V4 4-shard PDMS sweep (g0+g1)   | RUNNING    | ~1.8 h  |
| C  | P1/P2 R1'' probes               | DONE       | 0.8/1.6 s wall (CPU) |
| D  | bot-K freq scan across 7 layers | DONE       | 37 s    |

---

## 1. R1'' verdict — cross-layer probe (Task C)

**Setup.**
- Dataset: `exp/m1b2_phase2_v0/dataset_R1pp_target12_botK4.pt`
  - X: concat of mean-vision-attention over heads, layers L ∈ {0,4,8,16,20,24} → dim 96
  - y: multi-label 16-D bool, L12 bot-K=4 heads
  - n = 19,225 tokens, split 80/20 holdout, plus a `shifted` distribution split
- Probes:
  - P1 = `Linear(96, 16)` + BCEWithLogits + threshold@best-val
  - P2 = `MLP(96→64→16)` + dropout 0.1 + same loss
  - Optimizer Adam lr=1e-3, wd=1e-4, 30 epochs, batch 512, seed 0, CPU
- Baseline B0: const top-K = `[13, 14, 6, 0]` (most frequent bot heads).

**Results (final / best epoch).**

| Probe                | best_ep | train_loss | holdout F1 | holdout EM | shifted F1 | shifted EM |
|----------------------|--------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| **B0 const top-K**   | —       | —          | **0.2203** | **0.2981** | **0.2206** | **0.3068** |
| **P1 linear**        | 3       | 0.4458 →end | 0.2203     | 0.2981     | 0.2206     | 0.3068     |
| **P2 MLP**           | 1       | 0.1880 →end | 0.2203     | 0.2981     | 0.2206     | 0.3068     |
| closed-form `rank(x_concat)%16` | — | — | 0.0046 | 0.0000 | — | — |

`P1 ≡ P2 ≡ B0` to 4 decimal places on both holdout and shifted splits.

**Reproducibility.** Metrics and model checkpoints persisted at:
- `exp/m1b2_phase2_v0/p1_full_20260626_154930/{metrics.json,model.pt,train.log}`
- `exp/m1b2_phase2_v0/p2_full_20260626_154951/{metrics.json,model.pt,train.log}`

**Interpretation.**
- P2 fits the *training* distribution (train loss 0.19, vs P1 0.45), but its
  generalization metric is identical to the const baseline → it memorized noise
  inside train, found no transferable signal.
- The shifted-split numbers are within 0.0003 of holdout: no ID overfit, no OOD
  collapse, the predictor is just degenerate to a constant in expectation.
- The closed-form `rank(concat) % 16` heuristic (one of the candidate
  hand-coded mappings) is at chance (F1=0.005).
- **Conclusion: R1'' is rejected.** Any sequel that uses the
  `L{0,4,8,16,20,24} → L_target bot-K` framing is unlikely to be worth the
  compute. The signal is not in this X.

---

## 2. L24-mask validation — bot-K freq scan (Task D)

**Setup.**
- One-shot scan of all 19,225 token `.pt` dumps in
  `exp/m1b2_navtrain_full_alllayers/`.
- For each L ∈ {0,4,8,12,16,20,24}: compute per-head mean attention over vision
  tokens, take the K=4 lowest-mean heads ("bot-K"), accumulate per-head
  bot-K frequency across the dataset, and report a const-top-K baseline
  (predicting the 4 globally most-frequent bot heads for every token).
- 37 seconds wall, single CPU process.

**Per-layer table (K=4).**

| Layer | top-K heads (by freq) | top-K freqs                    | const F1 | const EM | mass(topK)/4 |
|------:|-----------------------|--------------------------------|---------:|---------:|--------------|
| L0    | h11 h12 h1  h5        | 1.000 1.000 1.000 0.998        | 0.250    | **0.998** | 3.998 |
| L4    | h4  h1  h15 h0        | 1.000 1.000 0.944 0.930        | 0.246    | 0.874     | 3.874 |
| L8    | h11 h14 h15 h12       | 1.000 1.000 0.999 0.999        | 0.250    | 0.997     | 3.997 |
| **L12** | h13 h14 h6  h0      | 1.000 0.927 0.775 0.525        | 0.220    | **0.300** | 3.228 |
| L16   | h1  h14 h10 h13       | 1.000 0.996 0.948 0.898        | 0.245    | 0.847     | 3.841 |
| L20   | h12 h8  h11 h14       | 1.000 1.000 1.000 1.000        | 0.250    | **1.000** | 4.000 |
| **L24** | **h9 h10 h7** h0    | **1.000 0.989 0.937** 0.535     | 0.229    | 0.501     | 3.461 |

Bold rows: layers where V0/V4 actively mask. Bold cells: V4's L24-mask
candidates `{7, 9, 10}`.

**Reproducibility.** `exp/m1b2_phase2_v0/botK_freq_alllayers.json`;
log at `logs/m1b2_phase2_v0/botK_alllayers_20260626_155453.log`;
script `scripts/_oneoff_botK_freq_alllayers.py`.

**Findings.**

1. **V4 L24-mask {7, 9, 10} is independently validated.** The three heads V4
   chose (originally selected via rank-variance explorer probes in a different
   pipeline) are exactly the top-3 most-frequently-bot heads at L24, with
   per-head bot-K frequencies of 100%/99%/94%. These three heads are bot-K in
   essentially every token — masking them is closest to a no-op among
   plausible mask choices.

2. **L0 / L8 / L16 / L20 are almost entirely token-invariant** (const EM ≥ 0.85,
   four of them ≥ 0.997). The "which heads are bot" decision at those layers
   is a layer-global property, not a token-conditional one.

3. **L12 is the most token-dependent of the sampled layers** (const EM = 0.30,
   topK mass 3.228 / 4). Specifically the *4th* bot head at L12 is genuinely
   variable (h0 only 52.5% of the time). This is consistent with V0's choice
   to only mask **h13** (the single 100%-bot head) at L12 rather than the full
   top-4.

4. **L24's structure mirrors L12 in tail behavior**: top-3 ∈ {h9, h10, h7} are
   pinned (94–100%) but the 4th slot (h0 at 53.5%) is borderline. V4's mask
   stops at the pinned-3, which is the same "only mask the unambiguous
   bots" discipline V0 applied at L12.

**Mechanistic interpretation of R1'' failure (linking Task C ↔ Task D).**
R1'' tried to predict L12's bot-K from concat-attention at
L{0, 4, 8, 16, 20, 24}. But Task D shows five of those six feature layers
have mass ≥ 3.84 / 4, i.e. they output **near-constant** per-head averages
across tokens. L24's first 3 heads are also pinned. So `X` is approximately a
constant vector (plus noise) for most tokens, while `y` (L12 bot-K) is
genuinely variable specifically on its 4th slot. There is no signal in X
about L12's only-actually-variable bit, so no predictor — linear, MLP, or
otherwise — can outperform B0. **The data is degenerate by construction,
not the probe.**

---

## 3. Forward implications

### 3.1 V4 4-shard sweep (Task B, in flight)

- Pre-PDMS evidence supports the mask choice `{12:[13], 24:[7,9,10]}`:
  every masked head is bot-K with frequency ≥ 92.7%.
- The mask is **conservative in shape**: V4 masks 4 heads total, only the
  heads that are bot in essentially every token. So the "head_mask hurts"
  failure mode is mechanistically unlikely; the remaining risk is "masking
  too few heads to matter" — failure mode is PDMS ≈ V0, not collapse.
- Decision: keep the V4 sweep running. No reason to abort.

### 3.2 What R1''-death does NOT imply

- It does NOT rule out that some *other* feature set could predict L12 bot-K.
  Specifically: per-token text embeddings, raw vision token features
  (`per_layer_vision_attn` is post-softmax avg, very lossy), or
  attention-pre-softmax logits. Those experiments are out of V0 scope.
- It does NOT rule out predictability of *L12's only-variable head selection*
  (which 4th head). That is a 16-way classification on the 525/475 split of
  "h0 vs not h0 as 4th bot", and would need targeted features. Not pursued
  here.

### 3.3 What V1 / V2 mask design should learn from this

- **Layers where const_EM = 1.00 (L0/L8/L20)**: masking the top-K bot heads is
  effectively a no-op as a *signal-removal* operation — those heads are
  always bot, so they were already contributing negligible attention. Mask
  there only if the goal is *parameter-prune-style* (and any mask choice is
  equivalent).
- **Layers where const_EM ≈ 0.30 (L12)**: masking by top-K freq picks up
  "head sometimes contributes, sometimes doesn't" heads. V0 keeping only
  h13 (100% bot) is the safe call.
- **L24 (const_EM ≈ 0.50)**: the {7,9,10} mask matches the 100%-pinned heads;
  adding h0 (53.5%) would be the next-most-aggressive step and is the obvious
  V5+ ablation if V4 underperforms V0.

---

## 4. Multi-K consistency scan (extension of Task D)

Same scan repeated with K ∈ {2, 3, 4, 5} in a single 37s IO pass.
Output: `exp/m1b2_phase2_v0/botK_freq_alllayers_multiK.json`.

**L12 vs L24, top-K identities and pinning level:**

| K | L12 top-K (freq)                                | L24 top-K (freq)                                  |
|--:|--------------------------------------------------|----------------------------------------------------|
| 2 | h13 h6 (1.000, **0.466**)                        | h9 h10 (0.991, 0.924)                              |
| 3 | h13 h14 h6 (1.000, 0.808, 0.647)                 | **h9 h10 h7** (0.999, 0.980, **0.847**)            |
| 4 | h13 h14 h6 h0 (1.000, 0.927, 0.775, **0.525**)   | h9 h10 h7 h0 (1.000, 0.989, 0.937, **0.535**)      |
| 5 | + h4 (0.665)                                     | + h6 (**0.444**)                                   |

**Key signal: L24 has a clean "freq cliff" exactly at K=3.**

- K=3 → K=4 jump for the 4th head: **0.847 → 0.535** (L24 4th head h0 only 53%).
  This is a **31-percentage-point drop** between the pinned plateau and the
  first borderline head.
- K=4 → K=5 next head: **0.535 → 0.444** for h6, another gap.
- The pinned-plateau cardinality at L24 is **exactly 3**, matching the V4
  mask `{7, 9, 10}` cardinality. **V4's choice of how many heads to mask is
  also independently validated**, not just which heads.

**L12 has no such plateau.**

- Already at K=2 the 2nd head h6 is only at 0.47. L12's bot-K cardinality is
  effectively 1 (just h13). This supports V0's `L12 = {13}` choice and is a
  warning sign for any V1 that escalates the L12 mask:
  *the second-most-frequent bot at L12 is below coin-flip frequency.*

**L0 / L8 / L20** show pinned plateaus extending to K≥5 (every freq ≥ 0.687,
several at 1.000). These layers are essentially constant w.r.t. bot-K head
identity. They are "safe to mask" in the sense that no token-conditional
signal lives there, but for the same reason **masking them carries no
information-theoretic effect either** — those heads were already
contributing near-zero attention to start with.

### 4.1 Forward implications updated

| Layer | Plateau K* | Mask discipline |
|------:|-----------:|------------------|
| L0    | ≥ 5         | safe, but uninformative — no PDMS effect expected from any choice ≤ 5 |
| L4    | ~4 (4th = 0.93)  | mask {4, 1, 15, 0} safe, K=5 onset is steeper |
| L8    | ~4 (4th = 0.999) | same as L0/L20 — safe, near-uninformative |
| **L12** | **1**      | V0's `{13}` is the right call. Going to K=2 already crosses the borderline; **V1 should avoid expanding L12** unless deliberately experimenting on the borderline head h6 |
| L16   | ~3 (3rd = 0.95, 4th = 0.90) | candidate L16 mask: `{1, 14, 10}` if a V*-variant wants to expand mask layers |
| L20   | ≥ 5         | safe but uninformative |
| **L24** | **3**      | **V4's `{7,9,10}` is exactly the plateau**. A V5-style ablation adding h0 (53%) would land on the same kind of borderline territory L12 sits in |

### 4.2 Candidate V5+ designs derived from this scan

- **V5 (more aggressive at L24)**: `{12:[13], 24:[7, 9, 10, 0]}`. Tests
  whether the borderline 4th head h0 at L24 affects PDMS. *Only run if V4 is
  flat or improving over V0; otherwise ignore.*
- **V6 (broaden uninformative layers)**: `{0:[1,11,12,5], 12:[13],
  24:[7,9,10]}`. Tests whether masking confirmed-pinned heads at L0 has any
  PDMS effect at all; if yes, suggests a downstream layer reads
  these as a signal despite per-layer ~0 attention; if no, confirms the
  "pinned plateau heads are noise" hypothesis and frees future search.
- **V7 (L16 mask)**: `{12:[13], 16:[1,14,10], 24:[7,9,10]}`. L16 also has a
  clean 3-head plateau (0.99/0.99/0.95). Tests whether mid-stack masking
  (beyond just L12, L24) adds.

These are **not** spec'd yet; gating on V4 PDMS numbers.

## 5. Open / parked threads

- **W_MORE_K** ✅ DONE (results above).
- **V4 navtest sweep** 🟡 PARTIAL — shard0+shard2 done at 17:38/17:39 before
  19:00 GPU recycle; shard1+shard3 graceful-killed at 17:42 (would not
  finish in time). See §6 for partial PDMS.

---

## 6. V4 navtest sweep — partial results (shard0 + shard2, n=5912)

V4 spec: `head_mask_layers={12: [13], 24: [7, 9, 10]}` (4 heads, predicted
KV saving ~0.89%, per V4 spec §3).

### 6.1 Per-shard numbers

| shard | n_valid / n | PDMS    | CSV |
|------:|-------------|--------:|-----|
| shard0 | 2948 / 2949 | 0.89466 | `exp/m1b_freelunch_V4_g0_20260626_154324/2026.06.26.15.44.01/2026.06.26.17.38.37.csv` |
| shard2 | 2963 / 2963 | 0.88904 | `exp/m1b_freelunch_V4_g1_20260626_154429/2026.06.26.15.45.08/2026.06.26.17.39.04.csv` |

**Combined (shard0 + shard2, n = 5912): PDMS = 0.89184**

ΔPDMS vs B0 baseline (0.8983) = **−0.0065**.

### 6.2 Sub-score breakdown (combined)

| Sub-score                                | Mean    |
|------------------------------------------|--------:|
| no_at_fault_collisions                   | 0.9915  |
| drivable_area_compliance                 | 0.9564  |
| ego_progress                             | 0.8263  |
| time_to_collision_within_bound           | 0.9733  |
| comfort                                  | 0.9985  |
| driving_direction_compliance             | 0.9772  |
| **score (= PDMS)**                       | **0.8918** |

### 6.3 Interpretation — *partial, with caveats*

**Caveat 1: single-shard variance is large.** PDMS(shard0) − PDMS(shard2)
= 0.0057, which is on the same order as Δ vs B0 (0.0065). With only 2/4
shards, **the −0.0065 estimate has noise floor ~±0.003**. A "final V4 not
free-lunch" verdict requires shard1+shard3.

**Caveat 2: even at the upper end, this is not M1.b₁ V1 territory.** Best
case if shard1+shard3 are uniformly higher than shard0+2, V4 still lands
near Δ ≈ −0.003, which is **6× the V1 floor of −0.0004**. The clean V1
free-lunch result is *very* unlikely to reproduce at V4's wider mask.

**Caveat 3: V4 ≫ V3.** Even partial, V4's Δ = −0.0065 is **~7× better than
V3's Δ = −0.044** (M1.b₁ V3 wide mask result). The multi-K freq-cliff
selection (§4) **did its job at the order-of-magnitude level** even if it
didn't reach V1 free-lunch.

### 6.4 Mechanistic read

L12 mask `{h13}` alone was free-lunch in V1 (M1.b₁). Adding the L24
`{7,9,10}` plateau heads costs ~0.0061 absolute PDMS on shard0+2. That
cost is **larger than the multi-K plateau-vs-borderline gap predicted**
(K=3 plateau at 0.85–1.00, K=4 borderline at 0.535 — the freq-cliff
predicts low risk, but reality says even pinned-plateau heads at L24
carry signal).

Two refinements implied for the V0 results doc:

1. **The freq-cliff predicts whether masking *destroys structure*, not
   whether it *destroys performance*.** L24 pinned heads might attend to
   the same dead-token region every time, *and still* be load-bearing —
   the rank-variance evidence (already cited) didn't rule this out.
2. **L24 is structurally different from L12.** V1 told us L12 has a
   single dispensable head (h13). The V4 partial number tells us L24's
   3-head plateau **is not similarly dispensable**, despite identical
   bot-K frequency profile. → Phase 2 v0's per-scene gating may be the
   *only* path to KV saving beyond V1's 0.39%.

### 6.5 Implications for tomorrow's work

| Question                                  | Answer (preliminary, pending shard1+3) |
|-------------------------------------------|-----|
| Should we run V5 (`{12:[13], 24:[7,9,10,0]}`)? | **No.** V4 already not free-lunch; adding borderline h0 (53% freq) will worsen. |
| Should we run V7 (`{12:[13], 16:[1,14,10], 24:[7,9,10]}`)? | **Optional / low priority.** L16 plateau is an independent direction; only worth it if Phase 2 v0 stalls. |
| Should we run V6 (broaden L0/L20 — pinned uninformative)? | **Yes, cheap diagnostic.** Confirms whether "pinned heads = noise" or "pinned heads can still hurt". 1 GPU-h. |
| Is Phase 2 v0 still the right main bet?   | **Yes, more so.** Static masks at L24 are *not* the route to better KV saving; per-scene gating becomes the only mechanism with theoretical room above V1. |

### 6.6 Re-run instructions for shard1+shard3

To complete the 4-shard V4 sweep on Monday:

```bash
# Lane A (GPU 0): only shard1
bash scripts/_oneoff_v4_4shard_2gpu.sh 0 shard1 shard1
# Lane B (GPU 1): only shard3
bash scripts/_oneoff_v4_4shard_2gpu.sh 1 shard3 shard3
```

(The wrapper will execute the for-loop twice on the same shard; we accept
the double-run and just take the first CSV. Cleaner: edit the wrapper to
accept a single shard. Either is ~1 min of editing.)

Expected wall: ~55 min per shard, 2 lanes parallel → ~55 min total.
Expected disk: 2× ~215 KB CSV.

---

## 7. Files & paths

- Design / spec: `docs/_internal/m1b2_phase2_design_2026-06-25.md`,
                 `docs/_internal/m1b2_v4_spec_2026-06-25.md`
- Probe scripts: `scripts/_drafts/m1b2_phase2_v0_train_p1.py`,
                 `scripts/_drafts/m1b2_phase2_v0_train_p2.py`,
                 `scripts/_oneoff_botK_freq_alllayers.py`,
                 `scripts/_oneoff_botK_freq_alllayers_multiK.py`
- Probe outputs:
  - `exp/m1b2_phase2_v0/p1_full_20260626_154930/`
  - `exp/m1b2_phase2_v0/p2_full_20260626_154951/`
  - `exp/m1b2_phase2_v0/botK_freq_alllayers.json`
  - `exp/m1b2_phase2_v0/botK_freq_alllayers_multiK.json`
- V4 sweep outputs: `exp/m1b_freelunch_V4_g{0,1}_20260626_*` (PDMS pending)
