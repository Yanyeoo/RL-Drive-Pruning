# M1.c LambdaRank Per-Head Importance Scorer — Spec (2026-06-22)

> **Status**: DRAFT v1. Authored *before* any code change, *before* M1.b sweep
> results are in. Locks the contract that Phase G (data extraction) / Phase H
> (scorer training) / Phase I (mask deployment + eval) must satisfy. Any
> deviation must update this spec first.
>
> **Predecessor**: `m1b_freelunch_spec.md` (Level-0 static mask of confirmed
> dead heads). M1.c is the *learned* successor: it ranks **all** heads by a
> light-weight neural scorer, enabling continuous Pareto sweeps instead of a
> single discrete "free-lunch" point.
>
> **Author tag (planned)**: `pre_m1c_phase_g` once M1.b sweep completes.
>
> **Purpose**: train a 1 M-parameter MLP that maps per-(layer, head) statistics
> to a scalar importance score; then prune the bottom-k heads at inference and
> measure the PDMS vs. compute trade-off across k ∈ {0, 5, 10, ..., 80}. This
> produces the main Pareto curve for the paper's §5.6.

---

## 1. Scientific question (paper-grade)

> **Q (M1.c)**: Can a *learned* per-head importance ranking (LambdaRank-style,
> trained on per-token attention statistics from M1.b) prune more heads than
> the static free-lunch mask **without sacrificing PDMS more than the
> free-lunch baseline does at the same head count**?
>
> **Q′ (ablation)**: Does *learned ranking* beat *magnitude-of-attention
> ranking* (the obvious heuristic)? If not, we report the negative result and
> default to the heuristic — that itself is a useful contribution.

### Why LambdaRank and not regression / classification

| Candidate loss | Issue | Verdict |
|---|---|---|
| MSE to (1 − Δ PDMS) when single head ablated | Requires N×16×28 ≈ 5000 full navtest evals to construct labels. Compute-prohibitive. | ✗ |
| Binary classification dead/alive (M1.b₀ threshold) | Only 12 positive class samples / layer; severe class imbalance. Throws away the *ordering* information that is the entire goal. | ✗ |
| Pairwise **LambdaRank** on attention-magnitude proxy | Uses only the ordering ⇒ tolerates noisy proxy labels. Established LTR theory (Burges 2010). Pairwise loss is differentiable via sigmoid surrogate. Trains in seconds on a single GPU. | **✓** |

LambdaRank uses ΔNDCG-weighted pairwise hinge loss; we use a simplified version
(Δ between adjacent ranks weighted by attention-mass difference) since our list
is short (16 heads / layer).

---

## 2. Acceptance criteria

The M1.c pipeline **passes** iff **all** of the following hold:

1. **Scorer training reproducibility**: fixed seed (`SEED=42`), training loss
   monotone-decreasing, final pairwise accuracy on held-out tokens ≥ 0.75
   (random = 0.50; we want clearly above-chance).
2. **Pareto dominance over uniform random pruning**: at every k ∈ {10, 20, 40,
   60, 80}, learned-rank pruning has PDMS ≥ random-rank pruning + 1.0
   absolute, averaged over 3 random seeds for the random baseline.
3. **Pareto dominance over magnitude heuristic**: at k = 40 (mid-point), learned
   rank beats magnitude rank by ≥ 0.3 PDMS, OR we report the parity finding
   honestly and recommend the simpler heuristic.
4. **At k = (number of confirmed-dead heads, V1 size from M1.b)**: learned rank
   PDMS ≥ M1.b V1 PDMS − 0.1. (Must not regress on the free-lunch baseline.)
5. **Iso-compute headline target**: at avg pruning ratio = 0.5 (k ≈ 224 of the
   total 28×16 = 448 heads), learned-rank PDMS ≥ B0 − 0.5 (matches the paper
   success criterion in `results/README.md`).

If 5 fails but 1–4 pass, we still publish M1.c as the methodology and report
the negative iso-compute result, then design M2 (token-level pruning) to
recover the gap.

---

## 3. Data extraction (Phase G)

### 3.1 Source: M1.b sweep outputs

The M1.b free-lunch sweep (Phase F of `m1b_freelunch_spec.md`) produces, for
each of V0/V1/V2/V3, a directory containing 24 layer-level attention tensors,
one per shard, of shape `[N_tokens, n_heads, n_vision_tokens]` after pooling
over the action-token query dimension.

We use **V0 only** (the unmasked run) for M1.c training data, so the scorer
learns the *natural* head distribution unperturbed by any mask.

### 3.2 Per-(token, layer, head) feature vector

For each `(t, ℓ, h)` triple (t = navtest token id, ℓ ∈ [0, 27], h ∈ [0, 15]):

| Feature | Definition | Dim |
|---|---|---|
| `mean_attn` | mean over vision-token axis of attention weight (action-tok → vision-tok) | 1 |
| `max_attn` | max over vision-token axis | 1 |
| `entropy` | Shannon entropy of vision-attention distribution (normalised) | 1 |
| `top1_pos_x`, `top1_pos_y` | spatial coordinate of arg-max vision token (normalised 0..1) | 2 |
| `layer_onehot` | one-hot of ℓ | 28 |
| `head_onehot` | one-hot of h | 16 |
| `layer_idx_norm` | ℓ / 27 (continuous companion to one-hot) | 1 |

Total feature dim **d = 50**. We deliberately keep it small and interpretable
so the scorer's behaviour is auditable.

### 3.3 Storage

- Path: `data/m1c_features/V0/{shard_id}.pt`
- Format: torch dict `{features: [N, 50] fp16, head_id: [N] int8, layer_id:
  [N] int8, token_id: [N] int32}`
- Total size estimate: 11576 × 28 × 16 × 50 × 2 byte ≈ **520 MB**, fits on the
  workspace volume.

### 3.4 Sanity checks before training

- Feature distribution per layer: mean_attn histogram for layer 12 must match
  the values reported in `docs/_internal/m1b_per_head_analysis_2026-06-18.md`
  §3 (within fp16 rounding).
- No NaN/Inf.
- Layer-0 and layer-27 head 0 feature row printed to log for manual diff.

---

## 4. Label construction (the key methodological choice)

We do **not** have ground-truth per-head PDMS contribution. Instead we
construct a *weak* pairwise label from the same attention statistics, then
train the scorer to *deepen* the signal.

### 4.1 Per-token gold ranking

For each token t and layer ℓ, define the gold ranking of heads h₁, ..., h₁₆ by
descending `mean_attn`. This is the obvious magnitude heuristic.

### 4.2 Why training on this isn't trivial

If we just learned `mean_attn`, we'd reproduce the heuristic exactly. To get
*beyond* it, we add two ingredients:

1. **Layer interaction**: the scorer sees all 28×16 heads together (concatenated
   per token), so it can learn that head 5 in layer 12 is only important when
   head 7 in layer 11 is also active. The single-head magnitude heuristic
   cannot represent this.
2. **Cross-token consistency**: pairwise loss is aggregated over many tokens
   ⇒ the scorer rewards heads whose ordering is *consistent* across tokens,
   downweighting heads whose magnitude is high on a few tokens but irrelevant
   on most.

### 4.3 LambdaRank pairwise loss

For a token t and layer ℓ, draw all C(16, 2) = 120 head pairs (i, j). Let
`r_ij = +1 if rank(i) < rank(j) else -1`. Define

```
L = Σ_{t, ℓ, (i,j)} w_ij · log(1 + exp(-r_ij · (s_i - s_j)))
```

where `s_i = scorer(feature[t, ℓ, i])` and `w_ij = |mean_attn_i - mean_attn_j|`
weighs pairs that disagree on important heads more heavily (this is the
"λ"-weighting that gives LambdaRank its name).

### 4.4 Train/val/test split

| Split | Tokens | Use |
|---|---|---|
| train | 80% (≈ 9260) random tokens | gradient steps |
| val | 10% (≈ 1158) | pairwise-accuracy early stop |
| test | 10% (≈ 1158) | reported pairwise accuracy + Kendall-τ |

Seed for split is `42`. We persist the split indices in
`data/m1c_features/split_indices.json` for full reproducibility.

---

## 5. Scorer architecture

A deliberately tiny MLP (smaller is more publishable for a pruning method):

```
Input:  [d=50]
        ↓ Linear(50, 256) + GELU
        ↓ Linear(256, 256) + GELU
        ↓ Linear(256, 1)
Output: scalar importance s ∈ ℝ
```

Param count: 50×256 + 256 + 256×256 + 256 + 256 + 1 ≈ **78 k params**. With
embedding factors and biases ≈ 1 M is the upper bound; we deliberately stay
two orders of magnitude below the VLM head count budget so the scorer is
"free" at inference.

### 5.1 Training hyper-parameters (locked)

| Hyper-param | Value |
|---|---|
| Optimizer | AdamW (β = 0.9, 0.999), wd = 1e-4 |
| LR | 3e-4, cosine to 1e-5 over 20 epochs |
| Batch | 256 tokens × 28 layers × 120 pairs ≈ 860 k pairs / step |
| Epochs | 20 (≈ 5 min on a single A100) |
| Early stop | val pairwise-acc plateau for 3 epochs |
| Seed | 42 |

### 5.2 Outputs

- `models/m1c_scorer/checkpoint.pt` (state_dict)
- `models/m1c_scorer/config.json` (architecture + hyperparams)
- `models/m1c_scorer/feature_norm.pt` (mean/std for input standardisation)
- `models/m1c_scorer/train_log.jsonl` (per-epoch metrics)

---

## 6. Inference-time deployment (Phase I)

### 6.1 Static mask materialisation

After training, we run the scorer on **every** (ℓ, h) pair using its
*averaged* features across the full navtest set. This gives a single global
score per head: `S[ℓ, h]`.

For a target average pruning ratio ρ ∈ [0, 1], the mask is

```
M[ℓ, h] = 1 if S[ℓ, h] is in top (1−ρ)·n_heads scores within layer ℓ, else 0
```

Per-layer top-k is preferred over global top-k because it preserves at least
some heads in every layer (avoids catastrophic layer-wipe-out failure mode).

### 6.2 Reusing M1.b's masking infrastructure

The mask plugs directly into `code/rldrive/agents/head_mask_patch.py`
(committed in `1e47e01`). Exactly the same `head_mask_layers` dict format
is used. No new inference code is needed.

### 6.3 Pareto sweep

We sweep ρ ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}. For each ρ:

1. Materialise mask `M_ρ` from scorer + ρ.
2. Run full navtest with `head_mask_layers=M_ρ`.
3. Record PDMS, EPDMS, latency.

Total: 9 runs × ~4 h = ~36 h serial. Acceptable for paper-grade ablation.

### 6.4 Baseline comparators (must also be in the curve)

- **Random rank** (3 seeds, averaged): the random pruning lower bound.
- **Magnitude rank**: prune lowest-`mean_attn` heads (the heuristic).
- **M1.b V1/V2/V3 points**: explicit annotated points on the curve.
- **B0** at ρ = 0.

---

## 7. Backup / manifest protocol

Identical to M1.b spec §5, with addition of:

- `models/m1c_scorer/manifest.json`:
  ```
  {
    "spec_version": "m1c_v1",
    "spec_sha": "<sha of this file>",
    "data_source": "V0 sweep tag <m1b_phase_f_complete>",
    "feature_dim": 50,
    "n_params": <int>,
    "train_seed": 42,
    "split_seed": 42,
    "final_val_pairwise_acc": <float>,
    "final_test_pairwise_acc": <float>,
    "final_test_kendall_tau": <float>
  }
  ```
- Git tags: `pre_m1c_phase_g`, `m1c_data_extracted`, `m1c_scorer_trained`,
  `m1c_pareto_complete`.

---

## 8. Failure modes & decision tree

| Failure | Detection | Response |
|---|---|---|
| Pairwise val acc < 0.6 after 20 epochs | train_log.jsonl | Feature engineering iteration: add `var_attn`, `cross_layer_corr`, or `top3_attn_mass`. Stay within spec §3 augmentation budget. |
| Pairwise val acc ≥ 0.75 but Pareto curve no better than magnitude | results table | Honest write-up: "learned scorer matches heuristic; we recommend the heuristic for simplicity." This is a publishable negative result. |
| ρ=0 (full mask, no pruning) regresses PDMS | sanity check | Bug in mask materialisation. Fix before any ρ>0 run. |
| Mask materialisation produces some layers with 0 active heads | mask audit script | Switch to *minimum-K-per-layer* constraint (K=2 floor). |

---

## 9. Compute budget

| Phase | Step | Wall clock |
|---|---|---|
| G | Extract V0 attention to features (reuse M1.b V0 outputs) | 0 (free) |
| G | Featurise + save | ~10 min |
| H | Train scorer | ~5 min × 3 seeds = 15 min |
| I | Pareto sweep (9 ρ values) | ~36 h |
| I | Baselines (random×3, magnitude) | ~16 h |
| **Total** | | **~52 h** of GPU + 25 min of CPU |

Well within the paper's deadline budget if Phase F finishes by 2026-06-24.

---

## 10. Relationship to paper deliverables

| Paper element (`results/README.md`) | M1.c contribution |
|---|---|
| `main_table.csv` row "AutoVLA + Learned Head Prune (ρ=0.5)" | **filled by Phase I** |
| `pareto_curve.pdf` | **owned by M1.c** (this is *the* main figure) |
| Ablation §5.6 "learned vs heuristic ranking" | **filled by Phase I baselines** |
| Ablation §5.7 "per-layer top-k vs global top-k" | quick follow-up after I |
| Ablation §5.8 "scorer interpretability" | feature-importance from scorer weights, free byproduct |

---

## 11. Done-when

M1.c is "shippable" when all of:

- `manifest.json` written with all fields populated.
- All 9 ρ-points + 4 baseline points have a row in `results/main_table.csv`.
- `pareto_curve.pdf` rendered with annotated V1/V2/V3 free-lunch points.
- Iso-compute PDMS @ ρ=0.5 verdict is logged (PASS / FAIL / NEGATIVE).
- Git tag `m1c_pareto_complete` pushed.
- Spec §2 acceptance criteria are individually checked off in
  `docs/_internal/m1c_acceptance_review.md`.

---

## Appendix A. Why not RL?

The natural alternative is to formulate head pruning as an RL problem (state =
remaining heads, action = drop next head, reward = ΔPDMS at end). We reject
this for the paper version of M1.c because:

1. **Sample efficiency**: each "trajectory" requires a full navtest eval (~4 h)
   ⇒ even 100 trajectories = 400 h. Out of budget.
2. **Credit assignment**: the reward is delayed and noisy (PDMS depends on 11k
   scenes), making policy-gradient variance enormous.
3. **Reviewer perception**: an RL formulation invites questions about
   exploration, baseline RL, etc. that distract from the head-pruning story.
   LambdaRank is the *least exotic* method that does the job.

We **do** sketch the RL variant in the paper's "future work" section, framed
as M2/M3, to acknowledge the natural extension.

---

## Appendix B. Why per-token attention is a valid weak label

A head with consistently high attention magnitude is, by definition,
contributing more vision information to the action token's prediction. If we
ablate it, that information must come from somewhere else (other heads or
hallucinated). For a frozen VLM (no re-training), high-magnitude heads are
*necessary* in expectation. The empirical question is whether magnitude alone
or magnitude + context (what the scorer adds) better predicts which heads can
be removed without PDMS loss. M1.c answers exactly that.
