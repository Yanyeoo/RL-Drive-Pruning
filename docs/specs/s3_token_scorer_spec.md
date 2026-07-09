# S3 Slice-1 Spec — Per-Token Importance Scorer (SFT, LambdaRank)

> **Status**: DRAFT v1 (2026-07-06). Authored before code, per journal rule #4.
> **Scope**: This is the **bounded** first slice of S3 = the token Importance
> Scorer SFT (design `Q4.1` / `implementation_plan.md` M1c, *token-level* variant).
> It **does NOT** include Budget Policy (Q4.2/M4) or online GRPO (Q4.3/M2/M5) —
> those are the multi-day part of S3 that needs explicit user scope alignment
> (HANDOFF_2026-07-04 §2). Here we only train an offline scorer and evaluate it
> as a *drop-in selector* against the live `attn_L12` selector.
>
> **Predecessor**: `dynamic_headroom_gate_S2_spec.md` (gate=PASS on full shard0).
> **Executor reused**: `AutoVLAWithTokenPruneAgent` (S1, GPU-verified lossless).

---

## 1. Scientific question

> **Q**: Can a light (~0.1–1M param) MLP that maps per-token ViT→LLM-interface
> features to an importance score, trained by LambdaRank to reproduce the L12
> last-instr→vision attention ranking, **match** the live `attn_L12` selector as
> a token-pruning selector on navtest — thereby removing the need to compute
> attention at inference (a distilled, cheap scorer)?

### Honest ceiling (stated up-front to avoid over-claiming)
The label is L12 attention itself. A distillation scorer's quality ceiling is
its teacher: **≈ attn_L12 (PDMS 89.0 @ r=0.5)**, NOT the per-scene oracle (91.8).
Closing the attn→oracle gap needs real-PDMS RL (M2, online GRPO), deferred.
- **PASS (expected)**: scorer PDMS @ r=0.5 within **±0.5 pt** of attn_L12 across
  r∈{0.25,0.5,0.75}. → SFT scorer is a valid cheap replacement; unlocks M2.
- **BONUS**: scorer > attn_L12 (denoising via cross-token/context interaction).
- **NEGATIVE (still publishable)**: scorer ≪ attn_L12 → features insufficient;
  report and iterate feature set.

---

## 2. Data

### 2.1 Labels (reuse existing, no GPU)
Source: `exp/m1b2_navtrain_full_alllayers/<token>.pt`, key `per_layer_vision_attn`
shape `(28,16,720)`. **Per-token importance label** = `[12].mean(0)` → `(720,)`
(L12, head-averaged). Aligned to that file's `vision_token_positions (720,)`.
19226 navtrain tokens available.

### 2.2 Features (new navtrain GPU dump)
For each navtrain scene, capture the **LLM-input hidden state at vision
positions** = the ViT features after the visual projector, entering decoder
layer 0. Shape `(720, H)` (H = Qwen2.5-VL-3B hidden = 2048), aligned to the SAME
`vision_token_positions`.
- Hook: `patch_vision_feature_capture(vlm, layer_idx=0)` — one-shot on prefill,
  `index_select` at `vision_token_positions`. Same pattern as
  `patch_attention_capture`. Additive to `AutoVLAWithAttentionAgent`
  (`feature_capture_enabled`, `feature_save_dir`, `feature_layer_idx=0`);
  disabled ⇒ zero hooks, bit-identical to upstream.
- **cam_id** (0/1/2) derived from `vision_blocks` = 3 camera spans.
- Alignment guard: feature dump and label dump are separate deterministic runs
  on the same scenes → assert per-token `vision_token_positions` are equal before
  pairing. Tokens that mismatch/missing are dropped.

### 2.3 Scene context (design Q3.c) — deviation noted
`scene_ctx` (nav_cmd/ego_speed/driving_instr) is **broadcast constant across the
720-token list**, so it cannot change a *within-scene* pairwise ranking except
via nonlinear interaction. Slice-1 uses **[token_embedding(2048) + cam_id]** only
(the discriminative, within-list features). scene_ctx is retained in the design
for the Budget Policy (Stage B) and may be added later as an interaction feature.
Reverse: add `scene_ctx` concat to the scorer input and retrain.

### 2.4 Split
navtrain tokens → 80/10/10 train/val/test by token, `seed=42`, persisted to
`data/s3_scorer/split.json`.

---

## 3. Scorer
```
Input:  x = [token_emb (H=2048) ; cam_id_onehot (3)]  -> d ≈ 2051
        LayerNorm -> Linear(d,256)+GELU -> Linear(256,256)+GELU -> Linear(256,1)
Output: scalar importance s
```
~0.6M params. Input standardized by train-set mean/std on the emb block.

### 3.1 Loss — LambdaRank pairwise (within-scene)
For each scene, over the 720 tokens, sample K pairs (i,j) by label; gold order by
descending L12-attention label. `L = Σ w_ij · softplus(-(s_i - s_j)·sign(lbl_i-lbl_j))`,
`w_ij = |lbl_i - lbl_j|`. Aggregate over scenes.

### 3.2 Train (locked)
AdamW lr=3e-4 cosine→1e-5, wd=1e-4, ≤20 epochs, early-stop on val pairwise-acc
(3-epoch plateau), seed=42. Metrics: pairwise-acc, Kendall-τ, NDCG@360 (=top-r0.5).
Outputs: `ckpt/s3_token_scorer/{checkpoint.pt,config.json,feature_norm.pt,train_log.jsonl,manifest.json}`.

Acceptance (offline): val pairwise-acc ≥ 0.75; NDCG@360 vs raw-attention ranking
reported (a scorer that perfectly distills → NDCG≈1 vs its own label).

---

## 4. Inference deployment (selector='scorer')
Add to `AutoVLAWithTokenPruneAgent`: `selector='scorer'` + `scorer_ckpt=...`.
In `_score_for`: pass-1 captures layer-0 vision features (via
`patch_vision_feature_capture`) instead of attention → build [emb;cam_id] →
standardize → MLP → `(720,)` scores. Then existing `select_prune_positions` +
pass-2 pruned generate (unchanged). `keep_ratio=1.0` still lossless no-op.

## 5. Eval (the comparison, GPU) — apples-to-apples with S2
Run `selector=scorer` at r∈{0.25,0.5,0.75} on the **same navtest shard0 subset**
used by S2, so results compare directly to existing:
- baseline `attn_L12`: S2 `results/raw/tokenprune_S2/S2sub*_attnL12_r0{25,50,75}.csv`;
- oracle ceiling & r* : `scripts/oracle_s2.py`.
Report Pareto (scorer vs attn_L12 vs random) + gap-to-oracle. Verdict per §1.

## 6. Compute
Feature dump: navtrain subset (≥3000 scenes) × 4×H20, forward-only ~5s/scene ⇒
~1h for 3000. Train: minutes. Eval: 3 ratios × subset × 2-pass on 4×H20 ~1.5h.
Fits one evening.

## 7. Deliverables
- `code/rldrive/scoring/run_feature_dump.py`, features under `data/s3_scorer/features/`.
- `scripts/s3_build_labels_train_scorer.py` (label pairing + train).
- scorer selector integrated; eval CSVs `results/raw/tokenprune_S3/`.
- `docs/results/key_results.md` new §10 + journal verdict.
