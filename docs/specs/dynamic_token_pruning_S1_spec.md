# S1 Spec — Vision Token Pruning Executor (ViT→LLM interface)

> **Status**: DRAFT v1 (2026-07-03) — for user review **before any code change**.
> **Owner doc**: `docs/plan/design_decisions.md` Revision 2026-07-03 (dynamic pruning revived).
> **Goal of S1**: build the ONE missing execution primitive — drop/mask the lowest-importance vision tokens at the ViT→LLM interface — with an r=1.0 lossless guarantee, WITHOUT modifying `code/third_party/AutoVLA` (subclass + hooks only, per project convention).
> **Non-goal of S1**: no training, no scorer, no budget policy (those are S3). S1 only makes "keep top-B vision tokens by a given per-token score" *executable*.

---

## 0. Context / facts (verified 2026-07-03)

- AutoVLA = Qwen2.5-VL-3B. Prompt built by `autovla.get_prompt(features)`; inference by `autovla.predict(features)` → `vlm.generate(...)`.
- Vision tokens enter the LLM as **placeholder tokens** (`image_token_id`/`video_token_id`) in `input_ids`, later replaced by vision-encoder embeddings at those positions. `PromptIndex.vision_token_positions` (in `rldrive/scoring/attention_capture.py`) already locates them: **N_vision ≈ 720** for navtest.
- Existing hook patterns we reuse:
  - `patch_attention_capture(vlm, layer_idx, prompt_index, bucket)` — patches `self_attn.forward`, grabs L\* attention row (query=last_instr, key=vision) in **one pre-fill forward**.
  - `patch_head_mask(vlm, {L:[h]})` — o_proj pre-hook, zeroes head slices; the reference for "clone-before-mutate + restore-on-exit + shape-assert" hook hygiene.
- Model runs `attn_implementation='eager'` (enforced by `AutoVLAWithAttentionAgent.__init__`).
- **Qwen2.5-VL uses M-RoPE** (3D positions for vision tokens). This is the crux for "truly dropping" tokens (see §3).

---

## 1. Two pruning variants (build order: A first)

### Variant A — **Attention-mask pruning** (S2-grade, low risk, NO position surgery)
Keep all token positions in the sequence, but make the pruned vision tokens **unattendable**: add `-inf` to the attention scores at the pruned vision key positions, for all query positions, in **every** decoder layer.

- **Effect on quality**: identical to removing those tokens from the LLM's view (they contribute nothing to any attention output). → faithful proxy for "external pruning" quality.
- **Effect on FLOPs**: none (positions still occupy the sequence). → **NOT** for the efficiency claim; only for the headroom question (S2).
- **Risk**: minimal. No `input_ids`/`inputs_embeds`/`position_ids` surgery, so M-RoPE untouched.
- **Mechanism**: a `logits_processor`-style additive bias is awkward inside attention; simplest is a **forward-pre-hook on each `layer.self_attn`** that injects/extends the `attention_mask` kwarg with a `(1,1,q_len,k_len)` additive mask having `-inf` columns at pruned vision positions. Under eager attention the additive mask is applied pre-softmax → clean. Register once per generate() (mask columns are fixed across decode steps; the KV of pruned tokens stays but is never attended).
  - Simpler alternative to evaluate: since `patch_head_mask` already proves the o_proj-hook pattern, we can instead **zero the pruned vision tokens' hidden states at the LLM input embedding stage** (a forward-pre-hook on `vlm.model` that zeroes `inputs_embeds[:, pruned_pos, :]`). Zeroing the embedding ≠ blocking attention (a zero vector still gets attended to and contributes value), so **prefer the additive-attention-mask approach** for faithfulness. Decide in impl after a 1-scene A/B vs true-drop.

### Variant B — **True token drop** (S3-grade, FLOPs-saving, higher risk)
Physically remove pruned vision token positions from `input_ids` / `inputs_embeds` / `position_ids` / `attention_mask` before the LLM prefill, so prefill runs on `B` vision tokens instead of `N`.

- **Effect**: real ~ (N−B)/total_seq prefill FLOPs saving (the paper's efficiency claim).
- **Risk (main S-track risk)**: **M-RoPE position ids**. Qwen2.5-VL computes 3D rope positions from the image/video grid; dropping a subset of vision placeholders requires recomputing `position_ids` (and the `image_grid_thw`-derived rope) consistently, or the remaining tokens get wrong positional encodings. Must verify the model's `get_rope_index` path tolerates a pruned vision block, or recompute positions for the kept subset.
- **Deferred to S3**: only needed once S2 gate confirms headroom. S2 uses Variant A.

---

## 2. Public API (proposed, all under `code/rldrive/`)

```python
# rldrive/agents/token_prune_patch.py  (NEW)
@contextmanager
def patch_vision_token_attn_mask(vlm, prune_positions: LongTensor):
    """Variant A: block attention to `prune_positions` (absolute seq indices)
    in every decoder layer for the whole generate() call. Restore on exit.
    No-op if prune_positions is empty (bit-identical to upstream)."""

# rldrive/agents/autovla_with_token_prune.py  (NEW)
class AutoVLAWithTokenPruneAgent(AutoVLAWithAttentionAgent):
    """2-pass per-scene token pruning agent (EVAL path).
      pass1: capture L*=12 vision-attn (reuse patch_attention_capture) -> score s∈R^N
      select: keep top-B by s; prune_positions = vision_token_positions[bottom (N-B)]
      pass2: generate() under patch_vision_token_attn_mask(prune_positions)
    Knobs: keep_ratio r∈{0.25,0.5,0.75,1.0}, selector='attn_L12'|'random'|'external',
           score_layer=12, prune_variant='attn_mask'(A)|'drop'(B, S3)."""
```

- **Selector for S2 is `attn_L12`** (deterministic, from the model itself, no training). `random` and `external`(precomputed) are baselines/ablations.
- Budget `B = round(r · N_vision)`. Ties broken by lowest position index (deterministic).

---

## 3. r=1.0 lossless acceptance (S1 DONE criterion)

1. **No-op path**: `keep_ratio=1.0` (or empty prune set) registers zero effective mask → generated trajectory **bit-identical** to upstream `AutoVLAAgent` on ≥5 navtest scenes (compare poses tensor exact-equal, and PDMS per-scene equal).
2. **Full-run recheck**: `keep_ratio=1.0` on one navtest shard reproduces B0 shard PDMS within ±1e-6.
3. **Mask correctness unit test** (no full model): analogous to `selftest_no_grad_equivalence` — construct a tiny attention, verify masked key columns get exactly zero post-softmax weight.
4. **Variant A vs B faithfulness (1 scene)**: for a fixed prune set, Variant A (attn-mask) and Variant B (true-drop) must yield the **same generated trajectory** (within decode determinism) → proves A is a faithful quality proxy before we rely on it in S2. (If they differ, investigate before trusting S2.)

---

## 4. Risks & mitigations

| risk | sev | mitigation |
|---|---|---|
| M-RoPE breaks on true-drop (Variant B) | 🟡 | S2 uses Variant A (no position surgery); B deferred to S3 with explicit `get_rope_index` recompute + lossless test |
| attn-mask not faithful (zeroed-embed vs true-drop mismatch) | 🟡 | §3.4 A/B faithfulness gate on 1 scene before S2 |
| eager-attention additive-mask kwarg shape/route changed across layers | 🟢 | shape-assert on first fire (mirror `patch_head_mask` hygiene); fail loud |
| 2-pass doubles prefill cost | 🟢 | acceptable for S2 eval (headroom, not speed); S3 can fuse to 1-pass with a cheap scorer |
| KV of pruned tokens still allocated (Variant A) | 🟢 | expected; A is quality-proxy only, not the efficiency number |

---

## 5. Deliverables of S1 (code, gated on user OK)
- `rldrive/agents/token_prune_patch.py` (Variant A; Variant B stub marked S3)
- `rldrive/agents/autovla_with_token_prune.py` (2-pass agent)
- `tests/test_token_prune_lossless.py` (§3.1–3.3)
- No change to `code/third_party/AutoVLA`.

**Est effort**: ~1 day code + 0.5 day lossless verification (CPU/1-GPU smoke). No large GPU run in S1.