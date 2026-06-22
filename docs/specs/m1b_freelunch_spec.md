# M1.b Level-0 Free-Lunch Head Mask — Spec (2026-06-22)

> **Status**: DRAFT v1. Authored before any code change. Locks the contract that
> Phase D (impl) / Phase E (correctness gate) / Phase F (navtest sweep) must
> satisfy. Any deviation must update this spec first.
>
> **Author tag**: `pre_m1b_phase_d` (commit `3d27cc6`).
>
> **Purpose**: implement and evaluate a static, per-(layer, head) attention
> mask that removes **only confirmed-dead heads** from the per-head landscape
> sweep, with the goal of demonstrating zero PDMS degradation at near-zero
> implementation risk. This is the lowest-risk anchor on the
> compute-vs-quality Pareto curve we will report in the paper.

---

## 1. Scientific question (paper-grade)

> **Q (M1.b₁)**: Given a per-head per-layer vision-attention landscape (§5.5 of
> `docs/results/key_results.md`), does masking confirmed-dead heads
> (mean vision attention < 1e-3, measured on n=100 navtest tokens) at
> inference time preserve PDMS within B0 noise (≤ 1e-3 absolute)?

Three increasingly aggressive variants test the dose–response curve. If V1
preserves PDMS but V3 degrades it, that itself is a useful finding — it
sets the "free-lunch ceiling" we report in the paper Section "free-lunch
upper bound".

## 2. Acceptance criteria

A variant **passes** iff **all** of the following hold on the full navtest set
(N = 11576 tokens after intersection with metric_cache, matching B0 setup
exactly):

1. **No process crash**: dispatcher exits 0 for all shards.
2. **Token coverage**: ≥ 99.9% of tokens produce a valid metric row (matches
   B0 baseline coverage ≥ 99.9%).
3. **PDMS delta**:
   - V1 / V2: `|PDMS_variant - PDMS_B0_recheck| ≤ 1e-3`
   - V3:      `PDMS_variant ≥ PDMS_B0_recheck − 5e-3`  (looser, as more heads removed)
4. **Sub-metric sanity** (per-token avg, on valid rows):
   - All six sub-metrics (no_at_fault_collisions, drivable_area_compliance,
     ego_progress, time_to_collision_within_bound, comfort,
     driving_direction_compliance) within 5e-3 of B0_recheck.

A variant **fails** if any criterion is violated. A failure is **a paper
result**, not a bug — it bounds the free-lunch claim.

## 3. Variant definitions

**Notation**: `L<layer> : { h<head_id>, ... }` means at decoder layer `<layer>`,
zero out the attention probabilities of the listed heads before they are
multiplied by V. Layer indices are 0-based, matching the
`vlm.model.layers[i]` numbering used in `attention_capture.py:204`.

| variant | head_mask_layers (cfg dict) | total heads removed | source justification |
|---|---|---:|---|
| **V1** (minimal) | `{12: [13]}` | 1 | `landscape_summary.json::L12::dead == [13]`. Mean = 1.92e-4 (n=100). |
| **V2** (moderate) | `{12: [13], 27: [0, 8, 9]}` | 4 | V1 ∪ `landscape_summary.json::L27::dead == [0, 8, 9]`. L27 h0=2.5e-4, h8=2.4e-26, h9=2.3e-5. |
| **V3** (aggressive) | `{12: [13], 24: [0,1,2,6,7,8,9,10,12,14,15], 27: [0, 8, 9]}` | 15 | V2 ∪ `landscape_summary.json::L24::dead` (11 heads). |
| **V0** (B0 re-check) | `{}` (or `head_mask_enabled: false`) | 0 | Baseline equivalence anchor; verifies wrapper introduces no drift. |

**Why these variants are non-arbitrary**: each removed head must have
been measured with `mean(vision_attn.sum(dim=-1)) < 1e-3` on n=100 tokens
sampled with the same scene_filter as navtest. **No "near-dead" heads are
included** in V1/V2/V3 — those are kept as a margin of safety. A later
variant series (V4+) can sweep near-dead inclusion if free-lunch is
established.

## 4. Implementation contract (Phase D code)

### 4.1 Files touched (whitelist — anything else is out of scope)

| file | change | rationale |
|---|---|---|
| `code/rldrive/agents/head_mask_patch.py` (NEW) | implement context manager `patch_head_mask(...)` | independent from `attention_capture.py`, so capture & mask can be composed |
| `code/rldrive/agents/autovla_with_attention.py` | extend ctor with `head_mask_layers: Optional[dict]`, wire into `compute_trajectory` | smallest delta to existing wrapper |
| `code/rldrive/configs/agent/autovla_with_attention.yaml` | add `head_mask_layers: null` default | hydra knob |
| `scripts/run_m1b_freelunch_sweep.sh` (NEW) | dispatcher wrapping `run_autovla_navtest_dual_gpu.sh` env-knob pattern | one entry point for V0/V1/V2/V3 |
| `code/rldrive/agents/__init__.py` | re-export | |

**Out of scope**:
- `code/third_party/AutoVLA/**` — frozen, no edit
- `attention_capture.py` — frozen, no edit
- B0 baseline checkpoint or weights — never touched
- Existing dispatchers (`run_autovla_navtest_dual_gpu.sh`) — never edited; called as-is via env knobs

### 4.2 Head-mask mechanic

In Qwen2.5-VL eager attention (transformers 4.49.0, modeling_qwen2_5_vl.py),
`Qwen2_5_VLAttention.forward` returns `(attn_output, attn_weights, past_kv)`.
`attn_output` is computed *inside* forward as
`out = attn_weights @ V` then projected.

We cannot modify `attn_weights` after the multiplication — we must
intercept **before** the matmul. The cleanest hook is to **monkey-patch
the layer's `self_attn.forward`** (same mechanism as `patch_attention_capture`,
proven on M1.a) and inject the mask between softmax and the V matmul.

But re-implementing the full forward body just to inject one line is
brittle (the body changes between transformers versions). A safer
mechanic is to **subclass-style override**: wrap `orig_forward` and post-multiply
the output:

```
attn_output, attn_weights, past_kv = orig_forward(hidden_states, *args, **kwargs, output_attentions=True)
# attn_weights: (bsz, num_heads, q_len, k_len) — post-softmax
# But we need to apply mask BEFORE attn_weights @ V.
```

Since `orig_forward` already returns `attn_output` after the matmul, we
cannot easily fix it post-hoc unless we recompute. **Three implementation
options**, ranked:

**(A) Pre-softmax additive mask via `attention_mask` argument (preferred).**
Qwen attention accepts `attention_mask` as an additive bias to logits
pre-softmax. If we set `mask[b, h, :, :] = -inf` for masked heads,
attn_weights becomes 0 for those heads, and `out_h = 0` — the head's
contribution to the residual stream is exactly zero. **But `attention_mask`
in the HF interface is `(bsz, 1, q_len, k_len)` — shared across heads,
not per-head**. So this doesn't work as-is.

**(B) Post-attention output zeroing (chosen).**
Patch `self_attn.forward` to call `orig_forward`, then zero specific head
slices of the *attention output before o_proj*. But the output we receive
is **after o_proj** which mixes heads. So we need to intercept inside.

The actually-feasible option:

**(C) Monkey-patch `attn_weights @ V` indirectly via a wrapper on
`self_attn` that re-uses `orig_forward` but stores `attn_weights`, then
manually re-does the head-mix outside.** Too brittle.

**Final choice (D) — patch `self_attn.o_proj` instead.**
After `orig_forward`, `attn_output` shape is `(bsz, q_len, hidden_dim)`,
already after `o_proj`. **We can't recover per-head info from this.**

**Final choice (E, CHOSEN) — reshape-aware post-hook before o_proj.**
We patch `self_attn.forward` to:
1. Call `orig_forward` with `output_attentions=True` to get `attn_weights`.
2. Recompute the head-output ourselves using `attn_weights` and the stored
   value states (we must also intercept value states by patching
   `repeat_kv` or by capturing `past_kv`).
3. Zero the masked heads in our recomputed output.
4. Apply `self_attn.o_proj` to get the corrected `attn_output`.

This is moderately complex. **Acceptable risk** because Phase E2 gate
verifies head_mask=off produces bit-identical output.

**Alternative — choice (F, FALLBACK) if (E) is too brittle**:
Use `transformers` hook on `o_proj` input. We can monkey-patch `o_proj` to
zero specific head slices of its input. Since `o_proj` linear takes
input shape `(bsz, q_len, num_heads * head_dim)`, we can directly slice
`x[:, :, h*head_dim : (h+1)*head_dim] = 0` for masked heads `h`. This is
**clean, single-line, transformers-version-agnostic**.

→ **Use choice (F)**. Patch `self_attn.o_proj` via pre-hook.

### 4.3 Implementation pseudo-code (Phase D target)

```python
# code/rldrive/agents/head_mask_patch.py
from contextlib import contextmanager
from typing import Dict, List, Optional
import torch

@contextmanager
def patch_head_mask(vlm, head_mask_layers: Optional[Dict[int, List[int]]] = None):
    """Zero specified attention heads in specified layers, for the entire
    duration of the with-block.

    Mechanic: register a pre-forward hook on each target layer's o_proj.
    The hook zeros the head-slice of the input tensor before it hits the
    linear projection. This is equivalent (under residual + o_proj
    linearity) to those heads producing zero attention output.

    Args:
        vlm: Qwen2_5_VLForConditionalGeneration (the inner HF model).
        head_mask_layers: dict {layer_idx: [head_idx, ...]}. None or {} disables.

    Yields:
        None. After with-block, hooks are removed.

    Guarantees:
        - If head_mask_layers is None or {}, this is a no-op (no hook
          registered) → bit-identical output to upstream.
        - Hooks are removed in finally even if forward raises.
    """
    if not head_mask_layers:
        yield
        return

    cfg = vlm.config
    num_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // num_heads
    handles = []

    def make_hook(heads_to_zero):
        # input is a tuple (x,) where x: (bsz, q_len, hidden_size)
        # heads_to_zero is a tensor of int head ids
        def pre_hook(module, args):
            (x,) = args
            # Validate exactly once per layer
            assert x.dim() == 3, f"o_proj input expected 3D, got {x.shape}"
            assert x.shape[-1] == num_heads * head_dim, \
                f"o_proj input last dim {x.shape[-1]} != {num_heads}*{head_dim}"
            x = x.clone()  # avoid mutating upstream tensor
            for h in heads_to_zero:
                x[:, :, h*head_dim:(h+1)*head_dim] = 0
            return (x,)
        return pre_hook

    try:
        for layer_idx, heads in head_mask_layers.items():
            if not heads:
                continue
            o_proj = vlm.model.layers[layer_idx].self_attn.o_proj
            h = o_proj.register_forward_pre_hook(make_hook(list(heads)))
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()
```

Wire into `AutoVLAWithAttentionAgent.compute_trajectory`:

```python
# pseudo
if self._head_mask_layers:
    with patch_head_mask(self.autovla.vlm, self._head_mask_layers):
        # existing attention_capture path (compatible — different hook targets)
        ...
else:
    # existing code path unchanged
    ...
```

**Composability with capture**: `patch_attention_capture` patches `self_attn.forward`,
`patch_head_mask` patches `self_attn.o_proj`. They are on different modules
→ can be active simultaneously. Verification target in §6.

### 4.4 Equivalence semantics (formal)

For a single layer L with H heads, let `A_h ∈ R^{q × k}` be the post-softmax
attention of head h, `V_h ∈ R^{k × d_h}` be its values, and `W_o ∈ R^{H·d_h × d}`
be the output projection. Standard forward:

```
out_h = A_h @ V_h                     # (q, d_h)
out_concat = concat(out_h, h=0..H-1)   # (q, H·d_h)
attn_out = out_concat @ W_o            # (q, d)
```

Masking head m by `o_proj` pre-hook:

```
out_concat_masked = out_concat
out_concat_masked[:, m*d_h:(m+1)*d_h] = 0
attn_out_masked = out_concat_masked @ W_o
              = (out_concat @ W_o) - (out_h_m_slice @ W_o[m*d_h:(m+1)*d_h])
```

The second term is exactly the contribution of head m to attn_out.
**Therefore zeroing the input slice is mathematically equivalent to zeroing
that head's contribution.** ✓

## 5. Backup / manifest protocol

Every navtest sweep (V0/V1/V2/V3) MUST emit:

1. `results/raw/M1b_freelunch_{variant}_{ts}/`
   - `shard{0..N}.csv` — exact dispatcher per-shard csv (cp from `NAVSIM_EXP_ROOT/`)
   - `merged.csv` — full merged token-level table
   - `aggregate.json` — `{"pdms": ..., "epdms": ..., "sub_metrics": {...}, "n_valid": ..., "n_total": ...}`
   - `manifest.json` — see §5.1
2. `results/ablations/A4_freelunch_headmask.csv` — accumulating ablation row table (one row per variant)

### 5.1 manifest.json schema (frozen)

```json
{
  "spec_doc": "docs/specs/m1b_freelunch_spec.md",
  "spec_version": "v1",
  "variant": "V1|V2|V3|V0",
  "head_mask_layers": {"12": [13]},
  "git_commit": "<sha>",
  "git_tag_pre_run": "m1b_v1_pre_run",
  "ts_start_utc": "2026-06-22T13:00:00Z",
  "ts_end_utc":   "2026-06-22T17:10:00Z",
  "wall_seconds": 15000,
  "gpu_arr": [0],
  "host": "<hostname>",
  "ckpt_sha256": "<sha256 of AutoVLA_PDMS_89.ckpt>",
  "navtest_scene_filter": "navtest_local_filtered",
  "n_tokens_dispatched": 11576,
  "n_valid_rows": 11570,
  "pdms": 0.8983,
  "epdms": ...,
  "sub_metrics": {...},
  "pdms_delta_vs_B0_recheck": 0.0,
  "passes_acceptance": true,
  "failure_reason": null,
  "shard_logs": ["logs/m1b_v1_shard0_<ts>.log"]
}
```

### 5.2 Git protocol

Before each variant run:
```
git status  # MUST be clean (no .py / .yaml / .md unstaged)
git tag m1b_v{variant}_pre_run_$(date +%Y%m%d_%H%M%S)
```
After each variant run completes (success or fail):
```
git add results/raw/M1b_freelunch_${variant}_*  results/ablations/A4_*.csv
git commit -m "m1b/{variant}: navtest sweep result PDMS=...  (PASS|FAIL)"
git tag m1b_v{variant}_post_run_$(date +%Y%m%d_%H%M%S)
```
Git is the source of truth for "what was run with what code at what time".

## 6. Phase E2 — correctness gate (gating Phase F)

Phase F (the actual 16-hour 4-variant sweep) is gated on Phase E2 passing.
Phase E2:

1. Run head_mask=off on **first 5 navtest tokens** (smoke, 1 GPU, ~3 min).
2. Compare per-token `score` (PDMS) from `merged.csv` against B0's existing
   record on the same 5 tokens.
3. **Pass criterion**: max per-token `|score_V0 - score_B0_orig| < 1e-6`
   (true bit-equivalence — head_mask=off is a no-op hook → no math change
   → fp output identical modulo non-determinism).
   - If non-determinism (eager attn on H20 may have tiny variations across
     runs) makes this strict bound fail, the **relaxed pass criterion** is
     `< 1e-4` (1 / 10000 PDMS point on each token).
4. Run **V1 on first 5 tokens** as a non-trivial smoke. Print which heads
   were masked from logs (assertion `[head_mask] L12: zeroed heads [13]`).
5. **Block Phase F until both smokes pass**, write `docs/_internal/m1b_phaseE2_gate.md`
   with concrete numbers.

If gate fails → revert to `pre_m1b_phase_d` tag, root-cause, redesign.

## 7. Phase F — execution plan (4 runs × ~4h, serial 1 GPU)

Schedule (assuming start 2026-06-22 22:00 local):

| order | variant | start | end (est.) | logs | exp_tag |
|---|---|---|---|---|---|
| 1 | V0 (B0 recheck) | 22:00 | 02:00 | `logs/m1b_v0_*.log` | `m1b_v0_recheck` |
| 2 | V1 | 02:00 | 06:00 | `logs/m1b_v1_*.log` | `m1b_v1` |
| 3 | V2 | 06:00 | 10:00 | `logs/m1b_v2_*.log` | `m1b_v2` |
| 4 | V3 | 10:00 | 14:00 | `logs/m1b_v3_*.log` | `m1b_v3` |

Sweep runner is one shell script that loops over variants, runs the
dispatcher with the right env knobs, and produces per-variant
`manifest.json`. Spawned with `setsid + nohup` so it survives IDE
disconnects. **Single source of failure**: if any variant's dispatcher
exits non-zero, the runner logs `FAIL` to `manifest.json::failure_reason`
and **continues** to the next variant (we want all 4 numbers even if some fail).

GPU 0 is dedicated to this run. If navtrain rsync (PIDs 10602-4) still
holding ceph-fuse IO, the GPU job is CPU-bottlenecked on data loading.
**Mitigation**: monitor `nvidia-smi` once per hour from `overnight_watch.sh`;
if GPU util < 20% sustained, flag in `logs/overnight_alerts.log`.

## 8. Failure modes & escalation

| symptom | likely cause | response |
|---|---|---|
| Phase E2 V0 smoke score differs from B0 by > 1e-4 | wrapper introduces non-determinism (e.g., capture hook on different layer perturbs CUDA state) | disable capture for free-lunch runs; only mask, no capture. Smoke again. |
| V1 PDMS < B0 - 1e-3 | head 13 is NOT actually dead — sample artifact | escalate: re-probe L12:h13 on n=500 sample, re-check landscape |
| dispatcher hangs > 4h on one variant | ceph-fuse stall + navtrain rsync competing | kill that variant, log, skip to next |
| GPU OOM (unlikely, B0 fits in 24GB) | hooks accumulating tensors | check `gc.get_referrers` of hook handle |

## 9. Done-when (Phase F → Phase H criteria)

Phase F is done when:
- 4 manifest.json files exist with `passes_acceptance` ∈ {true, false} (not null)
- `results/ablations/A4_freelunch_headmask.csv` has 4 rows
- `results/raw/M1b_freelunch_*/merged.csv` exists for all 4

Phase H (analysis) then produces:
- `results/figs/A4_freelunch.pdf` — bar plot of PDMS vs variant (B0/V1/V2/V3)
- Update `docs/results/key_results.md` §M1.b₁
- Update `results/main_table.csv` baseline row 5 (free-lunch entry)

---

## 10. Changelog

| date | change |
|---|---|
| 2026-06-22 21:05 | initial spec authored before any code touched. Locks contract for Phase D/E/F. |
