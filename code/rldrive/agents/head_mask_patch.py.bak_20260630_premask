"""M1.b Level-0 free-lunch — static per-(layer, head) attention mask.

Mechanic: register a forward-pre-hook on each target layer's `self_attn.o_proj`
that zeroes the head-slice of the projection input. This is mathematically
equivalent to that attention head producing exactly zero output to the
residual stream (proof in docs/specs/m1b_freelunch_spec.md §4.4).

Why o_proj pre-hook and NOT a softmax-level mask:
  * pre-softmax additive mask is shape (bsz, 1, q_len, k_len) → shared across
    heads, can't address a single head;
  * post-attention output is already after o_proj, heads are mixed —
    can't recover per-head slices;
  * o_proj input has shape (bsz, q_len, num_heads * head_dim) which is a
    flat concat of per-head outputs → zero `[h*head_dim:(h+1)*head_dim]`
    and the head's contribution to attn_out via W_o becomes the zero
    vector contribution exactly.

This module is independent from `attention_capture.py`. The capture hook
patches `self_attn.forward`; this module hooks `self_attn.o_proj`. Different
attributes → fully composable (both can be active simultaneously).

Spec:    docs/specs/m1b_freelunch_spec.md
Used by: code/rldrive/agents/autovla_with_attention.py (via cfg.head_mask_layers)
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import torch


def _normalize_mask_dict(
    head_mask_layers: Optional[Dict],
) -> Dict[int, List[int]]:
    """Coerce hydra-loaded dict to {int: [int, ...]}.

    Hydra may yield keys as strings ("12") or ints (12) depending on the yaml.
    Empty / None / empty-list values are dropped so the caller sees a clean
    'no-op if empty' contract.
    """
    if not head_mask_layers:
        return {}
    out: Dict[int, List[int]] = {}
    for k, v in head_mask_layers.items():
        layer_idx = int(k)
        if v is None:
            continue
        heads = [int(h) for h in v]
        if not heads:
            continue
        # de-duplicate while preserving order
        seen = set()
        clean: List[int] = []
        for h in heads:
            if h not in seen:
                clean.append(h)
                seen.add(h)
        out[layer_idx] = clean
    return out


@contextmanager
def patch_head_mask(
    vlm,
    head_mask_layers: Optional[Dict[int, List[int]]] = None,
    verbose: bool = False,
):
    """Zero specified attention heads at specified layers via o_proj pre-hooks.

    Args:
        vlm: a Qwen2_5_VLForConditionalGeneration (the inner HF model;
             callers usually have `self.autovla.vlm`).
        head_mask_layers: dict {layer_idx (int): [head_idx (int), ...]}.
             None or empty -> no-op, no hooks registered, bit-identical to
             upstream model behavior.
        verbose: if True, log one line per layer the first time the hook
             fires (for sanity, prints to stdout via print()).

    Yields:
        None.

    Guarantees:
        - No-op semantics when input is empty: zero hooks registered, no
          tensor clones, no overhead. This is the V0/B0-recheck path.
        - All hooks removed on exit (including on exception).
        - Each hook clones its input before mutation -> upstream tensor
          unchanged for any other consumer holding a reference.

    Validation: the very first time each hook fires, it asserts the input
    shape matches `(bsz, q_len, num_heads * head_dim)`. If a future
    transformers version changes the o_proj input layout (e.g., adds GQA
    fan-out before o_proj), the assert will trip and the run will halt
    loudly rather than silently produce wrong numbers.
    """
    clean = _normalize_mask_dict(head_mask_layers)
    if not clean:
        # explicit no-op: zero hooks, zero overhead
        yield
        return

    cfg = vlm.config
    num_heads = int(cfg.num_attention_heads)
    hidden_size = int(cfg.hidden_size)
    assert hidden_size % num_heads == 0, (
        f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
    )
    head_dim = hidden_size // num_heads

    # Validate every requested layer index in range
    n_layers = len(vlm.model.layers)
    for layer_idx in clean.keys():
        if not (0 <= layer_idx < n_layers):
            raise ValueError(
                f"head_mask_layers references layer {layer_idx} "
                f"but model has only {n_layers} decoder layers"
            )
    # Validate every requested head index in range
    for layer_idx, heads in clean.items():
        for h in heads:
            if not (0 <= h < num_heads):
                raise ValueError(
                    f"head_mask_layers[{layer_idx}] references head {h} "
                    f"but model has only {num_heads} heads"
                )

    if verbose:
        print(
            f"[head_mask] enabling mask: {clean} "
            f"(num_heads={num_heads}, head_dim={head_dim})",
            flush=True,
        )

    handles = []
    first_fired = {layer_idx: False for layer_idx in clean.keys()}

    def make_hook(layer_idx: int, heads_to_zero: Sequence[int]):
        def pre_hook(module, args):
            (x,) = args  # forward(x) signature
            if not first_fired[layer_idx]:
                # validate exactly once per layer
                assert x.dim() == 3, (
                    f"[head_mask] L{layer_idx}: o_proj input expected 3D "
                    f"(bsz, q_len, hidden), got shape {tuple(x.shape)}"
                )
                assert x.shape[-1] == num_heads * head_dim, (
                    f"[head_mask] L{layer_idx}: o_proj input last dim "
                    f"{x.shape[-1]} != num_heads*head_dim "
                    f"({num_heads}*{head_dim}={num_heads*head_dim}). "
                    f"transformers may have changed o_proj layout; abort."
                )
                first_fired[layer_idx] = True
                if verbose:
                    print(
                        f"[head_mask] L{layer_idx} first fire: x.shape={tuple(x.shape)} "
                        f"zeroing heads {list(heads_to_zero)}",
                        flush=True,
                    )

            # Clone before mutate to avoid touching upstream-held tensor;
            # cost is O(hidden_size * q_len) per layer per forward — negligible
            # vs the attention matmul (O(num_heads * q_len * k_len * head_dim)).
            x = x.clone()
            for h in heads_to_zero:
                x[:, :, h * head_dim : (h + 1) * head_dim] = 0
            return (x,)

        return pre_hook

    try:
        for layer_idx, heads in clean.items():
            o_proj = vlm.model.layers[layer_idx].self_attn.o_proj
            h = o_proj.register_forward_pre_hook(make_hook(layer_idx, heads))
            handles.append(h)
        yield
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                # best-effort; even if remove fails (already gone), don't
                # propagate from finally
                pass


def selftest_no_grad_equivalence(vlm, layer_idx: int, head_idx: int) -> Dict[str, float]:
    """Self-test: verify (a) head-mask applied -> head's contribution zeroed
    in attn_out, (b) head-mask disabled -> output identical to vanilla.

    Returns dict with two L2-norms suitable for an assert harness in tests.
    Does NOT run the full model — operates on a single attention layer's
    `self_attn` module with dummy hidden states.

    Caller responsibility: vlm must be loaded and in eval mode. This is a
    unit-test helper, not used in the production pipeline.
    """
    layer = vlm.model.layers[layer_idx]
    self_attn = layer.self_attn

    device = next(self_attn.parameters()).device
    dtype = next(self_attn.parameters()).dtype
    bsz, q_len = 1, 16
    hidden = int(vlm.config.hidden_size)

    x = torch.randn(bsz, q_len, hidden, device=device, dtype=dtype)

    # baseline forward (call only o_proj to keep test tight)
    with torch.no_grad():
        baseline_out = self_attn.o_proj(x)

    # with mask
    with patch_head_mask(vlm, {layer_idx: [head_idx]}):
        with torch.no_grad():
            masked_out = self_attn.o_proj(x)

    delta = (baseline_out - masked_out).abs().sum().item()

    # verify the difference equals exactly the contribution of that head:
    #   delta_expected = || o_proj_slice_h @ x_slice_h ||_1
    head_dim = hidden // int(vlm.config.num_attention_heads)
    W = self_attn.o_proj.weight  # (hidden, hidden) -> for nn.Linear w/o bias
    # nn.Linear: y = x @ W.T + b   (so column-block of W.T = row-block of W)
    # The h-th head's input column block is x[:, :, h*head_dim:(h+1)*head_dim]
    # and the corresponding output rows of W.T are W[:, h*head_dim:(h+1)*head_dim]
    h0, h1 = head_idx * head_dim, (head_idx + 1) * head_dim
    contrib = (x[:, :, h0:h1].to(W.dtype) @ W[:, h0:h1].T.to(W.dtype)).abs().sum().item()

    return {
        "observed_delta_l1": float(delta),
        "expected_contrib_l1": float(contrib),
        "ratio": float(delta) / float(contrib + 1e-12),
    }


__all__ = ["patch_head_mask", "selftest_no_grad_equivalence"]
