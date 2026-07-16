"""Variant B — True vision token drop (sequence shortening).

Unlike Variant A (attention mask), this physically removes pruned vision tokens
from the sequence AFTER embeddings are computed but BEFORE the LLM decoder runs.
This yields real FLOPs/latency savings proportional to the number of pruned tokens.

Key insight: M-RoPE position_ids are computed from the FULL sequence first (via
get_rope_index), then we drop the pruned token columns from inputs_embeds,
position_ids, and attention_mask simultaneously. The kept tokens retain their
original 3D positions — no re-indexing needed.

Does NOT modify code/third_party/AutoVLA or HuggingFace source.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch


@contextmanager
def patch_vision_token_drop(
    vlm,
    prune_positions: Optional[torch.Tensor],
    verbose: bool = False,
):
    """Physically drop vision tokens from the sequence before LLM decode.

    Hooks into vlm.forward to:
    1. Let get_rope_index compute position_ids from the FULL unmodified sequence
    2. Before calling self.model(...), drop pruned columns from:
       - inputs_embeds (bsz, seq, hidden)
       - position_ids (3, bsz, seq)
       - attention_mask (bsz, seq)
       - cache_position if present

    This shortens the actual sequence the LLM processes → real compute savings.

    No-op if prune_positions is None or empty.
    """
    if prune_positions is None or int(prune_positions.numel()) == 0:
        yield
        return

    prune_positions = prune_positions.flatten().to(torch.long)
    orig_forward = vlm.forward
    state = {"n_calls": 0, "prefill_done": False}

    def patched_forward(*args, **kwargs):
        state["n_calls"] += 1

        # Only modify the PREFILL call (first call with full sequence)
        # Decode steps (seq_len=1) are already on kept tokens only
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]

        inputs_embeds = kwargs.get("inputs_embeds", None)
        attention_mask = kwargs.get("attention_mask", None)

        # Detect prefill: input_ids or inputs_embeds has seq_len > 1
        seq_len = 0
        if input_ids is not None and input_ids.dim() >= 1:
            seq_len = input_ids.shape[-1]
        elif inputs_embeds is not None and inputs_embeds.dim() >= 2:
            seq_len = inputs_embeds.shape[1]

        is_prefill = seq_len > 1 and not state["prefill_done"]

        if not is_prefill:
            # Decode step or subsequent call — pass through unchanged
            return orig_forward(*args, **kwargs)

        state["prefill_done"] = True

        # --- PREFILL: Let the original forward compute embeddings + position_ids ---
        # We need to intercept AFTER inputs_embeds are built (with image tokens
        # scattered in) and position_ids computed, but BEFORE self.model() is called.
        #
        # Strategy: we monkey-patch self.model temporarily to intercept the call
        # and drop tokens from the arguments before they enter the decoder.

        orig_model_forward = vlm.model.forward
        drop_applied = {"done": False}

        def model_forward_with_drop(*m_args, **m_kwargs):
            """Intercept the call to the LM backbone (Qwen2_5_VLModel.forward).
            Drop pruned vision tokens from inputs_embeds, position_ids, attention_mask."""
            if drop_applied["done"]:
                return orig_model_forward(*m_args, **m_kwargs)
            drop_applied["done"] = True

            m_inputs_embeds = m_kwargs.get("inputs_embeds", None)
            m_position_ids = m_kwargs.get("position_ids", None)
            m_attention_mask = m_kwargs.get("attention_mask", None)
            m_cache_position = m_kwargs.get("cache_position", None)

            if m_inputs_embeds is None:
                # Shouldn't happen in Qwen2.5-VL prefill, but safety
                return orig_model_forward(*m_args, **m_kwargs)

            bsz, full_seq, hidden = m_inputs_embeds.shape
            pp = prune_positions.to(m_inputs_embeds.device)

            # Filter: only keep indices within range
            pp = pp[pp < full_seq]
            n_prune = pp.numel()

            if n_prune == 0:
                return orig_model_forward(*m_args, **m_kwargs)

            # Build keep mask
            keep_mask = torch.ones(full_seq, dtype=torch.bool, device=m_inputs_embeds.device)
            keep_mask[pp] = False
            keep_indices = keep_mask.nonzero(as_tuple=True)[0]  # (new_seq,)
            new_seq = keep_indices.numel()

            if verbose:
                print(
                    f"[token_drop] Variant B: dropping {n_prune} tokens, "
                    f"seq {full_seq} -> {new_seq} "
                    f"({(1 - new_seq/full_seq)*100:.1f}% reduction)",
                    flush=True,
                )

            # Drop from inputs_embeds: (bsz, full_seq, hidden) -> (bsz, new_seq, hidden)
            m_kwargs["inputs_embeds"] = m_inputs_embeds[:, keep_indices, :]

            # Drop from position_ids: (3, bsz, full_seq) -> (3, bsz, new_seq)
            if m_position_ids is not None and m_position_ids.numel() > 0:
                if m_position_ids.dim() == 3:
                    # M-RoPE: (3, bsz, seq)
                    m_kwargs["position_ids"] = m_position_ids[:, :, keep_indices]
                elif m_position_ids.dim() == 2:
                    # Standard: (bsz, seq)
                    m_kwargs["position_ids"] = m_position_ids[:, keep_indices]

            # Drop from attention_mask: (bsz, full_seq) -> (bsz, new_seq)
            if m_attention_mask is not None:
                if m_attention_mask.dim() == 2:
                    m_kwargs["attention_mask"] = m_attention_mask[:, keep_indices]
                # If 4D attention mask, leave untouched (rare for prefill)

            # Update cache_position: (full_seq,) -> (new_seq,)
            if m_cache_position is not None:
                m_kwargs["cache_position"] = m_cache_position[keep_indices]

            return orig_model_forward(*m_args, **m_kwargs)

        # Install the interceptor
        vlm.model.forward = model_forward_with_drop
        try:
            result = orig_forward(*args, **kwargs)
        finally:
            vlm.model.forward = orig_model_forward

        return result

    vlm.forward = patched_forward
    try:
        yield state
    finally:
        vlm.forward = orig_forward


__all__ = ["patch_vision_token_drop"]
