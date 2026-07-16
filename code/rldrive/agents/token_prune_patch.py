"""S1 — Vision token pruning executor (Variant A: attention-mask pruning).

Spec: docs/specs/dynamic_token_pruning_S1_spec.md

Mechanism (Variant A, low-risk, faithful quality proxy):
  Mark the pruned vision-token positions as PADDING in the 2D attention_mask
  passed to the VLM. The model's own causal-mask machinery then turns those
  columns into -inf pre-softmax at EVERY layer and EVERY decode step, so the
  pruned vision tokens become completely unattendable — mathematically
  equivalent (for quality) to removing them from the LLM's view.

Why 2D-mask (padding) instead of dropping tokens or zeroing embeddings:
  * dropping tokens requires recomputing Qwen2.5-VL M-RoPE 3D positions
    (that is Variant B, deferred to S3);
  * zeroing an embedding is NOT faithful — a zero vector still receives a
    (nonzero) attention weight and contributes its value;
  * flipping the padding mask (1 -> 0) routes through the model's standard
    `_update_causal_mask`, robust across transformers versions and across
    prefill + KV-cache decode steps (pruned key columns stay masked as the
    sequence grows, because their absolute indices are fixed).

Non-goal: this variant does NOT save prefill FLOPs (positions still occupy the
sequence). It is the quality-faithful proxy used by the S2 headroom gate.
True token drop (FLOPs-saving) = Variant B, S3.

Does NOT modify code/third_party/AutoVLA (forward-pre-hook + restore on exit).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch


def select_prune_positions(
    vision_token_positions: torch.Tensor,
    score: torch.Tensor,
    keep_ratio: float,
) -> torch.Tensor:
    """Return absolute sequence indices of the vision tokens to PRUNE.

    Keep the top-B = round(keep_ratio * N) vision tokens by `score`; prune the
    rest. Deterministic: ties broken toward keeping the LOWER sequence index
    (stable sort on -score with position as secondary key).

    Args:
        vision_token_positions: (N,) long, absolute indices of vision tokens in
            input_ids[0] (from PromptIndex.vision_token_positions).
        score: (N,) float, importance per vision token, SAME order as
            vision_token_positions (higher = more important -> keep).
        keep_ratio: fraction in (0, 1]. keep_ratio>=1.0 -> prune nothing.

    Returns:
        (N - B,) long tensor of absolute positions to mask out. Empty if
        keep_ratio >= 1.0.
    """
    vp = vision_token_positions.flatten()
    n = int(vp.numel())
    assert score.flatten().numel() == n, (
        f"score len {score.flatten().numel()} != n_vision {n}"
    )
    if keep_ratio >= 1.0 or n == 0:
        return vp.new_empty((0,), dtype=torch.long)
    b = int(round(float(keep_ratio) * n))
    b = max(0, min(n, b))
    if b >= n:
        return vp.new_empty((0,), dtype=torch.long)

    s = score.flatten().to(torch.float32)
    # rank by (score desc, position asc): keep first B -> prune the rest
    order = sorted(range(n), key=lambda i: (-float(s[i]), int(vp[i])))
    prune_local = order[b:]                      # lowest-score indices
    prune_pos = vp[torch.tensor(sorted(prune_local), dtype=torch.long)]
    return prune_pos.to(torch.long)


@contextmanager
def patch_vision_token_prune(
    vlm,
    prune_positions: Optional[torch.Tensor],
    verbose: bool = False,
):
    """Make `prune_positions` unattendable for the whole generate() call by
    flipping them to padding (0) in the 2D attention_mask on every forward.

    Registers a forward-pre-hook (with_kwargs) on the top-level `vlm`
    (Qwen2_5_VLForConditionalGeneration). On each call it clones the incoming
    2D attention_mask and sets `mask[:, prune_positions] = 0`. HF's
    `_update_causal_mask` converts the zeroed columns to -inf pre-softmax.

    No-op (zero hooks, bit-identical to upstream) if prune_positions is None or
    empty -> this is the r=1.0 lossless path.

    Restores the original forward on exit (including on exception).

    Verification items (see spec §3):
      * r=1.0 lossless: empty prune -> identical trajectory (this CM is a no-op).
      * decode-step coverage: if HF ever passes attention_mask=None during
        cached decode, pruned tokens would be attendable that step. We assert
        the mask is present on the FIRST (pre-fill) call and warn-once if a
        later call has None. AutoVLA's generate passes attention_mask through;
        confirmed via the GPU lossless/prune-effect test.
    """
    if prune_positions is None or int(prune_positions.numel()) == 0:
        # explicit no-op path
        yield
        return

    prune_positions = prune_positions.flatten().to(torch.long)
    orig_forward = vlm.forward
    state = {"n_calls": 0, "n_masked": 0, "warned_none": False}

    def _apply(mask: torch.Tensor) -> torch.Tensor:
        # mask: (bsz, seq_len) padding mask, 1=keep 0=pad
        m = mask.clone()
        pp = prune_positions.to(m.device)
        # guard: only touch in-range positions (seq grows during decode; all
        # prune positions are low prompt indices so always in range, but be safe)
        pp = pp[pp < m.shape[-1]]
        m[:, pp] = 0
        return m

    def patched_forward(*args, **kwargs):
        state["n_calls"] += 1
        am = kwargs.get("attention_mask", None)
        if am is not None and am.dim() == 2:
            kwargs = dict(kwargs)
            # --- M-RoPE fix (Qwen2.5-VL) ---
            # Qwen2.5-VL.forward computes 3D M-RoPE via get_rope_index() using the
            # 2D attention_mask to locate the vision block. If we zero pruned vision
            # columns BEFORE that, get_rope_index sees a broken vision block and
            # raises "video_token_id not in list". Fix: on the pre-fill call,
            # precompute position_ids with the ORIGINAL (unmasked) mask and inject
            # them (+ cache rope_deltas). forward then skips get_rope_index, so our
            # masked mask only affects attention — positions stay intact (faithful
            # Variant-A semantics). Decode steps (seq_len==1) use the cached
            # rope_deltas branch and are left untouched here.
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            if (
                kwargs.get("position_ids", None) is None
                and input_ids is not None
                and input_ids.dim() >= 1
                and input_ids.shape[-1] > 1
            ):
                position_ids, rope_deltas = vlm.get_rope_index(
                    input_ids,
                    kwargs.get("image_grid_thw", None),
                    kwargs.get("video_grid_thw", None),
                    kwargs.get("second_per_grid_ts", None),
                    am,  # ORIGINAL unmasked mask -> intact vision block
                )
                vlm.rope_deltas = rope_deltas
                kwargs["position_ids"] = position_ids
                if verbose and not state.get("rope_done", False):
                    state["rope_done"] = True
                    print(
                        f"[token_prune] precomputed M-RoPE position_ids on prefill "
                        f"(seq_len={int(input_ids.shape[-1])}) before masking",
                        flush=True,
                    )
            kwargs["attention_mask"] = _apply(am)
            state["n_masked"] += 1
            if verbose and state["n_masked"] == 1:
                print(
                    f"[token_prune] first mask applied: pruning {int(prune_positions.numel())} "
                    f"vision tokens (2D attention_mask, seq_len={am.shape[-1]})",
                    flush=True,
                )
        elif am is None and not state["warned_none"]:
            state["warned_none"] = True
            print(
                "[token_prune] WARN: attention_mask=None on a forward call; "
                "pruned tokens may be attendable this step. Verify AutoVLA "
                "generate passes attention_mask through decode (spec §3).",
                flush=True,
            )
        return orig_forward(*args, **kwargs)

    vlm.forward = patched_forward
    try:
        yield state
    finally:
        vlm.forward = orig_forward


def select_prune_positions_taucut(
    vision_token_positions: torch.Tensor,
    score: torch.Tensor,
    tau: float,
    min_keep: int = 36,  # minimum 5% of 720 = 36 tokens always kept
) -> torch.Tensor:
    """Return absolute sequence indices of the vision tokens to PRUNE using
    a global threshold τ (τ-cut): keep tokens with score > τ, prune the rest.

    Unlike top-B (`select_prune_positions`), this produces a **variable**
    keep_ratio per scene — the key ingredient of route-B "unified adaptive".

    Args:
        vision_token_positions: (N,) long, absolute indices.
        score: (N,) float, importance per vision token (higher = keep).
        tau: global threshold. Tokens with score > tau are kept.
        min_keep: minimum number of tokens to keep (safety floor).

    Returns:
        (N - n_kept,) long tensor of absolute positions to mask out.
    """
    vp = vision_token_positions.flatten()
    n = int(vp.numel())
    s = score.flatten().to(torch.float32)
    assert s.numel() == n

    keep_mask = s > tau  # bool (N,)
    n_keep = int(keep_mask.sum().item())

    # Safety floor: if fewer than min_keep pass threshold, keep top-min_keep
    if n_keep < min_keep:
        topk_idx = s.topk(min_keep).indices
        keep_mask = torch.zeros(n, dtype=torch.bool)
        keep_mask[topk_idx] = True
        n_keep = min_keep

    prune_local = torch.where(~keep_mask)[0]
    prune_pos = vp[prune_local.to(torch.long)]
    return prune_pos.to(torch.long).sort().values


__all__ = ["select_prune_positions", "select_prune_positions_taucut", "patch_vision_token_prune"]
