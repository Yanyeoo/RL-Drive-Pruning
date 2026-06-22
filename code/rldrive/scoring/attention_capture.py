"""M1.a / M1.b — Attention capture for Qwen2.5-VL (eager attention path).

DESIGN: see docs/_internal/m1_attention_hook_design.md  (Path C).

Goal:
  For one inference pass over a NAVSIM scene, extract a single
  vector of length N_vision_tokens representing
      attention[layer=L*, head-averaged,
                query = last_instruction_token,
                key   = vision_tokens]
  This vector is used:
    - in M1.a as the probe label to pick L*
    - in M1.b as the ranking-distill target for the SFT scorer

Why "Path C" (in-place slice during forward):
  Storing all-layer pre-fill attentions costs ~7 GB / scene
  (Path A: HF generate(output_attentions=True)).
  Slicing & saving 1 row inside the eager attention forward costs
  ~(N_vision_tokens * 4 B) ≈ 1 KB / scene.

State of this file:
  DRAFT  — not yet wired to autovla.py. The patch_attention_capture
  context manager is written against the eager Qwen2_5_VLAttention
  forward signature observed in transformers 4.49.0
  (autovla env at /apdcephfs/private_shayladeng/miniconda3/envs/autovla).
  Open verification items live at the bottom of this file.

  This module deliberately does NOT touch
  code/third_party/AutoVLA/models/autovla.py — we add an opt-in
  wrapper to be called from a fork of `AutoVLA.predict()` once
  navtrain probe-set A is available.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Token-index bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class PromptIndex:
    """Token-index landmarks inside a single (bsz=1) input_ids row.

    All indices are absolute positions inside `input_ids[0]`.

    vision_token_positions
        Sorted list of every position i such that input_ids[0,i] is a vision
        placeholder token (image_token_id or video_token_id). This is what
        the ranking label is over: shape (N_vision_tokens,).
    last_instr_idx
        Position of the last text token of the instruction, i.e. the
        query position we read attention from per Q4.1.b.
    vision_blocks
        List of (start, end) spans, one per vision block (e.g. one per
        camera-video). Useful for per-camera sanity prints, not for the
        label itself.
    """
    vision_token_positions: torch.LongTensor   # (N_vision_tokens,)
    last_instr_idx: int
    vision_blocks: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def n_vision(self) -> int:
        return int(self.vision_token_positions.numel())


def locate_prompt_landmarks(
    input_ids: torch.Tensor,
    vision_start_token_id: int,
    vision_end_token_id: int,
    image_token_id: int,
    video_token_id: int,
    action_start_id: Optional[int] = None,
) -> PromptIndex:
    """Walk a single-row input_ids and find vision token positions + query idx.

    Layout assumption (per Qwen2.5-VL chat template + AutoVLA prompt builder,
    see code/third_party/AutoVLA/models/autovla.py:get_prompt):

        <|im_start|>system ... <|im_end|>
        <|im_start|>user
            <|vision_start|> <video_pad> ... <video_pad> <|vision_end|>   # cam 1
            <|vision_start|> <video_pad> ... <video_pad> <|vision_end|>   # cam 2
            <|vision_start|> <video_pad> ... <video_pad> <|vision_end|>   # cam 3
            "instruction text ..."  <|im_end|>
        <|im_start|>assistant
            [action_start_id] ... generated trajectory tokens ...

    During pre-fill (predict path), action_start_id is NOT in the prompt:
    it appears at decode step 0 as the model's own first generated token.
    So `last_instr_idx = len(prompt) - 1` and the query is "the very last
    token the model sees in the prompt".

    TODO(M1.a): verify on a probe-set A scene that
        processor.decode(input_ids[0, last_instr_idx])
    is a content text token (NOT a chat-template end-marker like
    `<|im_end|>` or `<|im_start|>`). If it is the chat marker, fall back to
    the position just before it. See design doc §"Open issues / unknowns" #3.
    """
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1, (
        "Path C is wired for bsz=1 (one scene per forward, as autovla.predict does). "
        "Batching requires storing per-row landmarks instead."
    )
    row = input_ids[0]

    # vision token positions: image_token or video_token placeholders
    vis_mask = (row == image_token_id) | (row == video_token_id)
    vision_token_positions = torch.nonzero(vis_mask, as_tuple=False).flatten()

    # vision blocks (start, end) — for sanity / per-camera breakdown
    starts = torch.nonzero(row == vision_start_token_id, as_tuple=False).flatten().tolist()
    ends   = torch.nonzero(row == vision_end_token_id,   as_tuple=False).flatten().tolist()
    vision_blocks = list(zip(starts, ends))

    # Query position
    if action_start_id is not None:
        action_pos = (row == action_start_id).nonzero(as_tuple=False).flatten()
        if action_pos.numel() > 0:
            # action token already inside prompt (e.g. training-time path)
            last_instr_idx = int(action_pos[0].item()) - 1
        else:
            last_instr_idx = int(row.numel()) - 1
    else:
        last_instr_idx = int(row.numel()) - 1

    return PromptIndex(
        vision_token_positions=vision_token_positions,
        last_instr_idx=last_instr_idx,
        vision_blocks=vision_blocks,
    )


# ---------------------------------------------------------------------------
# Path C — monkey-patch one layer's eager attention, store 1 row, discard rest
# ---------------------------------------------------------------------------

@contextmanager
def patch_attention_capture(
    vlm,
    layer_idx: int,
    prompt_index: PromptIndex,
    bucket: Optional[Dict] = None,
    average_heads: bool = True,
):
    """Patch vlm.model.layers[layer_idx].self_attn.forward to capture
    attn_weights[query=last_instr_idx, key=vision_token_positions] from the
    FIRST forward call only (pre-fill), then auto-restore.

    Assumes:
      - vlm is a Qwen2_5_VLForConditionalGeneration with
        config._attn_implementation == "eager"
      - autovla.py:line 510 enforces attn_implementation='eager' by default
        on H20 (see SIGFPE notes there).

    The eager Qwen2_5_VLAttention.forward signature (transformers 4.49.0,
    modeling_qwen2_5_vl.py:739-749) returns
        (attn_output, attn_weights, past_key_value)
    where attn_weights is the post-softmax tensor of shape
        (bsz, num_heads, q_len, k_len)
    BUT only when output_attentions=True. To force its computation
    regardless of what HuggingFace generate() does, we wrap forward to
    pass output_attentions=True under the hood, then strip attn_weights
    back to None on the return path so the rest of the model doesn't
    accumulate them across decode steps (which was the 7GB Path A
    problem).

    Args:
        layer_idx: which decoder layer to capture from (0 .. num_hidden_layers-1)
        prompt_index: result of locate_prompt_landmarks()
        bucket: dict that this CM will populate with key "vision_attn".
                If None, a fresh dict is created and yielded.
        average_heads: if True, mean over num_heads → shape (N_vision,).
                       if False, store per-head → shape (num_heads, N_vision).

    Yields:
        bucket dict. After the with-block, bucket["vision_attn"] is a CPU
        tensor of shape (N_vision_tokens,) (head-averaged) or
        (num_heads, N_vision_tokens). If pre-fill never ran inside the
        block (bug or short prompt), bucket["vision_attn"] is absent.

    TODO(M1.a) #1 — first run verification:
        Use Path A on the SAME scene+layer (run with
        outputs = vlm.generate(output_attentions=True, return_dict_in_generate=True))
        and compare bucket["vision_attn"] against
        outputs.attentions[0][layer_idx][0].mean(0)[last_instr_idx,
            prompt_index.vision_token_positions]
        Must match within bf16/fp32 tolerance.

    TODO(M1.a) #2 — verify on a "go straight" probe scene that the top-k
    vision tokens in bucket["vision_attn"] visually correspond to front-
    camera tokens covering road / lead vehicle, not sky / background.
    """
    if bucket is None:
        bucket = {}

    # navigate to the target layer (works under FSDP/Lightning unwrap too —
    # vlm here is the inner HF model, not the LightningModule)
    layer = vlm.model.layers[layer_idx]
    self_attn = layer.self_attn
    orig_forward = self_attn.forward

    # one-shot flag — capture only the very first forward call (pre-fill)
    captured = {"done": False}

    q_idx = prompt_index.last_instr_idx
    # move to same device as attn output once we see it
    vis_pos = prompt_index.vision_token_positions

    def patched_forward(self, hidden_states, *args, **kwargs):
        # force attn_weights to be computed this call, regardless of caller
        force_oa = not captured["done"]
        if force_oa:
            kwargs = dict(kwargs)
            kwargs["output_attentions"] = True

        attn_output, attn_weights, past_kv = orig_forward(
            hidden_states, *args, **kwargs
        )

        if force_oa and attn_weights is not None and not captured["done"]:
            # attn_weights: (bsz=1, num_heads, q_len, k_len)
            # pre-fill means q_len == k_len == full prompt len; in decode
            # steps q_len == 1, so q_idx would be out of range — guard.
            q_len = attn_weights.shape[2]
            if q_idx < q_len:
                row = attn_weights[0, :, q_idx, :]              # (num_heads, k_len)
                vis_pos_dev = vis_pos.to(row.device)
                row_vis = row.index_select(dim=-1, index=vis_pos_dev)  # (num_heads, N_vision)
                if average_heads:
                    row_vis = row_vis.mean(dim=0)               # (N_vision,)
                bucket["vision_attn"] = row_vis.detach().to("cpu", torch.float32)
                bucket["captured_q_len"] = int(q_len)
                captured["done"] = True

            # strip attn_weights back to None so HF generate() doesn't
            # accumulate them; we own the only copy now in `bucket`.
            attn_weights = None

        return attn_output, attn_weights, past_kv

    self_attn.forward = MethodType(patched_forward, self_attn)
    try:
        yield bucket
    finally:
        self_attn.forward = orig_forward


# ---------------------------------------------------------------------------
# High-level wrapper around AutoVLA.predict()
# ---------------------------------------------------------------------------

def predict_with_attention(
    autovla_inner,          # the inner AutoVLA (nn.Module), NOT the GRPOAutoVLA LightningModule
    input_features: dict,
    layer_idx: int,
    average_heads: bool = True,
) -> Tuple[torch.Tensor, str, Dict]:
    """Run a single AutoVLA.predict() and also return per-vision-token attention.

    Returns:
        trajectory: same as AutoVLA.predict()[0]
        cot_results: same as AutoVLA.predict()[1]
        attn_info: dict with keys
            "vision_attn":   (N_vision,) float32 cpu tensor
            "vision_token_positions": (N_vision,) long cpu tensor
            "last_instr_idx": int
            "vision_blocks": list[(start, end)]
            "captured_q_len": int (sanity)

    NOT YET CALLED FROM ANYWHERE — see TODO(M1.a) #3 below.

    TODO(M1.a) #3 — integration point:
      In a thin wrapper script (rldrive/scoring/run_attention_probe.py,
      to be written), replicate the body of AutoVLA.predict() but enter
      `patch_attention_capture(...)` around the
      `outputs = self.vlm.generate(...)` line. Reasons not to live-patch
      autovla.py:
        * keeps the third_party fork unmodified (easier rebase)
        * M0.4 baseline numbers came out of the unpatched generate path;
          we want to be able to A/B against the same code on demand
    """
    raise NotImplementedError(
        "predict_with_attention scaffolding only. Wire up after "
        "(a) chain_complete confirms navtrain+probe_A.txt is on disk, and "
        "(b) Path A vs Path C cross-check on one probe scene passes."
    )


# ---------------------------------------------------------------------------
# Token-id helpers (resolved from the loaded model's config / tokenizer)
# ---------------------------------------------------------------------------

def resolve_vision_token_ids(vlm) -> Dict[str, int]:
    """Pull the four vision-related token ids from the model config.

    Defined in transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1607-1612.
    The action_start_id lives in the AutoVLA repo, NOT in the HF config —
    pass it separately via config['model']['tokens']['action_start_id'].
    """
    cfg = vlm.config
    return {
        "vision_start_token_id": cfg.vision_start_token_id,
        "vision_end_token_id":   getattr(cfg, "vision_end_token_id", None),
        "image_token_id":        cfg.image_token_id,
        "video_token_id":        cfg.video_token_id,
    }


# ---------------------------------------------------------------------------
# Open verification list (mirrors design doc §"Open issues / unknowns")
# ---------------------------------------------------------------------------
#
# V1. GQA / num_kv_heads correctness.
#     RESOLVED by reading modeling_qwen2_5_vl.py:770 (repeat_kv called
#     BEFORE the matmul that produces attn_weights). attn_weights shape
#     is (bsz, num_heads, q_len, k_len) — head-averaging across num_heads
#     is correct and no GQA fan-out is needed by our code.
#
# V2. Pre-fill = single forward?
#     For attn_implementation='eager' on a single-scene predict() with
#     no past_key_value, yes: one full-prompt forward, then loop of
#     1-token decode steps. The captured["done"] one-shot guard handles
#     the case anyway: even if pre-fill were chunked, we'd take the
#     first chunk's last row, which would be wrong. So before turning
#     this on for real M1.b runs:
#     TODO(M1.a) #4 — print captured_q_len once for a probe scene and
#     assert it equals input_ids.shape[1].
#
# V3. "Last instruction token" identity (design doc Q4.1.b).
#     TODO(M1.a) #5 — decode input_ids[0, last_instr_idx] for 3 probe
#     scenes and confirm it's a content text token, not a chat-template
#     marker. If it's an <|im_end|>-style marker, change
#     locate_prompt_landmarks to back up to the previous non-special
#     token. The design doc explicitly listed this as a likely fix-up.
#
# V4. Cross-check Path C vs Path A on at least one scene before scaling.
#     TODO(M1.a) #6 — see docstring of patch_attention_capture.
#
# All TODO(M1.a) items must clear before M1.a layer probing runs are
# trusted; M1.b is gated on M1.a.

__all__ = [
    "PromptIndex",
    "locate_prompt_landmarks",
    "patch_attention_capture",
    "predict_with_attention",
    "resolve_vision_token_ids",
]
