"""Unit tests for rldrive.scoring.attention_capture.

Run with the autovla env:
    PYTHONPATH=/apdcephfs/private_shayladeng/tokenrl_autoVLA/code \
    /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python \
        code/rldrive/scoring/test_attention_capture.py

Pure-tensor tests (no model load). Covers:
  T1  locate_prompt_landmarks: single vision block
  T2  locate_prompt_landmarks: 3 vision blocks (3-camera typical case)
  T3  locate_prompt_landmarks: image_token + video_token mix
  T4  locate_prompt_landmarks: action_start_id present (training-path)
  T5  locate_prompt_landmarks: NO vision tokens (edge case, must not crash)
  T6  patch_attention_capture: captures the right row from a toy nn.Module
       that imitates the eager attention return signature
  T7  patch_attention_capture: per-head storage (average_heads=False)
  T8  patch_attention_capture: pre-fill chunked (captured_q_len < prompt_len)
       — confirms one-shot guard fires only on the first call
"""
from __future__ import annotations

import sys
import torch

from rldrive.scoring.attention_capture import (
    PromptIndex,
    locate_prompt_landmarks,
    patch_attention_capture,
)


def _ok(name: str) -> None:
    print(f"  OK  {name}")


def t1_single_vision_block() -> None:
    # tokens: SYS=1, ME=2, U=3, VS=99, VID=77, VE=98, T=42
    ids = torch.tensor([[1, 1, 2, 3, 99, 77, 77, 77, 98, 42, 42, 2]])
    pi = locate_prompt_landmarks(
        ids,
        vision_start_token_id=99,
        vision_end_token_id=98,
        image_token_id=88,    # absent
        video_token_id=77,
        action_start_id=None,
    )
    assert pi.vision_token_positions.tolist() == [5, 6, 7], pi.vision_token_positions
    assert pi.last_instr_idx == 11
    assert pi.vision_blocks == [(4, 8)]
    assert pi.n_vision == 3
    _ok("T1 single vision block")


def t2_three_cam_blocks() -> None:
    # 3 cameras, 2 video tokens each
    ids = torch.tensor([[1, 99, 77, 77, 98, 99, 77, 77, 98, 99, 77, 77, 98, 42, 2]])
    pi = locate_prompt_landmarks(
        ids,
        vision_start_token_id=99, vision_end_token_id=98,
        image_token_id=88, video_token_id=77, action_start_id=None,
    )
    assert pi.vision_token_positions.tolist() == [2, 3, 6, 7, 10, 11]
    assert pi.last_instr_idx == 14
    assert pi.vision_blocks == [(1, 4), (5, 8), (9, 12)]
    assert pi.n_vision == 6
    _ok("T2 three-cam blocks")


def t3_image_and_video_mix() -> None:
    ids = torch.tensor([[99, 88, 88, 98, 99, 77, 77, 98, 42]])
    pi = locate_prompt_landmarks(
        ids,
        vision_start_token_id=99, vision_end_token_id=98,
        image_token_id=88, video_token_id=77, action_start_id=None,
    )
    assert pi.vision_token_positions.tolist() == [1, 2, 5, 6]
    assert pi.vision_blocks == [(0, 3), (4, 7)]
    _ok("T3 image+video mix")


def t4_action_start_in_prompt() -> None:
    # training-path: action_start_id (=999) appears before generated tokens
    ids = torch.tensor([[1, 99, 77, 98, 42, 42, 999, 7, 8]])
    pi = locate_prompt_landmarks(
        ids,
        vision_start_token_id=99, vision_end_token_id=98,
        image_token_id=88, video_token_id=77,
        action_start_id=999,
    )
    assert pi.last_instr_idx == 5, f"expected 5 (idx before 999), got {pi.last_instr_idx}"
    _ok("T4 action_start in prompt (training path)")


def t5_no_vision_tokens() -> None:
    # Pure text — vision_token_positions is empty, must not crash
    ids = torch.tensor([[1, 2, 3, 42, 42, 2]])
    pi = locate_prompt_landmarks(
        ids,
        vision_start_token_id=99, vision_end_token_id=98,
        image_token_id=88, video_token_id=77, action_start_id=None,
    )
    assert pi.vision_token_positions.tolist() == []
    assert pi.n_vision == 0
    assert pi.last_instr_idx == 5
    assert pi.vision_blocks == []
    _ok("T5 no vision tokens")


# --------- patch_attention_capture tests -----------

class _ToyAttn(torch.nn.Module):
    """Stub mimicking Qwen2_5_VLAttention.forward return signature.

    Returns (attn_output, attn_weights, past_key_value) with shapes:
      attn_output:  (bsz, q_len, hidden)
      attn_weights: (bsz, num_heads, q_len, k_len)  iff output_attentions else None
      past_key_value: just None for our tests
    """
    def __init__(self, num_heads: int = 4, hidden: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.hidden = hidden

    def forward(self, hidden_states, *args, output_attentions=False, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        k_len = kwargs.get("_k_len_override", q_len)  # so we can fake chunked pre-fill
        attn_output = torch.zeros(bsz, q_len, self.hidden)
        if output_attentions:
            # craft a known attn map so the slice can be verified exactly
            attn_weights = torch.arange(
                bsz * self.num_heads * q_len * k_len, dtype=torch.float32
            ).reshape(bsz, self.num_heads, q_len, k_len)
        else:
            attn_weights = None
        return attn_output, attn_weights, None


class _ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _ToyAttn(num_heads=4, hidden=16)


class _ToyModel(torch.nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyLayer() for _ in range(n_layers)])


class _ToyVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ToyModel(n_layers=3)


def _make_pi(vision_positions, last_instr_idx) -> PromptIndex:
    return PromptIndex(
        vision_token_positions=torch.tensor(vision_positions, dtype=torch.long),
        last_instr_idx=last_instr_idx,
        vision_blocks=[],
    )


def t6_basic_capture_head_averaged() -> None:
    vlm = _ToyVLM()
    pi = _make_pi(vision_positions=[2, 3, 5], last_instr_idx=7)

    # call forward once with q_len=8 (so q_idx=7 is in range)
    hs = torch.zeros(1, 8, 16)

    with patch_attention_capture(vlm, layer_idx=1, prompt_index=pi) as bucket:
        out, w, _ = vlm.model.layers[1].self_attn(hs)
        # the patch must have stripped attn_weights to None on the return
        assert w is None, "patch must strip attn_weights to None after capture"

    assert "vision_attn" in bucket
    assert bucket["vision_attn"].shape == (3,), bucket["vision_attn"].shape
    assert bucket["captured_q_len"] == 8

    # verify the values are exactly the head-mean of the known arange
    # attn_weights[0, :, 7, :] = arange offsets for q=7 → slice + mean
    expected_full = torch.arange(1 * 4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8)
    expected_row = expected_full[0, :, 7, [2, 3, 5]].mean(0)
    assert torch.allclose(bucket["vision_attn"], expected_row), \
        f"got {bucket['vision_attn']}, want {expected_row}"

    # also check that forward is restored after the context manager
    out2, w2, _ = vlm.model.layers[1].self_attn(hs, output_attentions=True)
    assert w2 is not None, "after context exit, output_attentions=True must work normally again"

    _ok("T6 basic head-averaged capture + restore")


def t7_per_head_storage() -> None:
    vlm = _ToyVLM()
    pi = _make_pi(vision_positions=[1, 4], last_instr_idx=5)
    hs = torch.zeros(1, 6, 16)

    with patch_attention_capture(vlm, layer_idx=0, prompt_index=pi, average_heads=False) as bucket:
        _ = vlm.model.layers[0].self_attn(hs)

    assert bucket["vision_attn"].shape == (4, 2), bucket["vision_attn"].shape  # (num_heads, N_vision)
    _ok("T7 per-head storage")


def t8_one_shot_guard() -> None:
    """The one-shot guard means only the FIRST forward populates bucket;
    subsequent decode-step forwards (q_len=1, q_idx out of range) must not
    overwrite or crash."""
    vlm = _ToyVLM()
    pi = _make_pi(vision_positions=[2, 3], last_instr_idx=5)

    with patch_attention_capture(vlm, layer_idx=2, prompt_index=pi) as bucket:
        # call 1: pre-fill, q_len=6, q_idx=5 in range
        hs1 = torch.zeros(1, 6, 16)
        _ = vlm.model.layers[2].self_attn(hs1)
        saved_first = bucket["vision_attn"].clone()

        # call 2: decode step, q_len=1, q_idx=5 out of range → no overwrite
        hs2 = torch.zeros(1, 1, 16)
        _ = vlm.model.layers[2].self_attn(hs2)

        assert torch.equal(bucket["vision_attn"], saved_first), \
            "second forward must not overwrite captured row"
    _ok("T8 one-shot guard survives decode steps")


def main():
    print("Running attention_capture unit tests...")
    t1_single_vision_block()
    t2_three_cam_blocks()
    t3_image_and_video_mix()
    t4_action_start_in_prompt()
    t5_no_vision_tokens()
    t6_basic_capture_head_averaged()
    t7_per_head_storage()
    t8_one_shot_guard()
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
