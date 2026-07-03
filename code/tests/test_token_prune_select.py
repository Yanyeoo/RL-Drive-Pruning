"""S1 unit tests — vision token prune selection logic (CPU, no model/GPU).

Covers docs/specs/dynamic_token_pruning_S1_spec.md §3.1 (no-op) + selection.
Run: <autovla-python> -m pytest code/tests/test_token_prune_select.py
 or: <autovla-python> code/tests/test_token_prune_select.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
from rldrive.agents.token_prune_patch import (  # noqa: E402
    patch_vision_token_prune,
    select_prune_positions,
)


def test_topb_basic():
    vp = torch.arange(10, 18)
    score = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    # keep top-4 {10,12,14,16} -> prune {11,13,15,17}
    assert sorted(select_prune_positions(vp, score, 0.5).tolist()) == [11, 13, 15, 17]


def test_keep_all_is_empty():
    vp = torch.arange(10, 18)
    score = torch.rand(8)
    assert select_prune_positions(vp, score, 1.0).numel() == 0


def test_quarter_keep():
    vp = torch.arange(10, 18)
    score = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    # B = round(0.25*8)=2 keep {10,12}; prune the other 6
    assert sorted(select_prune_positions(vp, score, 0.25).tolist()) == [11, 13, 14, 15, 16, 17]


def test_tiebreak_keeps_lower_position():
    vp = torch.tensor([5, 6, 7, 8])
    s = torch.tensor([0.5, 0.5, 0.5, 0.5])
    # B=2 -> keep lower positions {5,6}, prune {7,8}
    assert sorted(select_prune_positions(vp, s, 0.5).tolist()) == [7, 8]


def test_empty_prune_cm_is_noop():
    class Dummy:
        def forward(self, *a, **k):
            return "ok"

    d = Dummy()
    with patch_vision_token_prune(d, torch.empty(0, dtype=torch.long)):
        assert d.forward() == "ok"
    # forward restored / unchanged
    assert d.forward() == "ok"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
