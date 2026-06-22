"""rldrive agents — wrappers around third_party AutoVLA agents that add
project-specific instrumentation (attention capture, pruning hooks, etc.).
Do NOT modify third_party/AutoVLA directly — subclass and override here.
"""

from rldrive.agents.head_mask_patch import patch_head_mask  # noqa: F401

__all__ = ["patch_head_mask"]
