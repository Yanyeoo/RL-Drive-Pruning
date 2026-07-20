"""autovla_7b_adapter.py — Adapter to run Qwen2.5-VL-7B as an AutoVLA-style agent.

The original AutoVLA codebase only supports 3B (with a fine-tuned ckpt).
For the 7B scaling experiment, we need to:
1. Load Qwen2.5-VL-7B-Instruct (base model, no driving fine-tune ckpt)
2. Use the same prompt template as AutoVLA (nocot instruction → action tokens)
3. Generate trajectories using the same codebook decoding

Key differences from 3B:
- hidden_size: 2048 → 3584
- num_hidden_layers: 36 → 28  (NOTE: Qwen2.5-VL-7B has 28 layers!)
- num_attention_heads: 16 → 28
- No driving-specific fine-tuning (base model → may have worse PDMS)

For the token pruning paper, what matters is:
  "Given a 7B VLA backbone, how much can we prune?"
  Even with a weaker baseline, the *relative* pruning tolerance is the key result.

Strategy:
- Option A: Use base Qwen2.5-VL-7B (no driving fine-tune) — shows pruning tolerance
- Option B: Fine-tune 7B on navtrain with LoRA first, then apply scorer
  → Option B is better for paper but costs ~8h extra
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Dict

# Ensure paths
ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/third_party/AutoVLA"))
sys.path.insert(0, str(ROOT / "code/third_party/AutoVLA/navsim"))


def get_7b_model_info():
    """Return key architecture info for 7B model."""
    import json
    config_path = ROOT / "models/Qwen2.5-VL-7B-Instruct/config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"7B model not found at {config_path}")
    cfg = json.loads(config_path.read_text())
    return {
        "hidden_size": cfg["hidden_size"],           # 3584
        "num_hidden_layers": cfg["num_hidden_layers"],  # 28
        "num_attention_heads": cfg["num_attention_heads"],  # 28
        "num_key_value_heads": cfg.get("num_key_value_heads", 4),  # GQA: 4
        "model_path": str(ROOT / "models/Qwen2.5-VL-7B-Instruct"),
    }


def patch_autovla_for_7b(config: dict) -> dict:
    """Patch an AutoVLA config dict to use 7B model.
    
    Modifies model.pretrained_model_path and removes sft_model_path
    (since we don't have a 7B fine-tuned ckpt).
    """
    config = config.copy()
    if "model" not in config:
        config["model"] = {}
    config["model"]["pretrained_model_path"] = str(ROOT / "models/Qwen2.5-VL-7B-Instruct")
    config["model"]["sft_model_path"] = ""  # no fine-tuned ckpt
    return config


class AutoVLA7BAgent:
    """Minimal wrapper for running 7B as token-pruning target.
    
    Unlike the full AutoVLAAgent which requires a driving-finetuned ckpt,
    this loads the base 7B model and uses it for:
    1. Feature extraction (for scorer training)
    2. Attention probing (for labels)
    3. Token pruning evaluation (scorer applied at inference)
    
    For PDMS evaluation, we still need a driving-capable model.
    Options:
    - Fine-tune 7B with LoRA on navtrain (recommended, ~8h)
    - Use 7B base model zero-shot (weaker baseline, but pruning comparison still valid)
    """
    pass  # Implementation in the pipeline scripts


# ============================================================================
# 7B Scorer Architecture Config
# ============================================================================

# For reference: 3B scorer uses emb_dim=2048 (layer-0 hidden state of Qwen2.5-VL-3B)
# 7B scorer must use emb_dim=3584 (layer-0 hidden state of Qwen2.5-VL-7B)
SCORER_7B_CONFIG = {
    "emb_dim": 3584,  # Qwen2.5-VL-7B hidden_size
    "n_cam": 3,       # same 3 cameras
    "hidden": 256,    # same MLP hidden (could increase to 512 for capacity)
}


# ============================================================================
# Quick-start commands for 7B experiments
# ============================================================================

QUICKSTART = """
# ==========  7B Token Pruning Experiment Quick-Start  ==========

# 0. Pre-check
python3 -c "import json; c=json.load(open('models/Qwen2.5-VL-7B-Instruct/config.json')); print(f'7B ready: hidden={c[\"hidden_size\"]}, layers={c[\"num_hidden_layers\"]}')"

# 1. Feature dump (4 GPUs, ~6h for 4000 scenes)
bash scripts/run_7b_pipeline.sh features

# 2. Attention probe (4 GPUs, ~6h for 4000 scenes) — can run parallel with features on separate GPUs
bash scripts/run_7b_pipeline.sh attention

# 3. Train scorer (~30s, single GPU)
bash scripts/run_7b_pipeline.sh train

# 4. Eval (4-8 GPUs, ~4h for full navtest)
bash scripts/run_7b_pipeline.sh eval

# Or run everything:
bash scripts/run_7b_pipeline.sh all
"""

if __name__ == "__main__":
    print(QUICKSTART)
    try:
        info = get_7b_model_info()
        print(f"7B model info: {info}")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Download: python3 -c \"from huggingface_hub import snapshot_download; "
              "snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', "
              "local_dir='models/Qwen2.5-VL-7B-Instruct')\"")
