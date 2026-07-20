#!/usr/bin/env bash
# =============================================================================
# run_7b_finetune_lora.sh — LoRA fine-tune Qwen2.5-VL-7B for driving (GRPO)
#
# AutoVLA's training pipeline adapted for 7B backbone.
# Takes ~8h on 4×H20 for 1 epoch on navtrain (~103k scenes).
# For speed: use a subset (--max-scenes) or fewer epochs.
#
# Strategy for 21h window:
#   - Option A (fast, ~4h): SFT on 10k scenes, 1 epoch → decent baseline
#   - Option B (full, ~8h): SFT on full navtrain, 1 epoch → best baseline
#   - While fine-tune runs on GPU0-3, run feature/attention dump on GPU4-7
#     using the BASE 7B model (labels from base model are fine for scorer)
#
# Output: ckpt at runs/grpo_7b/<timestamp>/checkpoints/
# =============================================================================
set -euo pipefail

ROOT=/apdcephfs/private_shayladeng/tokenrl_autoVLA
cd "$ROOT/code/third_party/AutoVLA"

source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$ROOT/code/third_party/AutoVLA:$ROOT/code/third_party/AutoVLA/navsim:$PYTHONPATH"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Config for 7B GRPO fine-tune
CONFIG_FILE="$ROOT/code/third_party/AutoVLA/config/training/qwen2.5-vl-7B-navtest-grpo-nocot.yaml"

echo "========================================="
echo "[7B LoRA Fine-tune] Starting at $(date)"
echo "[7B LoRA Fine-tune] config=$CONFIG_FILE"
echo "========================================="

# For 7B LoRA fine-tune, we need to create a proper training config.
# The key issue: we don't have a 7B SFT checkpoint to load.
# Solution: Start from base Qwen2.5-VL-7B-Instruct, add LoRA, train with GRPO on navtrain.
#
# IMPORTANT: The existing training code expects sft_model_path to load state_dict.
# For 7B base model, we skip this step (model is already loaded from pretrained_model_path).
# This requires a small code patch — see below.

cat << 'PATCH_INFO'
=== REQUIRED PATCH for 7B (no SFT ckpt) ===
In code/third_party/AutoVLA/tools/run_rft.py, around line 115-120,
change the checkpoint loading to skip when sft_model_path is empty:

    sft_path = config['model'].get('sft_model_path', '')
    if sft_path and os.path.exists(sft_path):
        print(f"Loading checkpoint from: {sft_path}")
        full_checkpoint = torch.load(sft_path, map_location="cpu")
        sd = full_checkpoint['state_dict']
        msg = model.load_state_dict(sd, strict=False)
    else:
        print("No SFT checkpoint — starting from base pretrained model")
        
PATCH_INFO

echo ""
echo "To run (after applying patch):"
echo "  cd $ROOT/code/third_party/AutoVLA"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_rft.py --config training/qwen2.5-vl-7B-navtest-grpo-nocot"
echo ""
echo "Estimated time: ~8h on 4×H20 (full navtrain)"
echo "Estimated VRAM: ~60GB/GPU with LoRA + FSDP"
echo ""
echo "Alternative (fast baseline, ~2h):"
echo "  Add --max-scenes 10000 to limit training data"
