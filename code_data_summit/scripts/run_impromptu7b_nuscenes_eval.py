"""run_impromptu7b_nuscenes_eval.py — Token-pruned inference on Impromptu-VLA 7B.

Two-step pipeline:
  Step 1 (this script): Run Impromptu-VLA 7B with our token pruning → output .jsonl
  Step 2 (their script): evaluation_nuscenes.py reads .jsonl → computes L2/Collision/Intersection

This lets us directly compare with FastDriveVLA Table 1 (same model, same eval).

Architecture: Qwen2_5_VLForConditionalGeneration (hidden=3584, layers=28)
Our 7B scorer (ckpt/s3_token_scorer_7b) is directly compatible.

Usage:
    # Step 1: Generate predictions with pruning
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_impromptu7b_nuscenes_eval.py \
        --model-path models/ImpromptuVLA_7B/7B_AD_finetune \
        --scorer-ckpt ckpt/s3_token_scorer_7b \
        --keep-ratio 0.5 \
        --data-json <path_to_nuscenes_test_qa.json> \
        --output results/impromptu7b/pred_r05.jsonl

    # Step 2: Evaluate (use their script)
    cd code/third_party/ImpromptuVLA/data_qa_generate
    python data_engine/datasets/nuscenes/scripts/evaluation_nuscenes.py \
        --jsonl_file <pred_r05.jsonl> \
        --output_file results/impromptu7b/eval_r05.json \
        --mode x-y
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
sys.path.insert(0, str(ROOT / "code"))


def parse_args():
    p = argparse.ArgumentParser(description="Impromptu-VLA 7B inference with token pruning")
    p.add_argument("--model-path", type=str,
                   default=str(ROOT / "models/ImpromptuVLA_7B/7B_AD_finetune"))
    p.add_argument("--scorer-ckpt", type=str,
                   default=str(ROOT / "ckpt/s3_token_scorer_7b"))
    p.add_argument("--keep-ratio", type=float, default=0.5,
                   help="Fraction of vision tokens to keep (0.25/0.5/0.75/1.0)")
    p.add_argument("--data-json", type=str, required=True,
                   help="Path to nuScenes test QA json (ShareGPT format from Impromptu-VLA)")
    p.add_argument("--output", type=str,
                   default=str(ROOT / "results/impromptu7b/pred.jsonl"))
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--skip-first", type=int, default=0,
                   help="skip the first N resolvable samples (for sharding across GPUs)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--image-map", type=str, default=None,
                   help="JSON {qa_image_ref -> on-disk png}. Samples whose images "
                        "are not resolvable are skipped (not written as empty).")
    return p.parse_args()


def load_model(model_path: str, device: str):
    """Load Impromptu-VLA 7B with HuggingFace transformers."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer

    print(f"[infer] Loading model: {model_path}", flush=True)
    t0 = time.time()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # H20 sm_90: bf16 avoids fp16 GEMM SIGFPE
        device_map=device,
        attn_implementation="sdpa",  # eager triggers extra SIGFPE on H20; sdpa+no-cache is stable
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"[infer] Model loaded in {time.time()-t0:.1f}s", flush=True)
    return model, processor, tokenizer


def load_scorer(scorer_ckpt: str, device: str):
    """Load our 7B token importance scorer."""
    from rldrive.scoring.token_scorer import ScorerRunner
    return ScorerRunner(scorer_ckpt, device=device)


_IMG_MAP: Dict[str, str] = {}


def resolve_image(img_path: str) -> Optional[str]:
    """Resolve a QA image ref to an on-disk file.

    Order: direct existing path -> exact map hit -> map hit on 'samples/...' suffix.
    Returns None if unresolvable (caller should skip the sample).
    """
    import re
    if os.path.exists(img_path):
        return img_path
    if img_path in _IMG_MAP:
        return _IMG_MAP[img_path]
    m = re.search(r"samples/.*", img_path)
    if m and m.group(0) in _IMG_MAP:
        return _IMG_MAP[m.group(0)]
    return None


def get_vision_positions(input_ids: torch.Tensor) -> torch.Tensor:
    """Find vision token positions in the input_ids.
    
    Qwen2.5-VL uses special tokens for image placeholders:
    - 151652: <|vision_start|>
    - 151653: <|vision_end|>  
    - 151655: <|image_pad|> (the actual vision tokens)
    """
    IMAGE_PAD_ID = 151655
    positions = (input_ids.flatten() == IMAGE_PAD_ID).nonzero(as_tuple=True)[0]
    return positions


def capture_layer0_features(model, input_ids, pixel_values, image_grid_thw, device):
    """Run forward to capture layer-0 vision embeddings for scorer.
    
    Returns: vision_feat (N_vision, hidden_size=3584)
    """
    feat_bucket = {}
    
    # Hook into the first decoder layer to capture inputs.
    # forward_pre_hook callback signature is (module, args).
    def hook_fn(module, args):
        # args[0] is hidden_states entering the layer: (bsz, seq, hidden)
        if "hidden_states" not in feat_bucket:
            feat_bucket["hidden_states"] = args[0].detach()
    
    # Register hook on first decoder layer
    handle = model.model.layers[0].register_forward_pre_hook(hook_fn)
    
    with torch.no_grad():
        # Just run the prefill (no generation) to get the hidden states
        outputs = model(
            input_ids=input_ids.to(device),
            pixel_values=pixel_values.to(device) if pixel_values is not None else None,
            image_grid_thw=image_grid_thw.to(device) if image_grid_thw is not None else None,
            use_cache=False,
            output_hidden_states=False,
        )
    
    handle.remove()
    
    # Extract vision token features
    hidden_states = feat_bucket["hidden_states"]  # (1, seq, 3584)
    vision_positions = get_vision_positions(input_ids)
    
    if len(vision_positions) == 0:
        return None, vision_positions
    
    vision_feat = hidden_states[0, vision_positions, :]  # (N_vision, 3584)
    return vision_feat, vision_positions


def apply_vision_prune_mask(model, prune_positions: torch.Tensor, seq_len: int):
    """Create an attention mask that prevents attending to pruned vision tokens.
    
    Same logic as our Variant A (attn_mask) in token_prune_patch.py.
    Returns a 2D attention mask (1, seq_len) where pruned positions are 0.
    """
    mask = torch.ones(1, seq_len, dtype=torch.long)
    if prune_positions is not None and len(prune_positions) > 0:
        mask[0, prune_positions] = 0
    return mask


def run_inference_with_pruning(
    model, processor, tokenizer, scorer,
    sample: dict, keep_ratio: float, device: str
) -> Optional[str]:
    """Run one sample through the model with token pruning.
    
    Args:
        sample: dict with 'messages' and 'images' keys (ShareGPT format)
        keep_ratio: fraction of vision tokens to keep
    
    Returns:
        Generated text string (contains <PLANNING>...</PLANNING>)
    """
    # 1. Prepare inputs using processor
    messages = sample["messages"]
    images = sample.get("images", [])
    
    # Build the chat text (without images inline)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # ShareGPT content uses a literal "<image>" marker; Qwen2.5-VL expects the
    # vision placeholder span so the processor's image features have matching tokens.
    if images and "<image>" in text:
        text = text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    
    # Process with images
    if images:
        from PIL import Image
        pil_images = []
        for img_path in images:
            resolved = resolve_image(img_path)
            if resolved is None:
                raise FileNotFoundError(f"unmapped image: {img_path}")
            pil_images.append(Image.open(resolved).convert("RGB"))
        
        inputs = processor(
            text=[text],
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(text, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    
    # 2. If pruning enabled, capture features and score
    if keep_ratio < 1.0 and scorer is not None:
        vision_feat, vision_positions = capture_layer0_features(
            model, input_ids, pixel_values, image_grid_thw, device
        )
        
        if vision_feat is not None and len(vision_positions) > 0:
            n_vision = len(vision_positions)
            n_keep = max(1, int(n_vision * keep_ratio))
            
            # Score tokens. nuScenes uses a single CAM_FRONT image -> one vision
            # block spanning all vision positions (cam id 0 for every token).
            vp = vision_positions.flatten()
            vision_blocks = [(int(vp.min()) - 1, int(vp.max()) + 1)]
            scores = scorer.score(vision_feat, vp, vision_blocks).flatten()  # (N_vision,)
            
            # Select top-K
            _, top_indices = scores.topk(n_keep)
            keep_set = set(top_indices.cpu().numpy())
            
            # Prune positions = vision positions NOT in top-K
            prune_positions = torch.tensor(
                [pos.item() for i, pos in enumerate(vision_positions) if i not in keep_set],
                dtype=torch.long
            )
            
            # Create attention mask with pruned positions masked
            attention_mask = apply_vision_prune_mask(model, prune_positions, input_ids.shape[1])
            inputs["attention_mask"] = attention_mask.to(device)
    
    # 3. Generate
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": 96,   # PLANNING waypoints are short; keep it tight for speed
            "do_sample": False,     # clean greedy
            "temperature": None, "top_p": None, "top_k": None,
            "use_cache": False,     # H20+torch2.4: KV-cache path triggers SIGFPE; disable
        }
        
        generate_inputs = {
            "input_ids": input_ids,
            "attention_mask": inputs.get("attention_mask", torch.ones_like(input_ids)).to(device),
        }
        if pixel_values is not None:
            generate_inputs["pixel_values"] = pixel_values.to(device)
        if image_grid_thw is not None:
            generate_inputs["image_grid_thw"] = image_grid_thw.to(device)
        
        output_ids = model.generate(**generate_inputs, **gen_kwargs)
    
    # 4. Decode output (skip input tokens)
    generated = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return generated


def main():
    args = parse_args()
    device = args.device
    
    # Optional image map (QA ref -> on-disk png)
    global _IMG_MAP
    if args.image_map:
        with open(args.image_map) as f:
            _IMG_MAP = json.load(f)
        print(f"[infer] loaded image map: {len(_IMG_MAP)} entries", flush=True)

    # Load model + scorer
    model, processor, tokenizer = load_model(args.model_path, device)
    
    scorer = None
    if args.keep_ratio < 1.0:
        scorer = load_scorer(args.scorer_ckpt, device)
    
    # Load data (ShareGPT format JSON from Impromptu-VLA)
    print(f"[infer] Loading data from {args.data_json}", flush=True)
    with open(args.data_json, "r") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "data" in data:
        samples = data["data"]
    elif isinstance(data, list):
        samples = data
    else:
        print(f"[infer] ERROR: unexpected data format in {args.data_json}", flush=True)
        sys.exit(1)
    
    # Keep only samples whose images are all resolvable (skip rather than emit empty).
    if _IMG_MAP:
        before = len(samples)
        samples = [s for s in samples
                   if s.get("images") and all(resolve_image(im) is not None for im in s["images"])]
        print(f"[infer] resolvable samples: {len(samples)}/{before} (skipped {before-len(samples)})", flush=True)

    if args.skip_first:
        samples = samples[args.skip_first:]
    if args.max_scenes:
        samples = samples[:args.max_scenes]
    
    print(f"[infer] {len(samples)} samples, keep_ratio={args.keep_ratio}", flush=True)
    
    # Run inference
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_ok = n_err = 0
    
    with open(args.output, "w") as fout:
        for idx, sample in enumerate(tqdm(samples, desc=f"Infer r={args.keep_ratio}")):
            try:
                pred_text = run_inference_with_pruning(
                    model, processor, tokenizer, scorer,
                    sample, args.keep_ratio, device
                )
                
                # Write in the format expected by evaluation_nuscenes.py
                # {"predict": "<PLANNING>...", "label": "...", ...}
                out_record = {
                    "predict": pred_text,
                    "label": sample["messages"][-1]["content"] if len(sample["messages"]) > 1 else "",
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                n_ok += 1
                
            except Exception as e:
                n_err += 1
                fout.write(json.dumps({"predict": "", "label": ""}) + "\n")
                if n_err <= 5:
                    print(f"[infer] [{idx}] ERROR: {type(e).__name__}: {e}", flush=True)
    
    print(f"\n[infer] DONE: {n_ok} ok, {n_err} err -> {args.output}", flush=True)
    print(f"[infer] Next step: run evaluation_nuscenes.py --jsonl_file {args.output} --mode x-y")


if __name__ == "__main__":
    main()
