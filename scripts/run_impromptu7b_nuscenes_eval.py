"""run_impromptu7b_nuscenes_eval.py — Evaluate our token pruning on Impromptu-VLA 7B.

Goal: Reproduce FastDriveVLA Table 1 format (L2/Collision/Intersection at 25%/50%/75% pruning)
on the same Impromptu-VLA 7B backbone, using OUR scorer for token selection.

Strategy:
1. Load Impromptu-VLA 7B (Qwen2.5-VL-7B fine-tuned on nuScenes)
2. For each nuScenes val scene:
   a. Forward pass to get vision features + attention (for scorer)
   b. Apply our 7B scorer to select top-K tokens
   c. Run generation with pruned tokens (attention mask variant)
   d. Parse output trajectory (x-y text format)
3. Compute L2 error at 1s/2s/3s vs GT trajectory
4. Compare with FastDriveVLA Table 1 numbers

Key differences from our NAVSIM pipeline:
- Model: Impromptu-VLA 7B (not AutoVLA 3B)
- Output format: text x-y coordinates (not codebook tokens)
- Eval metric: L2 error (not PDMS)
- Data: nuScenes val (not NAVSIM navtest)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_impromptu7b_nuscenes_eval.py \
        --model-path models/ImpromptuVLA_7B/7B_AD_finetune \
        --scorer-ckpt ckpt/s3_token_scorer_7b \
        --keep-ratio 0.5 \
        --data-dir <nuScenes QA data> \
        --output results/impromptu7b_r05.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/third_party/AutoVLA/navsim"))
sys.path.insert(0, str(ROOT / "code/third_party/AutoVLA"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str,
                   default=str(ROOT / "models/ImpromptuVLA_7B/7B_AD_finetune"))
    p.add_argument("--scorer-ckpt", type=str,
                   default=str(ROOT / "ckpt/s3_token_scorer_7b"))
    p.add_argument("--keep-ratio", type=float, default=0.5)
    p.add_argument("--data-dir", type=str, default=None,
                   help="Dir with nuScenes QA jsonl (from Impromptu-VLA data pipeline)")
    p.add_argument("--output", type=str, default=str(ROOT / "results/impromptu7b_eval.json"))
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--baseline", action="store_true",
                   help="Run without pruning (r=1.0) for baseline comparison")
    return p.parse_args()


def load_model_and_processor(model_path: str, device: str):
    """Load Impromptu-VLA 7B using standard HuggingFace transformers."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"[eval] Loading model from {model_path}...", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",  # needed for attention capture
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"[eval] Model loaded in {time.time()-t0:.1f}s", flush=True)
    return model, processor


def load_scorer(scorer_ckpt: str, device: str):
    """Load our trained 7B token importance scorer."""
    from rldrive.scoring.token_scorer import ScorerRunner
    scorer = ScorerRunner(scorer_ckpt, device=device)
    print(f"[eval] Scorer loaded from {scorer_ckpt}", flush=True)
    return scorer


def get_vision_token_positions(input_ids: torch.Tensor, image_token_id: int = 151655) -> torch.Tensor:
    """Find positions of vision tokens in the input sequence."""
    # Qwen2.5-VL uses a specific token ID for image placeholders
    # After processing, vision tokens are contiguous blocks
    positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    return positions


def apply_attention_mask_pruning(
    model,
    prune_positions: torch.Tensor,
    seq_len: int,
):
    """Apply attention mask to prune vision tokens (same as our Variant A)."""
    # This creates a causal mask where pruned positions cannot be attended to
    # Implementation mirrors our token_prune_patch.py
    pass  # Will be filled with actual hook logic during integration


def parse_trajectory_from_output(text: str) -> Optional[np.ndarray]:
    """Parse x-y trajectory points from model output text.
    
    Expected format from Impromptu-VLA:
    <PLANNING>Trajectory points for the next 3 seconds: [x1, y1], [x2, y2], ...</PLANNING>
    
    Or direct coordinate format:
    (x1, y1), (x2, y2), ...
    """
    # Try PLANNING tag format first
    planning_match = re.search(r'<PLANNING>(.*?)</PLANNING>', text, re.DOTALL)
    if planning_match:
        text = planning_match.group(1)
    
    # Extract coordinate pairs
    pattern = r'\[([^\]]+)\]|\(([^\)]+)\)'
    matches = re.findall(pattern, text)
    
    points = []
    for m in matches:
        coord_str = m[0] if m[0] else m[1]
        try:
            parts = [float(x.strip()) for x in coord_str.split(',')]
            if len(parts) >= 2:
                points.append(parts[:2])
        except (ValueError, IndexError):
            continue
    
    if len(points) == 0:
        return None
    
    return np.array(points)  # (N, 2)


def compute_l2_metrics(pred_traj: np.ndarray, gt_traj: np.ndarray) -> dict:
    """Compute L2 displacement error at 1s, 2s, 3s.
    
    Both trajectories are in ego-relative coordinates (x=lateral, y=longitudinal).
    Assumes 2Hz sampling (0.5s per step).
    """
    # Cumulative sum to get absolute positions (if trajectories are displacements)
    # Impromptu-VLA outputs cumulative positions from ego origin
    
    metrics = {}
    # At 2Hz: 1s=2steps, 2s=4steps, 3s=6steps
    for t_sec, t_steps in [(1, 2), (2, 4), (3, 6)]:
        if t_steps <= len(pred_traj) and t_steps <= len(gt_traj):
            pred_pos = pred_traj[t_steps - 1]
            gt_pos = gt_traj[t_steps - 1]
            l2 = np.sqrt(np.sum((pred_pos - gt_pos) ** 2))
            metrics[f"L2_{t_sec}s"] = float(l2)
        else:
            metrics[f"L2_{t_sec}s"] = float('inf')
    
    # Average L2
    valid = [v for v in metrics.values() if v != float('inf')]
    metrics["L2_avg"] = float(np.mean(valid)) if valid else float('inf')
    
    return metrics


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = "cuda:0"
    
    # Load model + scorer
    model, processor = load_model_and_processor(args.model_path, device)
    
    if not args.baseline:
        scorer = load_scorer(args.scorer_ckpt, device)
    else:
        scorer = None
    
    keep_ratio = 1.0 if args.baseline else args.keep_ratio
    print(f"[eval] keep_ratio={keep_ratio} baseline={args.baseline}", flush=True)
    
    # Load evaluation data
    # TODO: Integrate with Impromptu-VLA's nuScenes data loader
    # For now, this is a placeholder for the data loading logic
    if args.data_dir is None:
        print("[eval] ERROR: --data-dir required. Point to Impromptu-VLA nuScenes QA data.", flush=True)
        print("[eval] Generate with: python data_qa_generate/data_engine/datasets/nuscenes/scripts/evaluation_nuscenes.py", flush=True)
        sys.exit(1)
    
    # Load jsonl data (Impromptu-VLA format)
    data_path = Path(args.data_dir)
    if data_path.suffix == '.jsonl':
        samples = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    elif data_path.is_dir():
        # Look for the evaluation jsonl
        jsonl_files = list(data_path.glob("*.jsonl"))
        if not jsonl_files:
            print(f"[eval] ERROR: no .jsonl files in {data_path}", flush=True)
            sys.exit(1)
        samples = []
        for f in jsonl_files:
            samples.extend([json.loads(l) for l in f.read_text().splitlines() if l.strip()])
    else:
        print(f"[eval] ERROR: {data_path} not found", flush=True)
        sys.exit(1)
    
    if args.max_scenes:
        samples = samples[:args.max_scenes]
    
    print(f"[eval] Loaded {len(samples)} samples", flush=True)
    
    # Run evaluation
    results = []
    n_ok = n_err = 0
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            # TODO: Full integration with Impromptu-VLA's inference + our pruning
            # This requires:
            # 1. Process images through Qwen2.5-VL vision encoder
            # 2. Capture layer-0 features for scorer
            # 3. Score tokens, select top-K
            # 4. Apply attention mask on pruned tokens
            # 5. Generate text output
            # 6. Parse trajectory
            # 7. Compare with GT
            
            # Placeholder for actual implementation
            pass
            
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"[eval] [{idx}] ERROR: {e}", flush=True)
    
    # Compute aggregate metrics
    if results:
        avg_metrics = {}
        for key in results[0].keys():
            vals = [r[key] for r in results if key in r and r[key] != float('inf')]
            avg_metrics[key] = float(np.mean(vals)) if vals else None
        
        # Compute relative performance (vs no-prune baseline)
        # Rel. = baseline_L2 / pruned_L2 * 100% (higher is better)
        print(f"\n[eval] Results (keep_ratio={keep_ratio}, N={len(results)}):")
        for k, v in avg_metrics.items():
            print(f"  {k}: {v:.4f}" if v else f"  {k}: N/A")
        
        # Save
        output = {
            "config": {
                "model": args.model_path,
                "scorer": args.scorer_ckpt,
                "keep_ratio": keep_ratio,
                "n_samples": len(results),
                "baseline": args.baseline,
            },
            "metrics": avg_metrics,
            "per_sample": results,
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[eval] Saved to {args.output}")
    else:
        print("[eval] No results generated. Integration TODO.")


if __name__ == "__main__":
    main()
