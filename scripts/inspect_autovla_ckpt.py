"""
Inspect AutoVLA_PDMS_89.ckpt to determine if it is:
  (1) raw lora adapter (lora_A/lora_B keys, PEFT structure)
  (2) PEFT-wrapped but unmerged (base_layer keys)
  (3) merged / full-FT (standard transformer keys only)

Decision matters for MA2 (C4 risk in integration map):
  - if (1) or (2): need to merge lora before inference, or use PEFT-aware loader
  - if (3): can load directly with HF AutoModel.from_pretrained-style logic

Usage:
  python scripts/inspect_autovla_ckpt.py
"""
import sys
import torch
from collections import Counter, defaultdict

CKPT = "/apdcephfs/private_shayladeng/tokenrl_autoVLA/models/AutoVLA/AutoVLA_PDMS_89.ckpt"


def main():
    print(f"[load] {CKPT}")
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)

    print(f"[type] top-level: {type(obj).__name__}")
    if isinstance(obj, dict):
        print(f"[type] top-level keys: {list(obj.keys())[:20]}")
        # Lightning ckpt typically wraps weights under 'state_dict'
        if "state_dict" in obj:
            sd = obj["state_dict"]
            print(f"[note] found 'state_dict' wrapper (Lightning-style)")
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
            print(f"[note] found 'model' wrapper")
        else:
            # assume the whole thing is state_dict
            sd = obj
    else:
        sd = obj

    if not isinstance(sd, dict):
        print(f"[error] state_dict is not a dict, got {type(sd)}")
        sys.exit(1)

    keys = list(sd.keys())
    print(f"[count] total tensors: {len(keys)}")

    # LoRA / PEFT detection
    lora_keys = [k for k in keys if "lora_A" in k or "lora_B" in k or "lora_embedding" in k]
    base_layer_keys = [k for k in keys if ".base_layer." in k]
    print(f"[lora] keys containing 'lora_A/B/embedding': {len(lora_keys)}")
    print(f"[lora] keys containing '.base_layer.': {len(base_layer_keys)}")

    if lora_keys:
        print(f"[lora-sample] first 5 lora keys:")
        for k in lora_keys[:5]:
            print(f"    {k}  shape={tuple(sd[k].shape)}")

    # Top-level prefix histogram (first 2 path segments)
    prefix_counter = Counter()
    for k in keys:
        parts = k.split(".")
        prefix = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        prefix_counter[prefix] += 1
    print(f"[prefix] top-20 prefixes by tensor count:")
    for p, c in prefix_counter.most_common(20):
        print(f"    {c:5d}  {p}")

    # Total param count + dtype breakdown
    total_params = 0
    dtype_counter = Counter()
    for k, v in sd.items():
        if hasattr(v, "numel"):
            total_params += v.numel()
            dtype_counter[str(v.dtype)] += 1
    print(f"[params] total scalars: {total_params:,} ({total_params/1e9:.2f} B)")
    print(f"[dtype] dtype histogram: {dict(dtype_counter)}")

    # Verdict
    print("\n========== VERDICT ==========")
    if lora_keys:
        print("RED: unmerged LoRA adapter detected (lora_A/lora_B keys present)")
        print("     -> C4 risk MATERIALIZED: must merge lora or use PEFT loader before inference")
    elif base_layer_keys:
        print("YELLOW: PEFT base_layer keys present without lora_A/B")
        print("     -> possibly partially merged or unusual PEFT state, inspect manually")
    else:
        print("GREEN: no lora/PEFT keys detected; likely merged or full-FT checkpoint")
        print("     -> C4 risk DOWNGRADED: standard HF-style loading should work")


if __name__ == "__main__":
    main()
