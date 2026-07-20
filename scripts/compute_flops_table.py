"""Compute theoretical FLOPs table for AutoVLA token pruning (AAAI paper).

Includes attention O(n²) term + FFN O(n) term for accurate LLM prefill estimate.

Model: Qwen2.5-VL-3B
  ViT: 32 layers, hidden=1280, intermediate=3420, 16 heads, patch=14
  LLM: 36 layers, hidden=2048, intermediate=11008, 16 heads (GQA 2 KV heads)

Deployment architecture (single-pass):
  ViT (once, all 720 tokens) → layer-0 embeddings → MLP scorer (0.6M, ~ms)
  → top-K selection → LLM prefill (text + kept vision tokens) → decode

Note: ViT always processes all 720 tokens (pruning is AFTER ViT output).
      FLOPs saving is entirely in LLM prefill.
"""
import json
from pathlib import Path

# --- Model constants ---
# ViT (InternViT-like in Qwen2.5-VL)
VIT_LAYERS = 32
VIT_HIDDEN = 1280
VIT_FFN = 3420
VIT_HEADS = 16
VIT_HEAD_DIM = VIT_HIDDEN // VIT_HEADS  # 80

# LLM (Qwen2.5-3B)
LLM_LAYERS = 36
LLM_HIDDEN = 2048
LLM_FFN = 11008
LLM_HEADS = 16
LLM_KV_HEADS = 2  # GQA
LLM_HEAD_DIM = LLM_HIDDEN // LLM_HEADS  # 128

# Tokens
N_VISION = 720       # 3 cameras × 240 tokens/cam
N_TEXT = 221         # typical instruction + system tokens
N_TOTAL = N_VISION + N_TEXT  # 941

# Scorer
SCORER_PARAMS = 0.6e6  # 0.6M params → ~1.2M FLOPs (negligible)


def vit_flops(n_tokens: int) -> float:
    """ViT FLOPs for n_tokens (all layers). Includes self-attention + FFN."""
    per_layer = 0
    # Self-attention: Q/K/V projections + attention matmul + output projection
    # QKV: 3 × (2 × n × d × d) = 6 × n × d²
    per_layer += 6 * n_tokens * VIT_HIDDEN * VIT_HIDDEN
    # Attention scores: n × n × d_head × n_heads = 2 × n² × d
    per_layer += 2 * n_tokens * n_tokens * VIT_HIDDEN
    # Output projection: 2 × n × d × d
    per_layer += 2 * n_tokens * VIT_HIDDEN * VIT_HIDDEN
    # FFN: 2 × (2 × n × d × d_ffn) = 4 × n × d × d_ffn (gate + up + down for SwiGLU: ×3)
    # Qwen2.5-VL ViT uses SwiGLU: gate(d→ffn) + up(d→ffn) + down(ffn→d) = 3 linear
    per_layer += 3 * 2 * n_tokens * VIT_HIDDEN * VIT_FFN

    return per_layer * VIT_LAYERS


def llm_prefill_flops(seq_len: int) -> float:
    """LLM prefill FLOPs for seq_len tokens (all layers).

    Includes:
    - QKV projections (linear, O(n))
    - Attention scores (quadratic, O(n²))
    - Output projection (linear, O(n))
    - FFN (linear, O(n)) — SwiGLU: gate + up + down
    """
    per_layer = 0

    # QKV projections: Q = n × d × d, K = n × d × d_kv, V = n × d × d_kv
    # GQA: K,V have kv_heads × head_dim = 2 × 128 = 256 dim
    d_kv = LLM_KV_HEADS * LLM_HEAD_DIM  # 256
    per_layer += 2 * seq_len * LLM_HIDDEN * LLM_HIDDEN        # Q projection
    per_layer += 2 * seq_len * LLM_HIDDEN * d_kv               # K projection
    per_layer += 2 * seq_len * LLM_HIDDEN * d_kv               # V projection

    # Attention scores: Q @ K^T = n × n × d_head × n_heads (but with GQA, repeated)
    # Effectively: 2 × n² × d (full heads dimension)
    per_layer += 2 * seq_len * seq_len * LLM_HIDDEN

    # Attention @ V: n × n × d → same as scores
    per_layer += 2 * seq_len * seq_len * LLM_HIDDEN

    # Output projection: n × d × d
    per_layer += 2 * seq_len * LLM_HIDDEN * LLM_HIDDEN

    # FFN (SwiGLU): gate(d→ffn) + up(d→ffn) + down(ffn→d) = 3 linears
    per_layer += 3 * 2 * seq_len * LLM_HIDDEN * LLM_FFN

    return per_layer * LLM_LAYERS


def scorer_flops() -> float:
    """MLP scorer: negligible (~1.2M FLOPs for 720 tokens × 2-layer MLP)."""
    return 2 * SCORER_PARAMS * N_VISION  # rough: 2 × params × tokens


def main():
    print("=" * 90)
    print("AutoVLA Token Pruning — Theoretical FLOPs Breakdown (AAAI Table)")
    print("=" * 90)
    print(f"Model: Qwen2.5-VL-3B (ViT {VIT_LAYERS}L + LLM {LLM_LAYERS}L)")
    print(f"Vision tokens: {N_VISION} (3 cams × 240/cam)")
    print(f"Text tokens: {N_TEXT}")
    print(f"Scorer: 0.6M MLP (negligible)")
    print()

    # ViT is constant (always processes all 720 tokens)
    vit = vit_flops(N_VISION)
    scorer = scorer_flops()

    ratios = [1.0, 0.75, 0.5, 0.25]
    results = []

    print(f"{'Ratio':<8} {'Kept':<6} {'Seq_len':<8} {'ViT(G)':<10} {'Scorer(G)':<11} "
          f"{'LLM_pf(G)':<11} {'Total(G)':<10} {'Save_vs_r1':<12} {'LLM_save':<10}")
    print("-" * 90)

    full_total = None
    full_llm = None

    for r in ratios:
        n_kept = int(round(r * N_VISION))
        seq_len = N_TEXT + n_kept
        llm = llm_prefill_flops(seq_len)
        total = vit + scorer + llm

        if full_total is None:
            full_total = total
            full_llm = llm

        save_total = 1.0 - total / full_total
        save_llm = 1.0 - llm / full_llm

        row = {
            "ratio": r,
            "n_vision_kept": n_kept,
            "seq_len": seq_len,
            "vit_gflops": vit / 1e9,
            "scorer_gflops": scorer / 1e9,
            "llm_prefill_gflops": llm / 1e9,
            "total_gflops": total / 1e9,
            "saving_total": f"{save_total*100:.1f}%",
            "saving_llm": f"{save_llm*100:.1f}%",
        }
        results.append(row)

        print(f"{r:<8.2f} {n_kept:<6} {seq_len:<8} {vit/1e9:<10.1f} {scorer/1e9:<11.4f} "
              f"{llm/1e9:<11.1f} {total/1e9:<10.1f} {save_total*100:<12.1f}% {save_llm*100:<10.1f}%")

    print()
    print("Key observations:")
    print(f"  • ViT = {vit/1e9:.1f} GFLOPs (constant, pruning is after ViT)")
    print(f"  • LLM prefill at r=1.0 = {full_llm/1e9:.1f} GFLOPs (dominates)")
    print(f"  • LLM prefill has O(n²) attention: r=0.5 saves {results[2]['saving_llm']} of LLM prefill")
    print(f"  • Total saving at r=0.5 = {results[2]['saving_total']} (because ViT is fixed cost)")
    print(f"  • Scorer overhead: {scorer/1e9:.4f} GFLOPs = {scorer/full_total*100:.3f}% of total (negligible)")
    print()
    print("Deployment architecture (single forward pass):")
    print("  ViT(720 tok) → layer-0 emb → MLP scorer(0.6M) → top-K → LLM(text + K tok) → trajectory")
    print()
    print("NOTE: 'Saving' assumes Variant B (true token drop). Current eval uses Variant A")
    print("(attention mask) for quality-faithful measurement; Variant B is pure engineering")
    print("(M-RoPE position reindex after drop).")

    # --- Component breakdown: attention (O(n²)) vs FFN+linear (O(n)) ---
    print("\n" + "-" * 90)
    print("Component Breakdown: Attention (O(n²)) vs FFN+Linear (O(n)) in LLM Prefill")
    print("-" * 90)
    print(f"{'Ratio':<8} {'Seq':<6} {'Attn(G)':<10} {'FFN+Lin(G)':<12} {'Attn%':<8} {'Attn_save':<12} {'FFN_save':<10}")
    print("-" * 90)

    breakdown = []
    full_attn = None
    full_ffn = None
    for r in ratios:
        n_kept = int(round(r * N_VISION))
        seq_len = N_TEXT + n_kept

        # Attention component (quadratic): 2 * n² * d * 2 (scores + values) per layer
        attn_per_layer = 2 * 2 * seq_len * seq_len * LLM_HIDDEN
        attn_total = attn_per_layer * LLM_LAYERS

        # Linear components: QKV proj + O proj + FFN
        d_kv = LLM_KV_HEADS * LLM_HEAD_DIM
        linear_per_layer = (
            2 * seq_len * LLM_HIDDEN * LLM_HIDDEN +    # Q
            2 * seq_len * LLM_HIDDEN * d_kv +           # K
            2 * seq_len * LLM_HIDDEN * d_kv +           # V
            2 * seq_len * LLM_HIDDEN * LLM_HIDDEN +    # O
            3 * 2 * seq_len * LLM_HIDDEN * LLM_FFN     # FFN (SwiGLU 3 linears)
        )
        ffn_linear_total = linear_per_layer * LLM_LAYERS

        if full_attn is None:
            full_attn = attn_total
            full_ffn = ffn_linear_total

        attn_pct = attn_total / (attn_total + ffn_linear_total) * 100
        attn_save = 1.0 - attn_total / full_attn if full_attn > 0 else 0
        ffn_save = 1.0 - ffn_linear_total / full_ffn if full_ffn > 0 else 0

        breakdown.append({
            "ratio": r,
            "seq_len": seq_len,
            "attention_gflops": attn_total / 1e9,
            "ffn_linear_gflops": ffn_linear_total / 1e9,
            "attention_pct_of_llm": f"{attn_pct:.1f}%",
            "attention_saving": f"{attn_save*100:.1f}%",
            "ffn_linear_saving": f"{ffn_save*100:.1f}%",
        })

        print(f"{r:<8.2f} {seq_len:<6} {attn_total/1e9:<10.1f} {ffn_linear_total/1e9:<12.1f} "
              f"{attn_pct:<8.1f}% {attn_save*100:<12.1f}% {ffn_save*100:<10.1f}%")

    print()
    print("Interpretation:")
    print(f"  • At r=1.0 (n=941): attention = {full_attn/1e9:.1f}G ({full_attn/(full_attn+full_ffn)*100:.1f}% of LLM)")
    print(f"  • FFN dominates at this sequence length → mixed saving (39.4%) closer to linear (38.3%) than quadratic (61.9%)")
    print(f"  • For longer contexts (8-cam, multi-frame), attention fraction grows → saving amplifies")

    # Save JSON with all intermediate quantities
    out = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/profiling")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "Qwen2.5-VL-3B (AutoVLA)",
        "constants": {
            "vit_layers": VIT_LAYERS, "vit_hidden": VIT_HIDDEN, "vit_ffn": VIT_FFN,
            "llm_layers": LLM_LAYERS, "llm_hidden": LLM_HIDDEN, "llm_ffn": LLM_FFN,
            "llm_heads": LLM_HEADS, "llm_kv_heads": LLM_KV_HEADS, "llm_head_dim": LLM_HEAD_DIM,
            "n_vision_tokens": N_VISION, "n_text_tokens": N_TEXT, "n_total_prompt": N_TOTAL,
        },
        "scorer": {"params": "0.6M", "flops_gflops": scorer / 1e9, "pct_of_total": f"{scorer/full_total*100:.4f}%"},
        "flops_table": results,
        "component_breakdown": breakdown,
        "deployment_note": (
            "Single-pass deployment: ViT(once) → MLP scorer(~ms) → top-K → LLM prefill. "
            "Scorer overhead <0.02% of total FLOPs. Current eval uses Variant A (attention mask, "
            "no sequence reduction) for quality-faithful ablation. Real FLOPs saving requires "
            "Variant B (physical token drop + M-RoPE reindex), which is straightforward engineering. "
            "Variant A and B are mathematically equivalent in output (masked tokens receive zero "
            "attention weight and no downstream token attends to them), differing only in realized compute."
        ),
        "forward_looking": (
            "Our saving is bounded by the FFN-dominated regime at 941 tokens. "
            "For longer-context VLAs (8-camera, multi-frame, >2k tokens), the quadratic attention "
            "term dominates and the saving amplifies beyond 40%."
        ),
    }
    (out / "flops_table.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out / 'flops_table.json'}")


if __name__ == "__main__":
    main()
