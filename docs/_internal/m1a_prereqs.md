# M1.a runtime prerequisites

Date: 2026-06-17
Audience: future AI agents and humans launching M1.a runs.

## Hard requirements

1. **`attn_implementation='eager'` on the loaded Qwen2.5-VL.**
   `AutoVLAWithAttentionAgent.__init__` checks this and raises if not.
   The eager attention forward returns `attn_weights` (post-softmax,
   shape `(bsz, num_heads, q_len, k_len)`) when we monkey-patch to
   pass `output_attentions=True`. The `sdpa` and `flash_attention_2`
   paths never produce `attn_weights` — they call fused kernels —
   so they would silently capture nothing.

   AutoVLA defaults to eager (see
   `code/third_party/AutoVLA/models/autovla.py:491-510` for the H20
   SIGFPE rationale). The only way to break this is to set
   `training.attn_impl` to `sdpa` or `flash_attention_2` in the
   training yaml, or pass `+model.attn_impl=sdpa` as a hydra override.

2. **Single-scene batch (bsz=1).**
   `locate_prompt_landmarks` asserts `input_ids.shape[0] == 1`.
   `compute_trajectory` in the navsim agent is always per-scene so
   this is fine for inference. If you switch to batched generation
   (e.g. for throughput), batch-aware landmark tracking is required.

3. **autovla conda env.**
   `/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python`
   has the right transformers (4.49.0), peft, navsim and AutoVLA
   dependencies. The default `python3` does not.

4. **PYTHONPATH order.**
   `code/` (for `rldrive`) must come BEFORE `code/third_party/AutoVLA/navsim`
   and `code/third_party/AutoVLA` so the wrapper imports resolve. The
   driver script `scripts/run_m1a_attention_probe.sh` sets this correctly.

## Soft requirements (verified by code)

5. **GQA correctness.**
   Resolved upstream by `repeat_kv` (modeling_qwen2_5_vl.py:770) before
   the QK matmul. Our head-mean across `num_heads` is valid.

6. **Pre-fill is a single forward.**
   For `attn_implementation='eager'` with no past_key_value, generate()
   issues one full-prompt forward then 1-token decode steps.
   `AutoVLAWithAttentionAgent` asserts `captured_q_len == prompt_len`
   on every call — flips loud if the model ever chunks pre-fill.

## Verification gates before scaling M1.a beyond smoke

These TODO items, listed in `code/rldrive/scoring/attention_capture.py`,
MUST clear on a first probe scene before we trust any layer-probe
numbers:

- TODO(M1.a) #4 — `captured_q_len == prompt_len`
- TODO(M1.a) #5 — `processor.decode(input_ids[0, last_instr_idx])` is
  a content token, not an `<|im_end|>` chat marker
- TODO(M1.a) #6 — Path A vs Path C cross-check (a 2-line vlm.generate
  with `output_attentions=True, return_dict_in_generate=True` and
  manual slice — must match within fp32 tolerance)

## Cost ballpark

- Pre-fill: one forward, ~70-80 ms on H20 for the 3-cam-4-frame prompt
  (B0 inference was ~93 tok/s, end-to-end ~3 s/scene including decode).
- Storage: one `.pt` per scene, head-averaged is ~(N_vision × 4 B)
  ≈ 1 KB / scene. 100-scene probe = ~100 KB. Negligible.

## Failure modes to watch for

- `attn_weights` shape mismatch (e.g. some HF version returns the
  pre-softmax product). Mitigation: log shape on first scene, ensure
  it equals `(1, num_heads, q_len, q_len)` and values sum-to-1 per row.
- `vision_token_positions` empty (probe ran on a text-only scene).
  `attention_capture._save_attention` writes a `.MISSING.json` sentinel
  in that case; not a crash.
- Chat-template marker as last_instr_idx (V3 above). Fall back to
  the position immediately before the marker.
