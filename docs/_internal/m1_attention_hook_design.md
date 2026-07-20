# M1.a / M1.b — attention hook design notes

Date: 2026-06-16 (drafted while navtrain is downloading)
Owner: shayladeng
Status: design draft + first code sketch ✅ (2026-06-17 morning)
Code sketch: `code/rldrive/scoring/attention_capture.py`
              (NOT YET wired into autovla.predict; integration after .chain_complete)

## Context

- Q4.1 chose **L1 — attention-distill from AutoVLA's frozen LLM**.
- M1.a probes which LLM layer's attention best correlates with PDMS.
- M1.b dumps that layer's attention over the full navtrain `\\` probe.

Both need to extract a `(N_vision_tokens,)` importance vector per scene
from a Qwen2.5-VL-3B forward pass.

## What we actually need

```
attention[layer L*, head averaged, query = last_instruction_token, key = vision_tokens]
  -> shape (N_vision_tokens,)   --> ranking label
```

Per Q4.1.b: query = last instruction text token (default; action first
token is an ablation). Per Q4.1.c: only ranking matters.

## Two extraction paths

### Path A — HuggingFace generate-time attentions (preferred)

`Qwen2_5_VLForConditionalGeneration.generate(..., output_attentions=True, return_dict_in_generate=True)`
returns `outputs.attentions`:

- Shape (per generate step):
  `tuple of length num_layers`, each element
  `Tensor[batch, num_heads, query_len, key_len]`
- **Pre-fill step (step 0)** has `query_len = full_prompt_len`,
  `key_len = full_prompt_len`. **This is the only step we need** — we
  read `attentions[L*][0, :, last_instr_idx, vision_start:vision_end]`
  and average over heads.
- All later decode steps have `query_len = 1`. We don't need them.

Pro: standard API, no monkey patching. Con: stores all-layer pre-fill
attention in memory = `num_layers × num_heads × prompt_len^2 × 2 B`
≈ `28 × 8 × 4000^2 × 2` ≈ 7 GB per scene. **Too much.**

### Path B — Targeted forward hook on layer L*

Register a `pre_forward` hook on `self.vlm.model.layers[L*].self_attn`
that records `attn_weights` from the kwargs, but **only for the first
forward call (pre-fill)**, then auto-removes itself.

Memory cost: 1 layer × `num_heads × prompt_len^2 × 2 B`
≈ `8 × 4000^2 × 2` ≈ 256 MB per scene. Acceptable but still wasteful.

### Path C — Patch attention computation to compute and store only the row we need

Wrap `self_attn.forward` so that during the pre-fill step it computes
the full QKᵀ matmul as usual but **only stores
`softmax_row[last_instr_idx, vision_start:vision_end]`** (a vector of
shape `(N_vision,)`). Memory cost: `N_vision × 4 B` ≈ 1 KB per scene.

**Decision: Path C** — minimal memory, no extra forward, allows the
attention scoring to be added to the live inference pass and merged
with M0.4. The implementation footprint is small (~50 LOC monkey
patch in a context manager).

## Token-index bookkeeping

Need to know:
1. `vision_start, vision_end` — the contiguous range of vision tokens
   in the LLM input sequence. Qwen2.5-VL inserts `<|vision_start|>`
   and `<|vision_end|>` special tokens around them. Walk
   `input_ids[0]` to find these markers (one per video, 3 cameras x 4
   frames = 12 image-grids).
2. `last_instr_idx` — index of the last text token before the action
   token starts. Find via `input_ids[0] == action_start_id` first
   occurrence minus 1; if no action token in prompt (it's generated),
   use `len(prompt) - 1`.

These are computed once from the prompt by `get_prompt` and passed to
the hook in the same forward pass.

## API sketch

```python
@contextmanager
def attention_capture(llm, layer_idx, vision_range, query_idx):
    """Yields a dict that is filled with attention[query_idx, vision_range]
    at the next forward call. Auto-restores."""
    bucket = {}
    layer = llm.model.layers[layer_idx].self_attn
    orig = layer.forward

    def patched(self, hidden_states, *args, **kwargs):
        out = orig(hidden_states, *args, **kwargs)
        # out = (attn_output, attn_weights, past_key_value) when
        # output_attentions=True; pull attn_weights when present.
        # Detailed indexing TBD when actually wired up — Qwen2.5-VL
        # may have grouped-query attention and a custom return shape.
        ...
        return out

    layer.forward = MethodType(patched, layer)
    try:
        yield bucket
    finally:
        layer.forward = orig
```

Worked example to validate first: take 1 probe-set scene, run with
`generate(output_attentions=True)` (Path A) on a tiny prompt, hand-pick
`vision_start/end, last_instr_idx`, save the slice, compare with
Path C output → must match within fp16 tolerance.

## Open issues / unknowns

1. ~~Does `Qwen2_5_VLForConditionalGeneration` use grouped-query
   attention? If yes, `num_heads_for_attention ≠ num_heads`, need to
   broadcast. Verify by `model.config`.~~ ✅ **RESOLVED 2026-06-17**:
   transformers 4.49.0 `Qwen2_5_VLAttention.forward` calls
   `repeat_kv(...)` BEFORE the matmul producing `attn_weights`
   (modeling_qwen2_5_vl.py:770), so attn_weights shape is
   `(bsz, num_heads, q_len, k_len)` with num_heads = full heads.
   Head-averaging across this axis is correct as-is.
2. Is the pre-fill step a single forward, or chunked? On H20 with 3-cam
   x 4-frame inputs it should be a single forward but worth confirming
   with a small prompt.  ⏳ deferred to first probe-scene smoke (TODO M1.a #4
   in attention_capture.py).
3. Q4.1.b says "last instruction token". The prompt format may end
   with `<|im_end|>` style tokens — the *content* last token is what
   we want, not the chat template end-marker. Verify by decoding
   `input_ids[0, last_instr_idx]` and inspecting.  ⏳ TODO M1.a #5.

Plus added during 2026-06-17 sketch:

4. The eager attention forward already returns `attn_weights` natively
   when `output_attentions=True` (line 800-803). Our `patch_attention_capture`
   force-enables it for the pre-fill call, takes the row we need, then
   nulls it out so HF generate doesn't accumulate across decode steps.
   This is strictly simpler than the original "patch softmax row" plan.

## When this gets implemented

After:
- navtrain download done ✅ (in progress 2026-06-17 morning, ETA noon)
- M0.2 splits produced ✅ (chain watcher armed)
- check_navtrain_sanity passes ✅

Then in M1.a iteration 0:
- code sketch already exists at `code/rldrive/scoring/attention_capture.py`
- write the runner `code/rldrive/scoring/run_attention_probe.py` (TODO M1.a #3)
- Path A vs Path C cross-check on 1 probe scene (TODO M1.a #6)
- decode `input_ids[0, last_instr_idx]` sanity (TODO M1.a #5)
- assert `captured_q_len == input_ids.shape[1]` (TODO M1.a #4)
- THEN run on probe set A (100 scenes) for layer L=14 (middle of 28 layers
  as a default starting probe). Iterate on token index bookkeeping until
  the saved attention row makes physical sense (front-vehicle tokens
  score high in a "go straight" scene, etc).
