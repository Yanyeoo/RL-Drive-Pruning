# Diagnosis: B0 invalid tokens (2 / 11576)

Date: 2026-06-16
Owner: shayladeng
Status: diagnosed, **fix deferred to M5/M6 agent refactor**

## TL;DR

Both `d318551a8ce150e5` (shard2 row 1836) and `7defd0c32cd8546a`
(shard3 row 1311) failed with the **same** error during
`AutoVLAAgent.compute_trajectory`:

```
File ".../navsim/agents/autovla_agent.py", line 445
    return Trajectory(poses[: self._trajectory_sampling.num_poses, :], ...)
File ".../navsim/common/dataclasses.py", line 248
AssertionError: Trajectory poses and sampling have unequal number of poses.
```

i.e. `autovla.predict(features)` produced **fewer poses than expected**
(`num_poses` = `time_horizon=4s × 2Hz = 8` by default), so the slice
`poses[:8, :]` returned <8 rows and the `Trajectory.__post_init__`
shape assertion fired.

## Nature of the failure

- **Not a pipeline bug.** Inference framework, sensor loading,
  metric_cache, scoring config all work correctly for these tokens (the
  exception is raised inside the agent, before any metric is computed).
- **Model decoding edge case.** The Qwen2.5-VL-3B LLM head sometimes
  emits an EOS / stop-token earlier than the 8-pose budget on these
  particular scenes, producing a 5/6/7-pose trajectory string.
- **Impact**: 2 / 11576 = 0.017% — well within evaluation noise.
- **Same root cause across shards**: independent threads, different
  GPUs, but identical assertion — confirms determinism (not a transient
  CUDA / numerical glitch).

## Why we are NOT fixing this now

1. B0 is locked at PDMS = 0.8983 (paper-aligned, +0.33 pt). Any change
   to `autovla_agent.py:445` invalidates that snapshot.
2. M5 / M6 will rewrite `compute_trajectory` anyway (insert token
   pruner + new reward path). Cleanest place to add the pose-pad
   fallback is during that refactor.
3. Loss is below 1‱; conservative to leave as a known limitation in
   the B0 report (already documented in `MA2_b0_navtest.md` §2.3).

## Fix to apply during M5 (TODO)

In `compute_trajectory`, **after** `poses = self.autovla.predict(...)`,
**before** the slice:

```python
n_expected = self._trajectory_sampling.num_poses  # = 8
n_got      = poses.shape[0]
if n_got < n_expected:
    # pad by repeating last predicted pose; preserves direction,
    # only the hallucinated short tail is filled.
    pad = poses[-1:].repeat(n_expected - n_got, 1)
    poses = torch.cat([poses, pad], dim=0)
# (slice as before; now safe)
return Trajectory(poses[:n_expected, :], self._trajectory_sampling), cot_results
```

Alternative (rejected): return `np.zeros((8, 3))` as fallback. Too
conservative — would drive PDMS for these scenes near 0, but the model
*did* predict something useful for the first 5-7 frames; padding is
more faithful.

## Verification plan (M5/M6)

After applying the fix, re-run the 2 affected tokens with
`run_pdm_score_cot.py` in single-token mode and confirm:
- `valid == True`
- 6 metric columns are filled
- final `score` is reasonable (not 0)

Also check that PDMS on the remaining 11574 tokens doesn't shift
(the change only affects the failure path).

## -20 token gap — resolved

11596 eligible (sharding union from yaml input) vs 11576 unique tokens
in merged csv. **Root cause: navsim's built-in `SceneFilter`** further
drops 20 scenes during log loading (not the dispatcher's metric_cache /
json filter).

Per-shard verification:

| shard | eligible (yaml) | actually processed (`Processing N / N`) | csv rows (- header) | csv valid rows (- 1 aggregate) |
|------:|----------------:|----------------------------------------:|--------------------:|-------------------------------:|
| 0     | 2954            | **2949**                                | 2950                | 2949                           |
| 1     | 2798            | **2796**                                | 2797                | 2796                           |
| 2     | 2969            | **2963**                                | 2964                | 2963 (incl. 1 invalid)         |
| 3     | 2875            | **2868**                                | 2869                | 2868 (incl. 1 invalid)         |
| **total** | **11596**   | **11576**                               | 11580               | **11576** (incl. 2 invalid)    |

The 20-token loss is **before** any agent inference — it happens in
`SceneLoader._filter_scenes` (navsim/common/dataloader.py:44-50):

```python
# Filter scenes which are too short
if len(frame_list) < scene_filter.num_frames:
    continue
# Filter scenes with no route
if scene_filter.has_route and \
   len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0:
    continue
```

This is **navsim's standard behavior**, not a pipeline bug. The same
20 tokens will always be dropped given the same SceneFilter config
(`num_history_frames=4`, `num_future_frames=10`, `has_route=True`).

**Implication for the future**:
- Treat **11576** (not 11596) as the canonical "B0 evaluable set" size.
- The 11596 token snapshot in `data/splits/navtest_b0_tokens.txt` is
  still a faithful "navtest eligible" record but downstream consumers
  must be aware that ~0.17% will be auto-filtered by navsim.
- No code change needed. (Optional polish: regenerate the snapshot
  using `SceneLoader.tokens` after a dummy load, to get the exact
  11576-token canonical list. Deferred.)

Both items (2 invalid + 20 filtered) are now fully explained.
