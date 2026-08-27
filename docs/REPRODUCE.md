# REPRODUCE — end-to-end reproduction guide

Reproducing the headline number (`v11_hard50_kl` = **0.898719** PDMS on navtest,
N=11576) from a bare machine.

**Read this first — three things will otherwise waste your time:**

1. **Scripts under `scripts/` hard-code the original node's absolute paths**
   (`/apdcephfs/private_shayladeng/...`). They were written for one machine and
   were never parameterised. `env.example.sh` gives you the variables; where a
   script still hard-codes a path you must edit its header (each one has the
   paths in the first ~15 lines).
2. **You cannot reuse our `data/navtest_nocot/*.json`.** Those files embed
   **absolute** image paths (`front_camera_paths: /apdcephfs/.../CAM_F0/xxx.jpg`).
   They are not shipped, and copying them would not work anyway — you must
   regenerate them (§4) so the paths point into *your* sensor blobs.
3. **`EVAL_WORKERS=1` is mandatory.** With >1 worker the navsim thread pool races
   while loading Qwen and silently drops scenes, inflating scores.

Hardware we used: 4× H20 (96GB). Section 7 has time estimates.

---

## 0. What you get vs what you must build

| | shipped in this repo | you must produce |
|---|---|---|
| pruning scorer code | ✅ `code/rldrive/` | — |
| trained scorer weights | ✅ `release_ckpt/` (58MB, 10 ckpts) | — |
| our AutoVLA-side edits | ✅ `autovla_overlay/` (patch + configs) | — |
| AutoVLA source | ❌ academic license | clone upstream (§3) |
| VLA backbone (16GB) | ❌ | download (§3) |
| NAVSIM sensor data (~116GB) | ❌ | download (§2) |
| per-scene json | ❌ absolute paths inside | regenerate (§4) |
| metric cache | ❌ | build (§4) |

If you only want to **evaluate our released scorer**, do §1-§5 and skip §6.

---

## 1. Environment

```bash
conda create -n autovla python=3.9.23 -y
conda activate autovla

# torch first, from the cu121 index (the +cu121 tag is not on PyPI)
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`nuplan-devkit==1.2.0` and `navsim==1.1.0` are not installable from PyPI at these
versions — install them in §3 after cloning AutoVLA.

**Verify**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# expect: 2.4.0+cu121 True <n_gpus>
```

`numpy` must stay at `1.23.4`; nuplan-devkit 1.2.0 still uses the removed
`np.float` / `np.bool` aliases and crashes on numpy ≥1.24.

---

## 2. Data

Get NAVSIM v2 / OpenScene (~116GB sensor blobs) and the nuPlan maps (~1.6GB)
from the [NAVSIM repo](https://github.com/autonomousvision/navsim) instructions.
Target layout:

```
$OPENSCENE_DATA_ROOT/
├── navsim_logs/{test,trainval}/
└── sensor_blobs/{test,trainval}/
$NUPLAN_MAPS_ROOT/          # nuplan-maps-v1.0
```

Then:

```bash
cp env.example.sh env.sh
$EDITOR env.sh      # set OPENSCENE_DATA_ROOT / NUPLAN_MAPS_ROOT / RLDRIVE_PY
source env.sh
```

**Verify** — the warnings printed by `env.sh` must all be gone except the
AutoVLA-overlay one (fixed in §3).

---

## 3. AutoVLA + our overlay

AutoVLA is under an academic license (© 2026 UCLA Mobility Lab), so we do not
redistribute it. Clone the exact commit we built on and apply our overlay:

```bash
cd code/third_party
git clone https://github.com/ucla-mobility/AutoVLA.git
cd AutoVLA
git checkout ba34eed74ce6729e7986592d0e66cbaca397b4fa

git apply ../../../autovla_overlay/patches/autovla_ba34eed_rldrive.patch
cp -a ../../../autovla_overlay/configs/. .

pip install -e navsim            # provides navsim==1.1.0
cd ../../..
```

You also need the frozen backbone `AutoVLA_PDMS_89.ckpt` (16GB) from the AutoVLA
release, placed at `$AUTOVLA_CKPT`, and `Qwen2.5-VL-3B-Instruct` (processor +
base weights) wherever the config's `pretrained_model_path` points.

Edit the paths inside the overlay configs to match your machine:

```bash
$EDITOR code/third_party/AutoVLA/config/dataset/qwen2.5-vl-3B-navtest.yaml
$EDITOR code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml
# pretrained_model_path / navsim_log_path / sensor_blobs_path
```

**Verify** — the reward patch is the single most important piece:

```bash
source env.sh
python -c "
from models.utils.score import PDM_Reward
assert hasattr(PDM_Reward, 'rl_pdm_score'), 'reward patch NOT applied'
print('reward patch OK')"

ls code/third_party/AutoVLA/navsim/navsim/planning/script/config/common/train_test_split/navtest_local_filtered_shard0_20260616_154858.yaml
```

Both must succeed. `rl_pdm_score` is the reward every RL cycle optimises; the
shard yaml defines the 4-shard protocol our numbers are measured on.

---

## 4. Per-scene json + metric cache

### 4a. json (embeds *your* absolute image paths)

```bash
cd code/third_party/AutoVLA          # the script does f"./config/{args.config}.yaml"
python tools/preprocessing/nocot_sample_generation.py \
    --config dataset/qwen2.5-vl-3B-navtest \
    --output_dir "$RLDRIVE_ROOT/data/navtest_nocot" \
    --num_workers 16
```

Re-run with `--pre_generated_dir <output_dir>` to resume; it skips finished
tokens. Repeat with `dataset/qwen2.5-vl-3B-navtrain_full` →
`data/navtrain_nocot` if you intend to train (§6).

**Verify** the embedded paths resolve on your box:

```bash
python - <<'PY'
import glob, json, os
f = sorted(glob.glob("data/navtest_nocot/*.json"))[0]
p = json.load(open(f))["front_camera_paths"]
p = p[0] if isinstance(p, list) else p
print(p)
assert os.path.exists(p), "image path in json does not resolve — check sensor_blobs_path"
print("json paths OK")
PY
```

### 4b. metric cache

```bash
cd "$NAVSIM_DEVKIT_ROOT"
python navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path="$RLDRIVE_ROOT/data/navtest_metric_cache" \
    worker=single_machine_thread_pool \
    worker.max_workers=16 \
    worker.use_process_pool=true
```

The knob is `cache.cache_path` — `metric_cache_path` in the config is only a
reference to it, so setting that instead silently caches to the default location.

For training you also need the navtrain cache; we used the
`navtrain_avail19k` split (the 19k scenes actually present in our local mirror):

```bash
python navsim/planning/script/run_metric_caching.py \
    train_test_split=navtrain_avail19k \
    cache.cache_path="$RLDRIVE_ROOT/data/navtrain_metric_cache" \
    worker=single_machine_thread_pool worker.max_workers=16 worker.use_process_pool=true
```

**Verify**

```bash
find data/navtest_metric_cache -name 'metric_cache.pkl' | wc -l
```

Ours reports **11596** for navtest (and 19231 for navtrain). A number far below
that means caching died partway — rerun, it is resumable.

---

## 5. Evaluate the released scorer

`scripts/eval_v7_folds_4gpu.sh` expects checkpoints under
`ckpt/v7_surrogate_<CYCLE_ID>/<arm>/`, so point it at a released checkpoint:

```bash
mkdir -p ckpt/v7_surrogate_repro
cp -a release_ckpt/v11_hard50_kl ckpt/v7_surrogate_repro/hard50_kl

# edit the header paths (ROOT / PY / SENSOR / CKPT / YAML) to match your machine
$EDITOR scripts/eval_v7_folds_4gpu.sh

# 2-shard gate first (~3.7h on 4 GPUs) — cheap smoke test of the whole chain
EVAL_WORKERS=1 EVAL_GPUS="0 1 2 3" \
  bash scripts/eval_v7_folds_4gpu.sh repro gate hard50_kl

# full 4-shard, the number we report (~3.7h on 4 GPUs)
EVAL_WORKERS=1 EVAL_GPUS="0 1 2 3" \
  bash scripts/eval_v7_folds_4gpu.sh repro full hard50_kl
```

Note `SENSOR` / `+agent.sensor_data_path` in that script points at a
`data/navsim_v2_local` directory that **does not exist even on our node**. It is
harmless: image paths come from the json (§4a), and `sensor_data_path` is only
stashed into the feature dict, never used to build a path. Leave it or set it to
your sensor root.

**Verify**

```bash
python - <<'PY'
import glob, pandas as pd
fs = sorted(glob.glob("results/raw/v7_surrogate_repro_full/*.csv"))
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
df = df[df.token != "average"].drop_duplicates("token")
print(f"N={len(df)}  PDMS={df.score.mean():.6f}")
PY
```

Expected: `N=11576`, `PDMS≈0.898719`. Per shard we measured
`0.898818 / 0.900555 / 0.893456 / 0.902266`.

Exact bit-level agreement is unlikely — the VLA decodes with `temperature=0.01`
(not 0) and cuBLAS kernels are not deterministic. Judge against the no-prune
baseline you compute yourself, not against our decimals: our reference is
no-prune = 0.898845, i.e. this checkpoint sits **0.000126 below** it. See
`release_ckpt/README.md` for the honest framing of that gap.

Baseline for comparison (same script, unpruned):

```bash
# +agent.keep_ratio=1.0 +agent.selector=attn_L12  → the 0.898845 upper bound
```

---

## 6. Retrain (optional)

Full RL pipeline, in dependency order:

```bash
# 6a. dump vision features on navtrain (input for the supervised scorer)
bash scripts/run_s3_feature_dump_4gpu.sh
#     -> data/s3_scorer/features/

# 6b. supervised scorer = the RL init (this is release_ckpt/sft_scorer_base)
#     Both scripts default every path; they need attention labels in
#     --label-dir (per-layer attention dumps produced by the earlier
#     m1b2 attention-extraction stage, layer 12 is the one we use).
python scripts/s3_build_labels_train_scorer.py \
    --feat-dir data/s3_scorer/features \
    --label-dir exp/m1b2_navtrain_full_alllayers \
    --out-dir ckpt/s3_token_scorer \
    --label-layer 12
#     -> ckpt/s3_token_scorer/  (ranking loss; this is what RL warm-starts from)

# 6c. v10: value baseline + efficiency floor  -> full_hi
bash scripts/run_v10_methodA_4gpu.sh

# 6d. v11: hard-example mining (warm-starts from a v10 checkpoint)
bash scripts/run_v11_mining_4gpu.sh 8192            # ~2.6h on 4 GPUs
python scripts/build_v11_scene_list.py \
    --mine-glob 'results/mining_*/mine_shard*.jsonl' \
    --out results/mining_repro/scenes_hard50.txt \
    --n-total 512 --hard-frac 0.5 --hard-repeat 2
bash scripts/run_v11_hardmine_4gpu.sh <list50> <list75> repro
```

`run_v11_mining_4gpu.sh` and `run_v11_hardmine_4gpu.sh` both hard-code
`INIT_CKPT=ckpt/v7_surrogate_20260825_v10r1/full_hi`. If you skip §6c, point
them at `release_ckpt/v10_full_hi` instead.

Or run mining → train → gate → pick winner → full unattended:

```bash
bash scripts/orchestrate_v11_pipeline.sh
```

**Sanity check on the mining step** — this is where the method's premise is
either confirmed or refuted on your data. We measured, over 8190 navtrain scenes:

```
catastrophic (delta <= -0.3):  200 scenes (2.44%)  sum delta -150.07
exactly zero               : 5852 scenes (71.45%)
positive (beats no-prune)  : 1210 scenes (14.77%)
=> catastrophic scenes carry 91.7% of all negative delta
```

If your mining output does not show a similar concentration, stop and
investigate before training — the whole approach rests on it.

**Do not select checkpoints on training reward.** We verified twice that it
misleads: the `hard75` arm had the best training reward (-0.29) and the worst
gate score; `hard50_kl` barely moved its keep_ratio (0.691→0.686) and won the
full set. Use gate, then confirm on full.

---

## 7. Time and cost (4× H20)

| step | wall time | note |
|---|---|---|
| §4a json (navtest) | ~1h | CPU, 16 workers |
| §4a json (navtrain) | ~3h | 19k scenes |
| §4b metric cache | ~2h | CPU-heavy |
| §5 gate (2 shards) | ~3.7h | 4 jobs, 1/GPU |
| §5 full (4 shards) | ~3.7h | 4 jobs, 1/GPU |
| §6b supervised scorer | ~1h | after features dumped |
| §6c v10 training | ~1.5h/arm | 4 arms in parallel |
| §6d v11 mining | ~2.6h | 8192 scenes, 4.5s/scene |
| §6d v11 training | ~1.9h/arm | 4 arms in parallel |

Evaluation dominates. Budget ~8h for evaluate-only (§1-§5, excluding downloads),
~24h for the full retrain.

---

## 8. Known issues

- **`EVAL_WORKERS>1` silently corrupts results** on H20: concurrent Qwen loading
  hits a meta-tensor race and scenes get dropped. Always `EVAL_WORKERS=1`.
- **`NVIDIA_TF32_OVERRIDE=0` is required** (set by `env.sh`): on sm_90 with
  torch 2.4+cu121, fp32 cuBLAS GEMM can raise SIGFPE without it.
- **`scripts/*.sh` are not path-portable.** Each has its paths in the first ~15
  lines; edit them. `env.example.sh` documents what each variable means.
- **Latency is not verified.** `scripts/profile_wallclock.py` still fails during
  trajectory decoding, so the efficiency claim rests on `keep_ratio` plus the
  theoretical FLOPs table (`scripts/compute_flops_table.py`): at r=0.72,
  -19.0% total FLOPs / -22.3% LLM prefill. Treat wall-clock speedup as unproven.
- **Pruning is attention-mask based** (variant A), so the sequence length is
  unchanged and FLOPs savings are theoretical rather than realised. A true
  token-drop variant needs M-RoPE position recomputation.
- **`data/navsim_v2_local` does not exist** and is referenced by several scripts.
  Harmless (see §5).

---

## 9. If a step fails

| symptom | cause |
|---|---|
| `AttributeError: module 'numpy' has no attribute 'float'` | numpy ≥1.24; pin `1.23.4` |
| `MissingConfigException: Cannot find primary config` | ran a script from the wrong cwd — navsim scripts must run from `$NAVSIM_DEVKIT_ROOT` |
| `FileNotFoundError: .../CAM_F0/xxx.jpg` | json holds foreign absolute paths; regenerate (§4a) |
| `hasattr(PDM_Reward,'rl_pdm_score')` is False | overlay patch not applied (§3) |
| scene count < 11576 in full eval | `EVAL_WORKERS>1`, or metric cache incomplete |
| `TypeError: 'Scene' object is not subscriptable` | passed a navsim `Scene` where the agent wants the json dict |
| SIGFPE / crash in fp32 GEMM | `NVIDIA_TF32_OVERRIDE=0` not exported |
