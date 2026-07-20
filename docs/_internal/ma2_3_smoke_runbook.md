# MA2.3 GRPO smoke — failure-mode runbook

For each likely hang/crash, lists the symptom, the relevant source location, a 1-shot diagnostic command, and the lowest-risk fix.

Path conventions:
- `$ROOT` = `/apdcephfs/private_shayladeng/tokenrl_autoVLA`
- `$AVLA` = `$ROOT/code/third_party/AutoVLA`
- `$LOG`  = `$ROOT/logs/ma2_3_smoke.log`
- `$PY`   = `/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python`

---

## P1. Hydra/argparse cannot find the config

**Symptom**: `FileNotFoundError: ./config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml`

**Source**: `tools/run_rft.py` line ~78–90 (argparse + path join).

**Diagnose**:
```
ls $AVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml
```

**Fix**: ensure shell `cd $AVLA` before running and pass `--config training/qwen2.5-vl-3B-navtest-grpo-nocot` (no `.yaml`, no leading `./`).

---

## P2. SFT checkpoint load fails

**Symptom**: `KeyError: 'state_dict'` or `RuntimeError: Error(s) in loading state_dict for AutoVLA`.

**Source**: `models/autovla.py:47-49`
```
state_dict = torch.load(config['model']['sft_model_path'])["state_dict"]
state_dict = {k.replace("autovla.", "").replace("drivevla.", ""): v ...}
self.reference_model.load_state_dict(state_dict, strict=False)
```

**Pre-verified (this session)**:
- `AutoVLA_PDMS_89.ckpt` is a dict with top keys `['pytorch-lightning_version', 'state_dict']` ✓
- All 825 weight keys start with `autovla.` ✓
- `strict=False` tolerates extra keys ✓

**Diagnose** (if it still fails):
```
$PY -c "import torch; ck=torch.load('$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt', map_location='cpu', weights_only=False); print(list(ck.keys())); sd=ck['state_dict']; print(len(sd), list(sd.keys())[:3])"
```

**Fix**: if path is wrong, fix `model.sft_model_path` in yaml. If file is corrupt, redownload.

---

## P3. Dataset init blows up (RFTDataset)

**Symptom**: `FileNotFoundError`, `ValueError: empty dataset`, or hang during DataModule.

**Source**: `dataset_utils/rft_dataset.py` — scans `data_path/*.json`.

**Diagnose**:
```
find $ROOT/data/navtest_nocot -maxdepth 1 -name '*.json' | wc -l
# expect ≥ 4 for smoke
```

**Fix**: re-run MA2.1 or point yaml `data.train.json_dataset_path` at the dry-run residue.

Pre-verified: RFTDataset has no cot/nocot branching at the file-scan level (`cot_output: []` is just an empty list in nocot json).

---

## P4. Sensor video path resolves to non-existent jpg

**Symptom**: `FileNotFoundError: .../front_left_2.jpg` inside `AutoVLA.get_prompt` (called from `generate_sample`).

**Source**: `models/autovla.py:610-624` — `f"file://{front_right_camera_X}"`. Paths come from MA2.1 json (which encoded `${SENSOR_BLOBS_ROOT}/<log>/<camera>/<token>.jpg`).

**Diagnose**:
```
# pick one json
F=$(ls $ROOT/data/navtest_nocot/*.json | head -1)
$PY -c "
import json
d=json.load(open('$F'))
for k,v in d.get('input_features',{}).items():
  if 'camera' in k.lower() or 'path' in k.lower() or isinstance(v,str) and 'jpg' in v:
    print(k, v[:120])
"
```

Then `ls` one of the printed paths.

**Fix**: if dirs exist but jpgs missing, MA2.1 didn't decode that log. If dirs missing entirely, `sensor_data_path` in MA2.1 (when generating the json) had a typo — regenerate.

Pre-verified: the nested `.../test/openscene-v1.1/sensor_blobs/test/` path is real and populated.

---

## P5. KeyError on metric_cache (the high-risk one)

**Symptom**: `KeyError: '<token>'` inside `models/utils/score.py:51`, *not* caught.

**Source**: `score.py:44-52`
```
def rl_pdm_score(self, trajectory, token):
    metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]   # <- raises
    with lzma.open(metric_cache_path, "rb") as f:
        ...
```

Note: the `try/except` starts at line 55, *after* the indexing. Indexing failures bubble all the way up and crash `training_step`.

**Diagnose** (also done by smoke shell preflight now):
```
$PY - <<EOF
import sys
sys.path.insert(0, "$AVLA"); sys.path.insert(0, "$AVLA/navsim")
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader
loader = MetricCacheLoader(Path("$ROOT/data/navtest_metric_cache"))
cache = set(loader.metric_cache_paths.keys())
jsons = {p.stem for p in Path("$ROOT/data/navtest_nocot").glob("*.json")}
print("cache:", len(cache), "json:", len(jsons), "covered:", len(jsons & cache), "missing:", len(jsons - cache))
print("examples missing:", list(jsons - cache)[:5])
EOF
```

**Fix options (priority order)**:
1. Wait until MA2.2 finishes the full 11600 → coverage 100%.
2. Soft-link only the covered tokens into a smaller `navtest_nocot_smoke/` dir and point yaml there.
3. (Patch-style, last resort) Wrap the indexing in `.get(token)` + early return — touches upstream code.

---

## Pass criteria (manual log inspection)

Look for these strings in `$LOG`:

| Order | Match | Means |
|---|---|---|
| 1 | `Using online reference model from .../AutoVLA_PDMS_89.ckpt` | SFT ckpt loaded |
| 2 | `LoRA` / `trainable params` | LoRA wrapped successfully |
| 3 | `rank=1` / `world_size=2` (or FSDP banner) | Both GPUs alive |
| 4 | `train_reward` or `loss` printed | First training_step completed |
| 5 | exit code `124` from `timeout` | Cleanly killed by us, not by a crash |

If 1–4 all appear and exit is 124, **smoke passes**. Anything else → triage with P1–P5 above.
