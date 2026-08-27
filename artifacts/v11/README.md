# v11 artifacts

Raw outputs of the 2026-08-26/27 cycle, kept in-repo because the training node
was reclaimed and they are not cheaply regenerable.

## `mining/` — per-scene pruning deltas on navtrain

8190 scenes rolled out with the deterministic `v10_full_hi` policy (2.6 GPU-hours
on 4× H20). One JSON per line:

```json
{"token": "...", "pruned_pdms": 0.879, "noprune_pdms": 0.879, "delta": 0.0,
 "keep_ratio": 0.707, "N": 720, "B": 509,
 "pruned_sub": {...}, "noprune_sub": {...}}
```

`delta = pruned_pdms - noprune_pdms` under the policy's own keep_ratio, so it
measures what pruning actually costs on that scene.

Distribution (this is the empirical basis of the whole v11 method):

| class | scenes | share | sum delta |
|---|---|---|---|
| catastrophic (`delta ≤ -0.3`) | 200 | 2.44% | **-150.07** |
| mild negative | 928 | 11.33% | -13.50 |
| exactly zero | 5852 | 71.45% | 0 |
| positive (beats no-prune) | 1210 | 14.77% | +178.32 |

**2.44% of scenes carry 91.7% of all negative delta**, and 71.45% of scenes lose
nothing at all from pruning ~28% of their vision tokens.

Reusable directly — the next cycle can rebuild scene lists without re-mining:

```bash
python scripts/build_v11_scene_list.py \
    --mine-glob 'artifacts/v11/mining/mine_shard*.jsonl' \
    --out <out.txt> --n-total 512 --hard-frac 0.5 --hard-repeat 2
```

`scenes_hard50.txt` / `scenes_hard75.txt` are the exact lists the reported arms
were trained on.

## `eval_full/`, `eval_gate/` — per-scene navtest PDMS

The CSVs every reported number is averaged from. `token`, `valid`, `score` plus
PDM sub-metrics per scene.

- `eval_full/` — 4 shards, N=11576, for `hard50` and `hard50_kl`
- `eval_gate/` — shards 0+1 for all four arms

Reproduce the headline number:

```bash
python - <<'PY'
import glob, pandas as pd
fs = glob.glob("artifacts/v11/eval_full/v7_full_hard50_kl_sh*.csv")
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
df = df[df.token != "average"].drop_duplicates("token")
print(f"N={len(df)} PDMS={df.score.mean():.6f}")   # 11576, 0.898719
PY
```

Keeping these also makes the per-scene attribution in `STATUS.md` auditable —
the claim that `hard50_kl` wins by *losing less* (606 vs 550 scenes beating
no-prune, at effectively identical catastrophic counts 83 vs 82) can be
recomputed from `eval_full/` against a no-prune baseline.
