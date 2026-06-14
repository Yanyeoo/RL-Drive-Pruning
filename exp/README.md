# `exp/` — experiment outputs

This directory is **gitignored**. Experiment artifacts (checkpoints,
intermediate predictions, oracle search results, etc.) are written here
at runtime and are not committed.

Naming convention (recommended):
```
exp/<MILESTONE>_<short-tag>_<YYYYMMDD>/
  ├── ckpt/
  ├── config.yaml
  ├── stdout.log
  └── metrics.json
```
