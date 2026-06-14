# `logs/` — training / evaluation logs

This directory is **gitignored**. All `*.log`, `slurm-*.out`,
TensorBoard / W&B caches go here.

For long-running jobs, prefer `tee`-ing into:
```
logs/<MILESTONE>_<YYYYMMDD_HHMM>.log
```
