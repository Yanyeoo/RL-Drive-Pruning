# `data/` — runtime mount point

This directory is **gitignored**. Datasets are not committed to the repository.

At runtime, this directory is expected to contain (or symlink to) the
NAVSIM dataset (`navtrain`, `navtest`) and any intermediate caches (e.g. `metric_cache`).

> `navhard_two_stage` is **not required** for the current pipeline.
> See `docs/plan/design_decisions.md` Revision 2026-06-15 for why navhard
> is deferred to future work.

Concrete paths are configured per-environment via YAML files under
`code/configs/`; nothing is hard-coded in source.
