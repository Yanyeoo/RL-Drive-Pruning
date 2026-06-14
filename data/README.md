# `data/` — runtime mount point

This directory is **gitignored**. Datasets are not committed to the repository.

At runtime, this directory is expected to contain (or symlink to) the
NAVSIM dataset (`navtrain`, `navtest`, `navhard_two_stage`) and any
intermediate caches (e.g. `metric_cache`).

Concrete paths are configured per-environment via YAML files under
`code/configs/`; nothing is hard-coded in source.
