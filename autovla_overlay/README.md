# AutoVLA overlay — what you need to reproduce our experiments

Our code depends on [AutoVLA](https://github.com/ucla-mobility/AutoVLA) at commit
`ba34eed` ("checkpoint release"), plus a set of local additions that live *inside*
that checkout. Because AutoVLA is a separate repository nested under
`code/third_party/AutoVLA/`, git only records a pointer to its commit — the files
we added or edited there are **not** captured by this repository.

This directory preserves exactly those changes so the experiments stay
reproducible.

AutoVLA is released under an **academic software license (© 2026 UCLA Mobility
Lab)**, so we do not redistribute its source here. You clone it from upstream and
apply our overlay on top.

## Contents

| path | what it is |
|---|---|
| `configs/` | 60+ config files we authored (dataset / training / navtest split definitions). Mirrors the AutoVLA directory layout, so it can be copied in directly. |
| `patches/autovla_ba34eed_rldrive.patch` | Our edits to 6 upstream files (+157 / -34 lines). |

## Why these files matter

Without them the pipeline cannot run:

- **`models/utils/score.py`** (patched, +77 lines) adds `PDM_Reward.rl_pdm_score` —
  the reward function every RL cycle in this project trains against.
- **`config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml`** is referenced by all
  training and evaluation scripts (`agent.config_path`).
- **`.../train_test_split/navtest_local_filtered_shard{0..3}_20260616_154858.yaml`**
  define the 4-shard navtest protocol. Every full-set number we report
  (e.g. `v11_hard50_kl` = 0.898719 over N=11576) is defined by these splits;
  without them the numbers are not comparable.
- `models/autovla.py`, `navsim/.../autovla_agent.py`, `tools/run_rft.py` and the
  preprocessing scripts carry smaller fixes needed for our data layout.

## How to apply

```bash
# 1. clone upstream at the exact commit we built on
cd code/third_party
git clone https://github.com/ucla-mobility/AutoVLA.git
cd AutoVLA
git checkout ba34eed74ce6729e7986592d0e66cbaca397b4fa

# 2. apply our source edits
git apply ../../../autovla_overlay/patches/autovla_ba34eed_rldrive.patch

# 3. drop in our config files
cp -a ../../../autovla_overlay/configs/. .

# 4. sanity check: these must exist afterwards
ls config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml
ls navsim/navsim/planning/script/config/common/train_test_split/navtest_local_filtered_shard0_20260616_154858.yaml
python -c "import sys; sys.path[:0]=['.','navsim']; from models.utils.score import PDM_Reward; print(hasattr(PDM_Reward,'rl_pdm_score'))"
```

The last command must print `True` — that confirms the reward patch landed.

## Note on the split definitions

The `navtest_local_filtered_shard*` splits enumerate the scene tokens that were
actually available in our local sensor-blob mirror, which is a subset of full
navtest (N=11576 across 4 shards). They are dated (`20260616_154858`) because
they were generated once and then frozen so that every cycle in this project is
measured on an identical scene set. Do not regenerate them if you want to compare
against our reported numbers.
