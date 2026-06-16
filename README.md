# RL-Drive

> **Driving-context-conditioned vision token pruning for VLA-based autonomous driving, learned via RL.**
>
> Status: 🚧 under active development — design freeze completed 2026-06-14, implementation in progress.

---

## TL;DR

Vision-Language-Action (VLA) models for autonomous driving spend the **vast majority of their FLOPs on vision tokens**, yet most of those tokens carry redundant information for the driving decision at hand. RL-Drive learns a **scene-adaptive token pruning policy** on top of a frozen VLA backbone, jointly optimizing **how many** tokens to keep (budget) and **which** tokens to keep (importance), trained with **GRPO** against a closed-loop driving metric.

**Key claims** (to be validated experimentally):
1. **≥ 50% vision-token saving** with negligible EPDMS drop on NAVSIM.
2. **Adaptive budget > fixed ratio** at matched FLOPs.
3. **Driving-context-conditioned scorer ≫ raw vision-attention** under the same pruning position.

---

## Method overview

```
                   ┌─────────────────────────────────────┐
                   │ VLA backbone (frozen)               │
                   │  ViT  ──►  [vision tokens]  ──► LLM │
                   └────────────────────┬────────────────┘
                                        │ pruning happens HERE
                                        │ (ViT–LLM interface, "external")
            ┌───────────────────────────┴───────────────────────────┐
            │                                                       │
   ┌────────▼─────────┐                                  ┌──────────▼─────────┐
   │ Importance       │  ← Stage A                       │ Budget Policy      │  ← Stage B
   │  Scorer (LLM-    │   SFT (attention distill,        │  (4-class:         │   GRPO with R3
   │  attention       │   LambdaRank) + GRPO             │  {0.25/0.5/0.75/1})│   piecewise Pareto
   │  distilled)      │                                  │                    │   reward, oracle
   └──────────────────┘                                  └────────────────────┘   from B3 label
```

Two stages, trained sequentially, both on top of a frozen VLA:

| Stage | Module | Objective | Label / Reward |
|-------|--------|-----------|----------------|
| **A** | Importance Scorer | rank vision tokens by driving-relevance | LLM-attention distillation (LambdaRank), then GRPO with EPDMS-advantage |
| **B** | Budget Policy | pick keep-ratio per scene from {0.25, 0.5, 0.75, 1.0} | B3 Pareto-aware oracle (SFT) + R3 piecewise Pareto reward (GRPO) |

Full design rationale and ablations: see [`docs/plan/design_decisions.md`](docs/plan/design_decisions.md) (Q1–Q5).

---

## Repository layout

```
RL-Drive/
├── README.md                       ← you are here
├── LICENSE                         ← All Rights Reserved (placeholder)
├── .gitignore
├── code/                           ← all source code
│   ├── rldrive/                    ← Importance Scorer + Budget Policy + reward
│   ├── scripts/                    ← train / infer / eval drivers
│   └── configs/                    ← YAML hyperparameters
├── docs/
│   ├── plan/
│   │   ├── design_decisions.md     ← Q1–Q5 design freeze (frozen 2026-06-14)
│   │   ├── implementation_plan.md  ← M0–M6 milestones, compute budget, deliverables
│   │   ├── milestones.md           ← (DEPRECATED — early draft, kept for history)
│   │   └── ...
│   ├── results/                    ← 📊 single source of truth for headline numbers (key_results.md)
│   ├── journal/                    ← daily journals
│   └── reports/                    ← per-milestone reports
├── data/                           ← (gitignored) datasets, mounted at runtime
├── exp/                            ← (gitignored) experiment outputs
├── logs/                           ← (gitignored) training logs
└── results/                        ← (gitignored) eval scores, plots, tables
```

> The `data/`, `exp/`, `logs/`, `results/` directories are **not tracked** by git; they exist on the workspace and are populated at runtime. Only `.gitkeep` and a short `README.md` per directory are committed, so the structure remains self-documenting.

---

## Quickstart

> ⚠️ Code is being uploaded incrementally as M0–M6 progress. This section will be filled in once Stage-A SFT lands (M2).

```bash
# 1. clone
git clone <repo-url>
cd RL-Drive

# 2. install (placeholder; see code/configs/env.yml when published)
conda create -n rldrive python=3.10 -y
conda activate rldrive
pip install -r code/requirements.txt   # TBD

# 3. point to your local NAVSIM / VLA backbone
# (paths configured in code/configs/*.yml; not hard-coded)

# 4. run baseline (M0)
bash code/scripts/eval_navtest_vanilla.sh   # TBD
```

---

## Roadmap (M0–M6)

| # | Milestone | Output | Status |
|---|-----------|--------|--------|
| **M0** | VLA baseline replication on NAVSIM (navtest) | EPDMS upper bound, FLOPs/latency reference | ⏳ |
| **M1** | Layer probing + attention extraction for distillation labels | per-layer ranking labels for navtrain \ A | ⏳ |
| **M2** | Stage A — Importance Scorer SFT (LambdaRank on attention) | scorer-v1 checkpoint | ⏳ |
| **M3** | B3 Pareto-aware oracle: per-scene optimal keep-ratio | budget labels for navtrain \ A | ⏳ |
| **M4** | Stage B — Budget Policy SFT (4-class) | budget-v1 checkpoint | ⏳ |
| **M5** | Stage A + Stage B joint GRPO (R3 reward, per-scene baseline) | RL-Drive-full checkpoint | ⏳ |
| **M6** | Evaluation: main table + Pareto curve + A1–A10 ablations | paper-ready figures/tables | ⏳ |

Full plan with compute budget, deliverable paths, and risk gates: [`docs/plan/implementation_plan.md`](docs/plan/implementation_plan.md).

---

## Reproducibility & evaluation

- **Backbone**: AutoVLA (Qwen2.5-VL-3B family) — frozen during all RL-Drive training.
- **Benchmark**: NAVSIM (open-loop EPDMS).
  - Main results: **navtest** (community-standard split, fair vs FastV / ToMe / AutoVLA).
  - **navhard** evaluation: deferred to future work — AutoVLA's upstream navsim fork does not natively support the navhard_two_stage two-path / reactive-synthetic protocol; revisiting it would require porting prior-work's navsim-v2 SceneLoader. See `docs/plan/design_decisions.md` Revision 2026-06-15 for full rationale.
- **Metric battery** (per Q5.a/b):
  - Quality: EPDMS, collision rate, comfort, progress.
  - Efficiency: avg kept ratio, FLOPs (vision forward), wall-clock latency.

---

## Related work

This repository represents the main line of the project. Earlier internal exploration on a different VLA backbone serves as **prior-work motivation only** (scorer-based pipeline ceiling observation that motivated the move to VLA-level pruning). It is intentionally not linked from this repository.

External baselines used in evaluation: AutoVLA (no pruning, upper bound), FastV (internal pruning), FastV-selector-at-input (controlled-variable ablation), random pruning, fixed-ratio pruning, and an Importance-Scorer-only fixed-budget variant of our own method.

ToMe / VisionZip / SparseVLM are discussed in the appendix as alternative compression families; they are not direct competitors to learned context-aware pruning.

---

## Citation

> A BibTeX entry will be added upon paper submission / acceptance.

---

## License

See [`LICENSE`](LICENSE). All rights are reserved while the paper is under submission. An open-source license is planned post-acceptance.

---

*Last updated: 2026-06-15*
