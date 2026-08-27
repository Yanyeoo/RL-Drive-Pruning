# DriveToken — Code and Data Supplement

This archive contains the key scripts, evaluation code, and result CSVs needed to reproduce
the experiments in "DriveToken: Learning Dynamic Vision-Token Budgets from Driving Rewards
for Autonomous VLAs" (AAAI 2027 submission).

## Directory Structure

```
code_data_summit/
├── README.md                     # This file
├── scripts/
│   ├── run_pdm_score_cot.py      # Main evaluation script (NAVSIM, physical token drop)
│   ├── train_scorer_grpo.py      # Stage 2 Budget RL training (scorer + budget head)
│   ├── train_scorer_budget_rl.py # Stage 2 Budget RL training (alternative)
│   ├── s3_train_scorer_mse.py    # Stage 1 MSE scorer training
│   ├── s3_aggregate_maintable.py # Aggregate 4-shard CSVs into main table
│   ├── gen_table1_draft.py       # Generate Table 1 draft from landed CSVs
│   ├── compute_flops_table.py    # FLOPs estimation for AutoVLA-3B
│   ├── profile_varB_speedup.py   # Wall-clock latency profiling
│   ├── analyze_taucut_dynamic.py # τ-cut calibration and analysis
│   ├── run_impromptu7b_nuscenes_eval.py  # 7B nuScenes cross-dataset evaluation
│   ├── run_budget_rl_eval.sh     # Budget RL evaluation launcher
│   ├── run_baseline_pareto_gpu4.sh # Baseline (FastV/Random/PruMerge/SparseVLM) evaluation
│   ├── run_s3_maintable_full_navtest.sh # Full navtest main table evaluation
│   └── setup_navsim_env_vars.sh  # Environment setup
├── results/
│   └── tokenprune_S3_full/       # Full navtest evaluation CSVs (4-shard)
│       ├── MT_varBsafe_scorer_r05_sh*.csv   # SFT scorer, r=0.5
│       ├── MT_rl_shaped_r05_sh*.csv         # RL-shaped scorer, r=0.5
│       ├── MT_budget_rl_dynamic_sh*.csv     # Budget RL dynamic
│       ├── MT_attn_L12_r10_sh*.csv          # No Prune baseline (r=1.0)
│       ├── MT_attn_L12_r05_sh*.csv          # Attention teacher, r=0.5
│       ├── MT_scorer_mse_r05_sh*.csv        # MSE scorer, r=0.5
│       ├── MT_fastv_l2_drop_r*_sh*.csv      # FastV baseline (physical drop)
│       ├── MT_random_drop_r*_sh*.csv        # Random baseline (physical drop)
│       ├── MT_prumerge_cls_drop_r*_sh*.csv  # PruMerge baseline (physical drop)
│       ├── MT_sparsevlm_text_drop_r*_sh*.csv # SparseVLM baseline (physical drop)
│       └── MT_sparsevlm_text_r075_sh*.csv   # SparseVLM, r=0.75
└── checkpoints/                  # Model weights (placeholder — will be released upon publication)
```

## Environment

- Python 3.10+
- PyTorch 2.4+
- NVIDIA H20 GPUs (97 GB VRAM)
- NAVSIM v1 (navtrain + navtest splits)

## Key Commands

### Stage 1: LambdaRank Scorer Training
```bash
python scripts/train_scorer_grpo.py --stage sft --epochs 20 --lr 3e-4
```

### Stage 2: Budget RL Training
```bash
python scripts/train_scorer_grpo.py --stage rl --epochs 5 --lr 1e-5 \
  --lambda_d 2.0 --lambda_s 0.5 --lambda_e 0.15 --G 8 --K 4
```

### Evaluation (Physical Token Drop, 4-Shard)
```bash
bash scripts/run_s3_maintable_full_navtest.sh
```

### Aggregate Results
```bash
python scripts/s3_aggregate_maintable.py
```

## Result CSVs

Each CSV row contains one scene with the following columns:
- `token`: Scene identifier
- `valid`: Whether the scene produced a valid trajectory
- `no_at_fault_collisions`, `drivable_area_compliance`, `ego_progress`,
  `time_to_collision_within_bound`, `comfort`, `driving_direction_compliance`: NAVSIM sub-metrics
- `score`: PDMS (aggregate metric, 0–1)

Full navtest results are split across 4 shards (sh0–sh3), each covering ~2,949 scenes.
Aggregate PDMS = weighted mean of per-shard means.

## Contact

Code and model weights will be released upon publication.
