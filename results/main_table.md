# Full navtest Main Table (PDMS)

Baseline = attn_L12 r=1.0 (no prune) = **0.89879** (N=11576).
All numbers: raw mean, then with denylist-fallback applied.
Methods marked * use only shard0 (2949 scenes), not full navtest.


| Method | Variant | N | raw PDMS | +fallback | Δ vs base |
|---|---|---:|---:|---:|---:|
| **baseline** (attn_L12 r=1.0) | — | 11576 | 0.89879 | 0.89879 | 0 |
| scorer r=0.10 (SFT) | drop/mask | 11576 | 0.68858 | 0.70365 | -0.1951 |
| scorer r=0.35 (SFT) | drop/mask | 11576 | 0.85515 | 0.87426 | -0.0245 |
| scorer r=0.50 (SFT) | drop/mask | 11576 | 0.89199 | 0.89587 | -0.0029 |
| scorer r=0.75 (SFT) | drop/mask | 11576 | 0.89830 | 0.89860 | -0.0002 |
| scorer_mse r=0.50 | drop/mask | 11576 | 0.88940 | 0.89216 | -0.0066 |
| scorer_mse r=0.75 | drop/mask | 11576 | 0.89681 | 0.89648 | -0.0023 |
| attn_L12 r=0.50 | drop/mask | 11576 | 0.89006 | 0.89406 | -0.0047 |
| varBsafe r=0.50 (drop) | drop/mask | 11576 | 0.87253 | 0.90452 | +0.0057 |
| varBsafe r=0.75 (drop) | drop/mask | 2949 | 0.88498 | 0.89625 | -0.0025 |
| sparsevlm r=0.75 | drop/mask | 11576 | 0.89908 | 0.89902 | +0.0002 |
| fastv r=0.50 | drop/mask | 11576 | 0.83296 | 0.84108 | -0.0577 |
| fastv r=0.75 | drop/mask | 11576 | 0.88226 | 0.88556 | -0.0132 |
| rl_shaped r=0.50 | drop/mask | 11576 | 0.89094 | 0.89281 | -0.0060 |
| BUDGET_RL dynamic | drop/mask | 11576 | 0.84865 | 0.87010 | -0.0287 |
| BUDGET_RL fixed r=0.50 | drop/mask | 11576 | 0.87227 | 0.90399 | +0.0052 |

## Caveats
- 0.9045 headline = varBsafe r=0.50 (drop) + denylist fallback. NOT from RL dynamic budget.
- BUDGET_RL dynamic (RL-learned per-scene budget) is lowest at 0.8701, below baseline by 2.9pt.
- random / scorer r=0.65 / scorer r=0.90 CSVs not yet produced.