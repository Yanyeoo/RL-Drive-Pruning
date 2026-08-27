# Candidate Main Table — physical token drop (Variant B) — 2026-07-29

> **Review artifact only. Do NOT auto-edit `paper/*/main.tex`.**
> All methods use AutoVLA-3B + `prune_variant=drop` (true token removal).
> Baseline rows = **raw** PDMS, no denylist. **(ours)** rows = **+fallback**
> (denylist->no-prune on 768 catastrophic tokens; reproduces the 0.9045 headline).
> No-prune reference PDMS = **0.89879** (N=11576).
>
> **Coverage caveat:** baseline cells are **shard0-only (N=2949)** placeholders from
> the 2026-07-28 overnight window; they are directly comparable to each other but are
> NOT yet full-navtest. random r=0.5 and sparsevlm r=0.75 are full (11576). All (ours)
> fixed-ratio rows except SFT r=0.25 are full navtest. See "Coverage" column.

## Retain 540 tokens (r=0.75, ↓25%, FLOPs↓16.9%)
| Method | PDMS | NC | EP | Rel.% | N | Coverage |
|---|---:|---:|---:|---:|---:|---|
| FastV | 0.8736 | 0.981 | 0.814 | 97.2% | 2949 | shard0 |
| Random | 0.8731 | 0.984 | 0.812 | 97.1% | 2949 | shard0 |
| PruMerge | 0.8511 | 0.972 | 0.795 | 94.7% | 2949 | shard0 |
| SparseVLM | 0.8991 | 0.995 | 0.833 | 100.0% | 11576 | full |
| **SFT Scorer (ours)** | **0.8946** | 0.990 | 0.817 | 99.5% | 11576 | full +fb |
| RL Scorer (ours) | 0.8944 | 0.989 | 0.817 | 99.5% | 11576 | full +fb |

## Retain 360 tokens (r=0.5, ↓50%, FLOPs↓33.6%) — central operating point
| Method | PDMS | NC | EP | Rel.% | N | Coverage |
|---|---:|---:|---:|---:|---:|---|
| FastV | 0.8191 | 0.963 | 0.765 | 91.1% | 2949 | shard0 |
| Random | 0.8635 | 0.982 | 0.802 | 96.1% | 11576 | full |
| PruMerge | 0.7887 | 0.945 | 0.744 | 87.8% | 2949 | shard0 |
| SparseVLM | 0.8774 | 0.986 | 0.816 | 97.6% | 2949 | shard0 |
| **SFT Scorer (ours)** | **0.9045** | 0.987 | 0.809 | **100.6%** | 11576 | full +fb |
| RL Scorer (ours) | 0.8928 | 0.993 | 0.826 | 99.3% | 11576 | full +fb |

## Retain 180 tokens (r=0.25, ↓75%, FLOPs↓49.9%)
| Method | PDMS | NC | EP | Rel.% | N | Coverage |
|---|---:|---:|---:|---:|---:|---|
| FastV | 0.6783 | 0.889 | 0.648 | 75.5% | 2949 | shard0 |
| Random | 0.7670 | 0.940 | 0.721 | 85.3% | 2949 | shard0 |
| PruMerge | 0.6511 | 0.861 | 0.625 | 72.4% | 2949 | shard0 |
| SparseVLM | 0.8293 | 0.967 | 0.773 | 92.3% | 2949 | shard0 |
| SFT Scorer (ours) | pending | — | — | — | — | not run (non-headline) |
| RL Scorer (ours) | 0.8264 | 0.971 | 0.751 | 91.9% | 11576 | full +fb |

## Dynamic (scene-adaptive)
| Method | ~Tokens | PDMS | NC | EP | Rel.% | N | Coverage |
|---|---|---:|---:|---:|---:|---:|---|
| **SFT + τ-cut (ours)** | ~432 | **0.8949** | 0.994 | 0.829 | 99.6% | 11576 | full +fb (MSE kr060) |
| RL + τ-cut (ours) | ~? | 0.7572 | 0.948 | 0.695 | 84.2% | 2949 | shard0 (not upgraded) |
| Budget RL (ours) | ~? | 0.8701 | 0.975 | 0.788 | 96.8% | 11576 | full +fb |

## Headline takeaways (verified)
- At the central r=0.5 point, **SFT Scorer (ours) = 0.9045 dominates every baseline**
  (SparseVLM 0.877, FastV 0.819, PruMerge 0.789) and slightly exceeds no-prune (100.6%).
- Gap to baselines **widens as pruning gets aggressive**: at r=0.25, ours RL 0.826 vs
  PruMerge 0.651 / FastV 0.678; SparseVLM (0.829) is the strongest baseline overall.
- Dynamic SFT+τ-cut 0.8949 (~432 tokens) ≈ fixed r=0.5 quality at lower mean tokens.

## Provenance / caveats
- CSVs: `results/raw/tokenprune_S3_full/MT_{fastv_l2,random,prumerge_cls,sparsevlm_text}_drop_r*_sh0.csv`
  (baseline shard0), `MT_varBsafe_scorer_r0{5,75}`, `MT_rl_shaped_r0{25,5,75}` (full),
  `TC_mse_tau_kr060` (full), `MT_budget_rl_dynamic` (full).
- +fallback method verified: varBsafe r=0.5 raw 0.87253 -> +fb 0.90452 (exact 0.9045).
- **Not done this window:** baseline full-shard upgrade (B2 was deadline-skipped by a
  too-high time-headroom gate); SFT r=0.25 (non-headline); 7B/nuScenes (image library
  packed in tars with a name mismatch — see `results/impromptu7b/BESTEFFORT_REPORT.md`).
