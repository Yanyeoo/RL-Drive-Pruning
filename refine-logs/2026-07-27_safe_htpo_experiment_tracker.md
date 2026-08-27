# Safe-HTPO Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | DDP synchronization smoke | Joint BudgetRL; `selection_pg_weight=1` | 32 navtrain scenes | rank checksum, token/budget grad norms, ratio std | MUST | TODO | Must complete before any final run. |
| R002 | M0 | Selection-gradient control | Joint BudgetRL; `selection_pg_weight=0` | 32 navtrain scenes | token-net gradient comparison | MUST | TODO | Verifies the new selection path changes gradients. |
| R003 | M1 | AAAI single-policy screen | Safe-HTPO-Lite; 1 epoch; logistic-normal ratio | navtrain, 8-GPU DDP | reward components, `k` distribution, KL, safety loss | MUST | TODO | New output directory; no resume from v2 budget-only checkpoints. |
| R004 | M2 | Full raw dynamic evaluation | R003 checkpoint, Variant-B physical drop, no denylist | full navtest, 4 shards | raw PDMS, submetrics, P5, CVaR, tokens, FLOPs, latency | MUST | TODO | Reuse fixed controls only if exact protocol matches. |
| R005 | M3 | Paper decision analysis | Aggregate R004 | full navtest | matched-retention Pareto, bootstrap CI | MUST | TODO | Stop if dynamic policy fails matched-retention control. |
| R006 | M4 | Ranking-only screen | fixed `r`, selection policy only | navtest shard 0 | PDMS, safety deltas | MUST for joint claim | TODO | Run after R004 is positive. |
| R007 | M4 | Budget-only screen | `selection_pg_weight=0`, random ratio | navtest shard 0 | PDMS, ratio distribution | MUST for joint claim | TODO | Compare to R006 and joint policy. |
| R008 | M5 | Exact subset pilot | Plackett–Luce / Gumbel-Top-K, action likelihood | small navtrain validation | stability, reward/sample, gradient variance | POST-AAAI | TODO | Promote only if it beats surrogate under equal rollout budget. |
| R009 | M5 | Action-level PPO pilot | exact hybrid action PPO with old logprob cache | small navtrain validation | clipped ratio stats, validation Pareto | POST-AAAI | TODO | GSPO principle at composite-action level. |
| R010 | M6 | Cost-constrained policy | dual compute constraint | navtrain + navtest screen | constraint violation, PDMS–cost | POST-AAAI | TODO | Dynamic `k_s`; no fixed per-scene ratio. |
| R011 | M6 | Safety-constrained policy | cost + CVaR safety constraint | navtrain + navtest screen | tail collision/TTC/drivable loss | POST-AAAI | TODO | Compare against R010. |
| R012 | M7 | Multi-seed confirmation | finalist Safe-HTPO | full navtest, seeds 1–3 | mean/CI/P5/CVaR/Pareto | POST-AAAI | TODO | Required for ICML central result. |
| R013 | M7 | 7B transfer | verified ImpromptuVLA-7B pipeline | designated 7B split | driving metric + ranking | POST-AAAI | TODO | Do not claim before the processor/eval pipeline is fixed. |
| R014 | M7 | Matched SOTA benchmark | strongest comparable baselines | predeclared benchmark track | official metric / latency / tokens | POST-AAAI | TODO | “SOTA” only under a matched public protocol. |
