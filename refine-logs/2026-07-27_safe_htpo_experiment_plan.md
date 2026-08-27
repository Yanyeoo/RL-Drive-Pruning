# Safe-HTPO Research and Execution Plan

**Problem**: A driving VLA must choose both *which* visual tokens to preserve and *how many* tokens each scene can afford, while avoiding safety-critical pruning failures.

**Method thesis**: Safe-HTPO models pruning as a single stochastic hybrid action—continuous scene-level keep ratio plus a token subset—and optimizes it from driving-quality feedback. The keep ratio remains random and scene-adaptive throughout; fixed ratios are evaluation controls only.

**Date**: 2026-07-27

## Non-negotiable protocol decisions

1. **No fixed per-scene keep ratio in the method.** The policy samples `k_s`; 35.5% is an observed mean from a prior policy, not a target to force every scene toward. Fixed `r=0.35`, `r=0.50`, and `r=0.75` are controls for Pareto comparison only.
2. **No manual denylist in a headline result.** Any known-scene denylist, entropy threshold, or decode retry must be reported separately as an engineering safeguard; it cannot be described as an emergent learned fallback.
3. **No unsupported performance claim.** The existing dynamic BudgetRL result is raw PDMS 0.84865 / denylist-fallback 0.87010, below the 0.89879 no-prune baseline. The new policy must be retrained and independently evaluated before it appears in the paper.
4. **A final model must be one synchronized policy.** The current eight independent training shards are not DDP; evaluating `sh0` alone is not a valid eight-GPU training result. Final runs require gradient synchronization across ranks and checkpoint-hash verification.

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
|---|---|---|---|
| C1: The same driving reward improves both token identity and scene-level budget. | This is the core method claim. | A jointly trained stochastic policy beats fixed-ratio SFT at matched *mean* retention without a denylist. | B1, B2, B3 |
| C2: Dynamic budget allocation preserves safety under compute reduction. | Dynamic ratios must be justified by more than a mean score. | Better PDMS–retention Pareto point plus non-inferior P5 / CVaR safety-tail metrics. | B2, B4 |
| Anti-claim: gains come from post-hoc fallback, fixed ratio, or a more permissive average budget. | Prevents invalid attribution. | No-denylist raw results; matched-retention controls; keep-ratio distribution. | B2, B3 |

## Safe-HTPO Technical Framework

### 1. State and frozen environment

For scene `s`, the frozen AutoVLA extracts `N=720` ViT-to-LLM token features. The pruning policy observes:

- token features `x_1, ..., x_N` with camera identity;
- pooled scene feature `h_s`;
- score-distribution statistics (entropy, top-gap, camera-wise mass);
- optional deployment context such as a latency budget profile.

AutoVLA receives no gradient and is only the rollout environment. It generates a trajectory after pruning; NAVSIM computes task quality.

### 2. Hybrid stochastic action

The action is `a_s = (k_s, z_{1:K})`.

#### Continuous random budget

Use a bounded logistic-normal policy during AAAI experiments:

`u_s ~ Normal(mu_phi(h_s), sigma_phi^2)`

`k_s = k_min + (k_max - k_min) sigmoid(u_s)`, where `[k_min, k_max] = [0.2, 0.9]`.

The ratio is intentionally stochastic in training and remains scene-adaptive at evaluation. We report its empirical distribution, mean, variance, quantiles, and conditional behavior by scene difficulty. We never force `k_s=0.5`.

For ICML, compare logistic-normal against a Beta policy. Beta is natively bounded and can express boundary-seeking allocations, but it is not assumed superior without a controlled ablation.

#### Token identity action

The Stage-1 LambdaRank scorer gives logits `s_i=f_theta(x_i)`. Given sampled `k_s`, `K=round(k_s N)`.

**AAAI-safe implementation (low variance):** deterministic Top-K execution with the multi-label Top-K policy surrogate:

`log pi_sel ~= (sum_{i in S} s_i - K logsumexp_j s_j) / K`.

This is the revised `train_scorer_budget_rl.py` implementation. It gives the same driving advantage to ranking and budget paths, but it must be described as a surrogate—not as exact subset sampling.

**ICML-full implementation:** sample an ordered token subset without replacement using Gumbel-Top-K / Plackett–Luce:

`z_1,...,z_K ~ pi_theta(. | x, k_s)`

`log pi_sel(z_{1:K}) = sum_{j=1}^K log[ exp(s_{z_j}/tau) / sum_{i notin z_{<j}} exp(s_i/tau) ]`.

This provides a tractable action likelihood for true action-level PPO/GSPO-style clipping.

### 3. Joint policy objective

The factorized policy is:

`log pi(a_s|s) = log pi_budget(k_s|h_s) + lambda_sel log pi_sel(S_s|x_s,k_s)`.

One task reward updates both factors. In AAAI-Safe-HTPO-Lite, use group-normalized REINFORCE with a trust-region anchor to the SFT scorer. In ICML-Safe-HTPO, cache rollout old-policy likelihoods and optimize the **whole composite action** with action-level clipped PPO:

`rho_s = exp(log pi_new(a_s|s) - log pi_old(a_s|s))`

`L_actor = -min(rho_s A_s, clip(rho_s,1-eps,1+eps) A_s)`.

This borrows the appropriate GSPO principle—outcome reward, importance ratio, and clipping use the same action granularity—without incorrectly applying sequence-length normalization to a variable-K set action.

### 4. Driving reward and safety risk

Let `q_m` denote six NAVSIM submetrics. The quality component is:

`R_drive = alpha sum_m w_m q_m^pruned + beta sum_m w_m (q_m^pruned - q_m^base)`.

Use the current weights only as an initial setting and log all components. Add a safety-degradation term rather than claiming an emergent fallback:

`L_safe = sum_{m in {collision, drivable, TTC}} w_m [q_m^base - q_m^pruned - delta_m]_+`.

AAAI-Lite reward:

`R_AAAI = R_drive - lambda_safe L_safe + lambda_eff (1-k_s)`.

`lambda_safe` is selected before final training using a held-out validation sweep. The raw evaluation result contains no denylist.

ICML constrained objective:

`max E[R_drive]`

subject to `E[C(k_s)] <= C_max` and `CVaR_alpha(L_safe) <= delta_safe`.

`C(k_s)` is a calibrated FLOPs/latency model measured from physical token removal, not merely `1-k_s`. Dual variables update online:

`lambda_c <- [lambda_c + eta_c (mean(C)-C_max)]_+`

`lambda_s <- [lambda_s + eta_s (CVaR_alpha(L_safe)-delta_safe)]_+`.

This does **not** make ratios fixed: it controls average deployment cost and tail safety while allowing each scene's sampled `k_s` to vary. Different `C_max` values produce a dynamic Pareto frontier rather than a single fixed-r method.

### 5. Variance control and data efficiency

- Keep the Stage-1 scorer as a reference policy; use adaptive KL/action-distribution regularization, not only weight-space L2, once stochastic subset likelihood is available.
- For expensive NAVSIM reward, do not directly adopt DAPO dynamic sampling: discarding all-good/all-bad groups wastes completed VLA rollouts.
- AAAI: use global or bucketed group normalization with reward-component logging; group size 16 is the starting point.
- ICML: compare a learned centralized critic `V(h_s, score_stats, k_s)` against same-scene small-group relative advantages. A critic is justified here because each black-box trajectory reward is expensive.
- Use common evaluation seeds and cache metric results. Never replay stale rollout data without old-policy likelihoods and clipping.

### 6. Role of OPD, G-OPD, ExOPD, DAPO, and GSPO

- **GRPO:** useful conceptual baseline for group-relative outcome rewards; expensive when applied as multiple full NAVSIM rollouts per identical scene.
- **DAPO:** do not use as the main algorithm. Its dynamic-sampling recipe assumes cheap verifiable rewards and would discard expensive driving rollouts.
- **GSPO:** use its action-consistency principle in ICML-Safe-HTPO: composite pruning action, composite likelihood ratio, composite clipping.
- **OPD:** optional post-training stabilizer / transfer tool. The SFT scorer is a reference policy, not yet a reward-optimized teacher.
- **G-OPD / ExOPD:** defer. They require a reliable reward-trained teacher and exact policy likelihoods. Applying reward extrapolation before establishing a strong Safe-HTPO teacher risks amplifying attention/SFT biases and budget collapse.

## Paper Storyline

### AAAI submission: compact and defensible

The paper must prove only:

1. a driving-reward-trained **joint stochastic pruning policy** improves the quality–compute tradeoff against fixed-ratio controls at matched mean retention;
2. its ratio distribution is scene-adaptive and does not obtain gains through a manual denylist;
3. Stage-1 ranking and Stage-2 stochastic budget are both necessary.

Do not claim closed-loop evaluation, dynamic SOTA, emergent fallback, 7B driving performance, or an unpruned-baseline win unless new raw results directly establish them.

### ICML continuation: full method paper

The paper extends AAAI-Lite into Safe-HTPO with exact stochastic subset likelihood, action-level PPO, dual compute constraints, CVaR safety-tail constraints, multi-seed reliability, multi-backbone transfer, and dynamic Pareto-frontier evaluation.

## Experiment Blocks

### B0: Instrumentation and policy smoke test

- **Claim tested:** both policy factors receive gradients and the policy is genuinely stochastic.
- **Data:** 32 navtrain scenes; 8 GPU DDP smoke.
- **Systems:** SFT scorer; joint BudgetRL with `selection_pg_weight=0`; joint BudgetRL with `selection_pg_weight=1`.
- **Metrics:** token-net grad norm, budget-net grad norm, ratio mean/std, selection entropy, checkpoint SHA256 across all ranks, finite reward/advantage rate.
- **Success criterion:** all ranks have identical synchronized model checksum; both parameter groups have nonzero gradient when selection weight is one; sampled ratios have nonzero std and stay inside [0.2, 0.9].
- **Failure interpretation:** any mismatch blocks all expensive training.
- **Target:** Appendix reproducibility paragraph / internal gate.
- **Priority:** MUST-RUN.

### B1: AAAI-Lite single-policy training

- **Claim tested:** a common driving reward can update token ranking and dynamic budget.
- **Data:** navtrain; clean split from navtest.
- **Systems:** synchronized 8-GPU joint policy initialized from LambdaRank SFT; no manual denylist; attention-mask training proxy.
- **Setup:** 1 epoch for the time-boxed paper screen; group size 16; random logistic-normal budget in [0.2, 0.9]; `selection_pg_weight=1`; Stage-1 reference anchor; pre-registered `lambda_safe`, `lambda_eff`, and seed.
- **Metrics:** training reward components, `k` mean/std/quantiles, token ranking movement, safety loss, KL, action entropy.
- **Success criterion:** stable optimization with no ratio collapse to either bound; selection policy actually changes relative to SFT.
- **Failure interpretation:** remove joint-RL claim and do not proceed to headline evaluation.
- **Target:** method validation / internal gate.
- **Priority:** MUST-RUN.

### B2: Full dynamic Variant-B evaluation

- **Claim tested:** learned dynamic allocation improves the PDMS–retention Pareto point.
- **Data:** all 11,576 navtest scenes in four verified disjoint shards; physical Variant-B token drop.
- **Compared systems:** no prune; SFT scorer at fixed r=0.35/0.50/0.75; attention-L12; FastV; SparseVLM; new joint dynamic policy. Existing baselines are reused only if their code path, split, and Variant are identical; otherwise rerun.
- **Metrics:** raw PDMS; six submetrics; average/median/P10/P90 keep ratio; actual retained tokens; measured FLOPs/latency; P5 PDMS; worst-5% collision/drivable/TTC delta. No denylist-fallback score in the main table.
- **Success criterion:** dynamic policy exceeds the fixed SFT scorer at matched mean retention. Its first relevant screen is the existing raw SFT r=0.35 value, 0.85515; for a strong AAAI claim it should produce a statistically credible positive Pareto improvement and no worsening of safety-tail metrics.
- **Failure interpretation:** if it only improves mean PDMS via safety-tail regression, the safety claim fails. If it does not beat matched fixed retention, the dynamic method is not paper-ready.
- **Target:** Main table + Pareto figure.
- **Priority:** MUST-RUN.

### B3: Mechanism isolation

- **Claim tested:** both selection RL and budget RL are needed.
- **Systems:** (i) SFT fixed r; (ii) selection-RL fixed r; (iii) budget-only dynamic (`selection_pg_weight=0`); (iv) joint dynamic (`selection_pg_weight=1`).
- **Dataset:** first navtest shard for screening; only promote positive variants to full navtest.
- **Metrics:** PDMS at mean retention, keep-ratio distribution, submetric deltas.
- **Success criterion:** joint policy improves over both one-factor variants at comparable mean retention.
- **Target:** main ablation or appendix depending on full-result availability.
- **Priority:** MUST-RUN for a joint-method claim; otherwise cut the claim.

### B4: Dynamic and safety diagnosis

- **Claim tested:** dynamic allocation is tied to observable scene risk, not random noise.
- **Analyses:** correlation and calibrated bins between `k_s` and baseline difficulty, safety loss, vehicle/VRU density if metadata is available, and score entropy; show easy/medium/hard examples with the actual selected token maps.
- **Metrics:** Spearman correlation; ratio distributions; P5 safety deltas; bootstrap confidence intervals.
- **Success criterion:** monotone or otherwise interpretable allocation pattern, plus no hidden denylist.
- **Target:** one qualitative figure and appendix diagnostics.
- **Priority:** MUST-RUN only after B2 is positive.

### B5: ICML exact-action Safe-HTPO

- **Claim tested:** exact stochastic subset likelihood and action-level PPO are more stable/effective than deterministic Top-K surrogate.
- **Systems:** AAAI-Lite surrogate; Plackett–Luce policy; Gumbel-Top-K estimator; action-level PPO vs REINFORCE; optional critic.
- **Success criterion:** exact-action method gives a better multi-seed Pareto frontier or lower variance under equal rollout budget.
- **Target:** ICML main method ablation.
- **Priority:** POST-SUBMISSION.

### B6: ICML constrained safety frontier

- **Claim tested:** dual cost and CVaR safety constraints replace hand-designed fallback with learned safe compute allocation.
- **Systems:** fixed efficiency coefficient; cost-only dual policy; cost + CVaR policy.
- **Metrics:** PDMS–FLOPs curve, CVaR safety loss, worst-5% submetric degradation, constraint violation frequency.
- **Success criterion:** cost+CVaR dominates cost-only on safety tail with minor or no average quality loss.
- **Target:** ICML central figure.
- **Priority:** POST-SUBMISSION.

### B7: ICML generalization and SOTA verification

- **Dataset/backbone targets:** NAVSIM AutoVLA-3B primary; ImpromptuVLA-7B only after processor/eval pipeline is fixed; nuScenes open-loop only as transfer, clearly separated from NAVSIM PDM; an additional publicly reproducible driving VLM/VLA benchmark if a matched token-drop protocol is available.
- **Protocol:** three training seeds; frozen evaluation scripts; full raw result release; matched physical-drop comparisons; direct reimplementation or official baselines where required.
- **SOTA criterion:** predeclare exact benchmark, split, backbone, token budget convention, and physical-drop implementation. “SOTA” is valid only within the matched protocol and with confidence intervals.
- **Priority:** POST-SUBMISSION.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision gate | 8-GPU cost | Risk / mitigation |
|---|---|---|---|---:|---|
| M0 | Synchronization and gradient smoke | B0, 32 scenes | Same checkpoint hash on all ranks; finite joint gradients | <0.5 h | DDP bug; block final run until fixed |
| M1 | AAAI policy screen | B1, 1 epoch | Random ratios remain stable; selection changes; no NaNs | 5–6 h estimated by current launcher | Existing launcher is not DDP; replace it before run |
| M2 | Full raw evaluation | B2, dynamic only, 4 navtest shards | Beat matched fixed-SFT Pareto without denylist | ~3.5–4 h from prior 4-shard eval | Variant-A training / Variant-B eval mismatch; report explicitly |
| M3 | Aggregate and paper gate | metrics + bootstrap + safety tail | Decide submit-ready / revise method / stop claim | 0.5–1 h | Do not use fallback-adjusted headline |
| M4 | Joint ablation screen | B3 on one navtest shard | Joint > either single-factor variant | 1–2 h | Run only if M2 passes |
| M5 | ICML method completion | B5/B6 | Exact policy + constraints exceed AAAI-Lite | multi-week | Use staged pilots and ASHA |
| M6 | ICML SOTA confirmation | B7, 3 seeds / multi-backbone | Matched-protocol SOTA with CIs | multi-week | Freeze benchmark and baselines first |

## AAAI 12-Hour Window

### Feasible scope

A **one-epoch synchronized joint-policy screen plus one full dynamic Variant-B evaluation** can plausibly fit in a 12-hour window:

- M0 smoke: <0.5 h;
- M1 one epoch: 5–6 h according to the current launcher’s validation estimate;
- M2 dynamic 4-shard full-navtest eval: approximately 3.5–4 h;
- aggregation and buffer: 1.5–2 h.

Total: approximately 10.5–12 h.

### Not feasible within the same window

- 3-epoch full training: current launcher documents 16–17 h before evaluation;
- multiple training seeds;
- full B3 factor ablation;
- implementation and debugging of exact Plackett–Luce PPO;
- full fixed-control reruns when existing matched controls are valid.

### Required preflight before the 12-hour clock

1. Replace the independent-shard launcher with real DDP / synchronized gradient all-reduce.
2. Add an explicit `--selection-pg-weight` launcher argument and create a new run directory; never resume a v2 budget-only checkpoint.
3. Disable all score-changing denylist fallback configuration for the dynamic run.
4. Verify four navtest shard YAML files are disjoint and union to 11,576 scenes.
5. Run M0 and verify rank checksums and action statistics.

## ICML Continuous Optimization Program

### P0: Reproducible platform (week 1)

- DDP training, checkpoint provenance, immutable configs, metrics database, deterministic eval manifests.
- Automated collection of raw PDMS/submetrics, retained-token statistics, latency/FLOPs, P5/CVaR safety, and action likelihood diagnostics.
- Build a small navtrain validation suite for 2-hour candidate screening.

### P1: Policy representation and estimator search (weeks 1–2)

Search factorially but cheaply:

- budget distribution: logistic-normal vs Beta;
- subset policy: Top-K surrogate vs Plackett–Luce / Gumbel-Top-K;
- estimator: REINFORCE vs action-level PPO; critic on/off;
- trust region: fixed KL vs adaptive KL;
- safety loss coefficient and safety margin;
- entropy schedule and selection temperature.

Use ASHA / Bayesian search on fixed compute budgets. Promote candidates only if they improve matched-retention validation Pareto and do not worsen safety-tail statistics.

### P2: Constrained Safe-HTPO (weeks 2–4)

- Fit token count to real Variant-B FLOPs / latency.
- Train separate dynamic policies under multiple **average compute caps**, e.g. a low-, medium-, and high-cost profile. These caps never make per-scene `k_s` fixed.
- Add dual updates for compute cap and CVaR safety constraint.
- Compare against fixed-ratio and unconstrained dynamic policies.

### P3: Reliability and mechanism evidence (weeks 4–5)

- Three independent training seeds for finalists;
- bootstrap CIs over navtest scenes;
- matched-retention controls for every dynamic policy;
- difficult-scene taxonomy; qualitative token maps; failure cases;
- audit all fallback code paths.

### P4: Generalization and SOTA protocol (weeks 5–8)

- Repair and validate the ImpromptuVLA-7B processor/evaluation path before making any 7B claim;
- evaluate 3B-to-7B transfer and, if needed, train a 7B-native policy;
- run nuScenes only as open-loop transfer; do not mix it with NAVSIM PDM claims;
- reproduce strongest current competitors under matched token count, backbone, and physical-drop protocol;
- pre-register the SOTA table and report only the benchmark-track claim that is directly supported.

### P5: Optional OPD / ExOPD transfer (after a strong Safe-HTPO teacher exists)

- Train a high-quality constrained Safe-HTPO teacher first.
- Define exact likelihoods for both budget and subset actions.
- Use on-policy distillation to transfer to a new backbone or compact policy.
- Test reward extrapolation only with strict action KL, ratio clipping, and external driving evaluation; it is a research branch, not a prerequisite for the ICML core method.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Ratio collapses near 0.2 due to efficiency bonus | Preserve stochastic policy entropy; lower `lambda_eff`; use adaptive KL; in ICML use dual cost constraint rather than a fixed reward coefficient. |
| Joint selection training is unstable | AAAI uses low-variance Top-K surrogate; ICML pilots exact Plackett–Luce policy separately. |
| Mean PDMS rises while dangerous tail worsens | Gate on P5 / CVaR safety loss and collision/TTC/drivable tail metrics. |
| DDP changes behavior relative to old shard training | M0 hash, gradients, and per-rank action-statistics tests; never evaluate only rank 0 without synchronization. |
| Variant-A training fails to transfer to Variant-B drop | Run a held-out Variant-B validation screen; if gap is material, train with drop or report the mismatch as a limitation. |
| Reward calls make GRPO/DAPO too expensive | Avoid large same-scene group sampling; use critic / small-group pilots only. |
| “SOTA” comparison is protocol-mismatched | Predeclare benchmark/split/backbone/token metric/physical-drop convention; rerun or exclude incomparable baselines. |

## Final Checklists

### AAAI submission gate

- [ ] New synchronized joint-policy checkpoint, not a v2 budget-only checkpoint.
- [ ] Raw no-denylist full-navtest Variant-B result.
- [ ] Dynamic policy exceeds fixed SFT control at matched mean retention.
- [ ] Keep-ratio distribution and safety-tail metrics are reported.
- [ ] All claims match actual evidence; no unsupported 0.9045/0.9107 attribution.
- [ ] 7B claim restricted to verified ranking analysis unless 7B driving eval completes.

### ICML gate

- [ ] Exact stochastic subset policy with tractable action likelihood.
- [ ] Action-level PPO / GSPO-style clipping ablation.
- [ ] Compute and CVaR safety constrained policy is validated.
- [ ] Three-seed full results and CIs.
- [ ] Matched-protocol generalization and direct competitor comparison.
- [ ] SOTA claim names one exact benchmark track and metric.
