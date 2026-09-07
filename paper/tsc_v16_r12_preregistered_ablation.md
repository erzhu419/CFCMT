# TSC v16 r12 Target-Specialist Ablation Protocol

Date fixed: 2026-08-08, before any r12 development result was available.

## Purpose

The r9 and partial r10 matrices show that target counterfactual information is useful, but the existing full CFCMT path does not absorb it consistently. At budget 60, the coverage-first r10 target-only guard improves 5/6 city groups with no degrading city, whereas the old full CFCMT regularized controller improves only 1/6 city groups. The r12 ablation therefore tests a conservative four-level controller:

1. causal mechanism refinement;
2. independently fitted target-only specialist;
3. rigid causal-core fallback;
4. pressure-policy prior.

Every learned layer retains its independently calibrated uncertainty and safety gate. The target specialist is unavailable unless the adaptation split trains a positive-weight target head. Calibration groups are never refitted into the deployment predictor.

## Frozen protocol

- Snapshot SHA-256: `ab581dd306d91b3b4697444b5b0ca837055685e7740d288c59d0e63994b45e44`
- Source-tree SHA-256: `658df6b1ffe2c39b586008471de05946e511fd143e23d7fa75a9e1198fdb0b93`
- Hierarchy protocol: `conservative-bound-mechanism-target-specialist-rigid-core-pressure-v1`
- SUMO/libsumo: 1.22.0
- Networks: 16 networks in 6 city groups
- Evaluation seeds: 131, 337, 911
- Target group budgets: 0, 8, 16, 32, 60, 120
- Counterfactual cache: audited no-teleport v15 cache
- Source-rule cache: audited v2 cache
- Deployment fitting: adaptation labels only; disjoint calibration labels select gates only
- Cluster tasks: t0233--t0238

## Required checks

The r12 result is considered technically valid only if all of the following hold:

- every budget result passes the protocol, source-hash, seed-separation, completeness, and finite-value audits;
- all fixed-policy rows are invariant to the preceding revision under the same scenario and seed;
- collision events and teleport events remain zero for every guarded or regularized policy row; explicitly unguarded MPC diagnostics are audited and excluded from safety claims;
- `selected_target_specialists` is zero when no fitted target head exists;
- every accepted specialist action is represented in the hierarchical audit rather than inferred from model-fit metadata;
- the target calibration groups remain absent from deployment-model fitting.

## Performance acceptance gates

The new full CFCMT guard is considered to have repaired the old full path only if budgets 60 and 120 satisfy all primary gates:

- mean paired city effect versus `selected_source_prior` is negative;
- median paired city effect versus `selected_source_prior` is negative;
- at least 4/6 city groups improve;
- worst-city relative degradation is no greater than 0.5%;
- the top-gain city contributes less than 90% of total positive gain;
- the method is no worse than the standalone target-only guard by more than 0.2% in mean paired city effect;
- performance does not rely on a nonzero collision or teleport rate.

Secondary evidence is stronger if the mean effect improves monotonically from budgets 32 to 60 to 120, the effective gain-city count increases with budget, and full CFCMT improves over the target-only specialist in at least two city groups.

## Interpretation rules

- Passing only the mean-effect gate is insufficient when the median is zero or all gain comes from one city.
- If the target specialist is never selected, r12 is an implementation or calibration failure even if aggregate performance improves.
- If the target-only baseline passes but full CFCMT fails, the next revision must calibrate hierarchy subsets on disjoint target calibration groups; it must not tune thresholds on evaluation rollouts.
- The 600-second, three-seed matrix remains developmental. Formal claims require a fresh long-horizon, multi-seed run after the development protocol is frozen.
