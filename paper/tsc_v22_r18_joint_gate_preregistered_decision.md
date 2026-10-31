# TSC v22/r18 Joint Mechanism-Gate Development Decision

Recorded on 2026-08-08 (Asia/Shanghai) after physically validating frozen r21
budgets 0, 8, 16, 32, and 60. The r21 budget-120 result had not been inspected
when this decision was written.

## Evidence available at freeze time

- r21 budgets 8 and 16: full and exact-rigid guarded controllers were identical
  in all 48 paired target-seed rollouts and both fell back to the selected
  source pressure prior.
- r21 budget 32: the full mechanism path improved Cologne but suppressed the
  exact-rigid improvement in Atlanta; city-first guarded performance was worse
  for full than rigid.
- r21 budget 60: full guarded transfer improved more than rigid on average, but
  failed the preregistered safety admission with 45 collision incidents versus
  40 for the selected source prior and slightly exceeded the worst-city harm
  tolerance.
- The r21 policy list contains an exact-rigid guarded controller but not an
  exact-rigid unguarded MPC diagnostic. It therefore cannot isolate model
  contribution when both guards fall back.

## Frozen r22 change set

1. Source training, cached counterfactual groups, city folds, evaluation seeds,
   durations, source rule grid, target budgets, and safety thresholds remain
   unchanged from r21/r17.
2. For target budgets with at least four adaptation groups, grouped OOF target
   regret jointly selects a mechanism retention gate
   `g in {0, 0.25, 0.5, 0.75, 1}` and a sample-capped target-specialist weight.
3. A positive mechanism gate is admitted only if it improves mean regret over
   the jointly optimized `g=0` exact-rigid candidate, respects the target-group
   harm-fraction limit, and stays within the worst-group regret tolerance.
4. Zero-shot transfer is unchanged: no target labels means the source-selected
   mechanism stack is used as frozen.
5. The exact-rigid implementation uses a prediction-equivalent core-only fast
   path. A regression test compares it row by row with the previous full-fit
   then forced-off implementation.
6. The matrix adds `causal_target_only_contrast_mpc` and
   `cfcmt_fused_rigid_contrast_mpc`. Model contribution is evaluated with MPC
   diagnostics; deployability is evaluated separately with guarded policies.
7. No target closed-loop selection seeds are used in r22. If the model gate
   passes but guarded deployment fails safety, closed-loop target selection is
   a subsequent, separately accounted experiment rather than an r22 patch.

## Acceptance contract

Integrity requires 16 target networks, six city/benchmark groups, three
evaluation seeds, all requested policies, SUMO/libsumo 1.22, the immutable
source-tree hash, zero teleports, complete collision ledgers, and no failed
rollouts.

Model contribution is tested at budgets 8, 16, 32, 60, and 120 using fused MPC
against both target-only MPC and exact-rigid MPC. A qualifying budget requires
at least 0.1% city-first mean gain against each comparator, source mechanism
use in at least two city groups, a non-degenerate source-target fusion, no fewer
winning city groups than the comparators, and worst-city harm no greater than
0.5% relative to the selected source prior.

Guarded efficacy is tested at budgets 60 and 120. Each must have negative mean
and median city-first effects versus the selected source prior, wins in at
least four of six city groups, worst-city harm no greater than 0.5%, top gain
share below 90%, no additional collision incidents, and no teleports.

## Precommitted interpretation

- If model contribution and guarded efficacy both pass, freeze r22 for an
  untouched 3,600-second confirmatory matrix.
- If model contribution passes but guarded safety fails, keep the estimator
  frozen and test a disjoint closed-loop target selector with its simulator
  rollout budget reported separately.
- If the mechanism gate is zero in every target or fused MPC does not improve
  exact-rigid MPC at any contribution budget, do not claim a causal-mechanism
  gain. The next development hypothesis is a lane- and movement-normalized
  local coordinate core, evaluated as a new module rather than hidden tuning.
