# TSC v34/r30 Pairwise Target-Simulator Adaptation Curve

Date frozen: 2026-08-09. This document is written before any v34/r30 result is generated.

## Question

Does the source-trained antisymmetric pairwise causal action model become safe and useful when it receives a small, increasing number of complete counterfactual action groups from an uncalibrated target-city simulator?

This experiment is not passive historical-data adaptation. Each target information unit is one matched simulator snapshot with outcomes for every admissible signal action. The selected groups are removed from evaluation. Zero-shot uses no target transition outcome.

## Frozen evidence and holdout

- Development targets: `grid4x4`, `cologne1`, `ingolstadt1`, `atlanta_1x5`, `hangzhou_4x4`, and `manhattan_28x7`.
- Frozen counterfactual bank: v25/r21, aggregate SHA-256 `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`.
- Frozen source-rule baseline SHA-256: `25e6291ba9ff7348642baf09498c9c7bea36ef3fd77a7ee6680b289a03201acb`.
- Frozen v26/r22 rigid reference audit SHA-256: `05fc6e94757b74221b781878fa6bdae3b4142ee4d866aceea13332b5e4932154`.
- Salt Lake City remains unopened during method development and this entire curve.

## Information budgets

Run complete target action-group budgets `B = {0, 8, 16, 32, 60}`. The initially considered `B = 120` point is excluded before execution because the frozen `cologne1` and `ingolstadt1` targets contain only 76 and 90 complete groups, respectively; it would empty or nearly empty their evaluation sets. Selection uses the already frozen coverage-first protocol and seed `20260803`. Groups generated with seed `4047` supply 40% of each nonzero budget to calibration; they are excluded from model fitting and from evaluation. The remaining selected groups enter model adaptation and are also excluded from evaluation.

## Same-subset comparators

Every city-budget shard independently fits and evaluates all four families on the identical retained target groups:

1. `causal_rigid_advantage`: original rigid causal core, fitted under the same source-plus-target information budget.
2. `causal_group_normalized_rigid_advantage`: per-action-group normalized rigid core, fitted under the same information budget.
3. `causal_target_only`: the established low-capacity target specialist comparator.
4. `causal_antisymmetric_pairwise_advantage`: the v32/r28 all-action pairwise core pooled with target adaptation groups.

No family may read the calibration groups in this offline screening stage. No target evaluation outcome may enter fitting, family selection, or hyperparameter selection.

## Primary estimand

The primary metric is the city-macro mean normalized action regret. Within every retained matched action group, regret is divided by that group's observed action-cost range. Lower is better. Family comparisons at a budget are paired because they use exactly the same retained group IDs.

## Frozen success rule

The pairwise family passes the adaptation screen only if all conditions hold:

- At `B = 60`, macro regret is at least 10% lower than the same-subset `causal_rigid_advantage` comparator.
- At that passing budget, at least four of six cities improve over the same-subset rigid comparator.
- Maximum absolute city regression at that budget is no greater than `0.05` normalized regret.
- Pairwise macro regret at `B = 60` is lower than at `B = 0`.
- Spearman rank correlation between budget and pairwise macro regret is at most `-0.8`; individual adjacent steps need not be monotone.

If no budget passes, the pooled pairwise model is rejected. A later target-specific pairwise specialist must be specified as a new method and tested in a separately frozen experiment; v34 results cannot be used to tune v34.

## Promotion rule

Passing v34 permits, but does not itself trigger, Salt Lake evaluation or closed-loop SUMO promotion. The selected budget and family must first pass a separate source/calibration-only deployment selector specification. Failing v34 keeps Salt Lake sealed.
