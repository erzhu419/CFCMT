# TSC v32/r28 Preregistered Decision: All-Action Causal Preference

Date frozen: 2026-08-09

## Motivation

v31 showed that aligning the target with matched-group normalized regret is
useful but insufficient. The existing core still regresses each candidate only
against the selected pressure-rule reference. v32 changes only the ranking
representation: every unordered action pair in a matched state becomes a
training contrast, and both orientations are included to enforce antisymmetry.

## Frozen candidate

- family: `causal_antisymmetric_pairwise_advantage`
- target for pair `(i,j)`: `(interval_cost_i - interval_cost_j) /
  action_group_range`
- state parents: exact `PAIRWISE_CAUSAL_STATE_PARENTS`; every state parent must
  be numerically invariant across actions in its matched group
- action parents: exact `PAIRWISE_CAUSAL_ACTION_PARENTS`; model inputs use only
  `action_i - action_j`
- estimator: one `HistGradientBoostingRegressor` with learning rate 0.05,
  100 iterations, 15 leaves, minimum leaf size 20, and L2 penalty 10
- inference: antisymmetrize forward/reverse predictions and recover each action
  score by averaging its signed pairwise differences
- target data budget: zero groups and zero labels
- city ID and static target context are excluded from predictor features

There is no result-dependent blend, source expert gate, mechanism residual, or
hyperparameter grid in this stage.

## Data and integrity

- six development targets: grid4x4, Cologne1, Ingolstadt1, Atlanta 1x5,
  Hangzhou 4x4, and Manhattan 28x7
- immutable cache aggregate SHA-256:
  `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`
- immutable source-rule baseline SHA-256:
  `25e6291ba9ff7348642baf09498c9c7bea36ef3fd77a7ee6680b289a03201acb`
- Salt Lake remains sealed and may be opened only after this stage passes
- every target excludes its complete city group from fitting

## Frozen promotion gate

Promote the candidate only if all conditions hold:

1. city-macro normalized action regret improves by at least 10% versus the
   domain-normalized rigid reference;
2. at least four of six target cities improve;
3. maximum absolute city regression is at most 0.05;
4. all snapshot, cache, holdout, target-budget, causal-parent, antisymmetry,
   pair-count, and action-group checks pass.

No result-dependent hyperparameter change is permitted. Failure returns the
project to source-only architecture design; Salt Lake remains unopened.
