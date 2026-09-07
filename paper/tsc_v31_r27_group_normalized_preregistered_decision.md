# TSC v31/r27 Preregistered Decision: Matched-Group Regret Normalization

Date frozen: 2026-08-09

## Motivation

The immutable offline screen reports regret after dividing each matched action
group by its own counterfactual cost range. The existing rigid causal core is
instead fitted after dividing labels by one robust scale per source city. This
experiment changes only that estimand alignment. It does not add graph parents,
mechanism heads, target labels, model capacity, or a deployment guard.

## Frozen candidate

- family: `causal_group_normalized_rigid_advantage`
- parents: exact `CAUSAL_RIGID_RANKING_PARENTS`
- estimator: the existing candidate-only, sign-balanced
  `HistGradientBoostingRegressor`
- target: action-minus-source-rule `interval_cost`
- target scale: each source matched action group's `action_group_range`
- target data budget: zero groups and zero labels
- hyperparameters: unchanged `ActionAdvantageConfig` defaults

The frozen comparator is `causal_rigid_advantage`, whose only relevant
difference is city-level target normalization.

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

1. city-macro normalized action regret improves by at least 10% versus rigid;
2. at least four of six target cities improve;
3. maximum absolute city regression is at most 0.05;
4. all snapshot, cache, holdout, target-budget, parent-set, and row-accounting
   checks pass.

No result-dependent hyperparameter change is permitted. If the gate fails, the
next admissible module is an all-action antisymmetric pairwise preference loss
using the same matched-group normalization; it must be preregistered and tested
as a separate stage.
