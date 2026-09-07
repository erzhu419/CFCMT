# TSC v36/r32 Cross-Fitted Target Pairwise Gate

Date frozen: 2026-08-09. This document is written after v35/r31 failed and before any v36 out-of-fold or evaluation result is generated. v35 remains an immutable negative result. Salt Lake remains unopened.

## Motivation fixed before execution

The v35 model still beat its fallback on all six external development targets, but its 24-group calibration-only gate selected pairwise on zero targets. The failure came from sparse action disagreement and high block variance. Threshold relaxation on the observed v35 rows is prohibited.

v36 replaces the inefficient fixed adaptation/calibration split with cross-fitting. Every target simulator group contributes either to a fold's training set or to that fold's strictly out-of-fold decision evidence, never both. The final target evaluation remains outside the entire 60-group information budget.

## Frozen protocol

- Target information budget: the same deterministic 60 complete action groups used by v34/v35.
- Candidate: `causal_antisymmetric_pairwise_advantage`.
- Fallback: `causal_group_normalized_rigid_advantage`.
- Five target-group folds, stratified by simulator seed and assigned round-robin after deterministic time/TLS ordering.
- For each fold, fit both families on all source cities plus the other 48 target groups and evaluate paired normalized regret on the 12 held-out target groups.
- Concatenate the five fold predictions to obtain exactly one out-of-fold decision for each of the 60 groups.
- After the family decision is frozen, fit both families on all 60 target groups and evaluate on every unselected target group.

## Per-target OOF gate

Define OOF paired gain as fallback regret minus pairwise regret. Select pairwise only if:

1. all 60 groups have exactly one OOF prediction;
2. mean OOF paired gain is at least `0.01`;
3. pairwise is non-worse on at least 50% of OOF groups;
4. the worst simulator-seed mean gain is at least `-0.02`;
5. the worst fold mean gain is at least `-0.03`.

Otherwise deploy the exact fallback. No fold count, threshold, estimator, parent set, or family may change after v36 starts.

## Development-stage success rule

Relative to fitting and deploying the fallback on all 60 target groups, the selected final policy passes only if:

- city-macro normalized action regret improves by at least 5%;
- at least four of six cities select pairwise and strictly improve;
- maximum city-level absolute regression is at most `0.02`;
- every selected target satisfies all OOF gates.

Passing freezes the full cross-fitted selector and all-60-group refit for one-time Salt Lake confirmation. Failure keeps Salt Lake sealed.
