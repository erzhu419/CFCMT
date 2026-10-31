# TSC v37/r33 Leave-One-Group-Out Pairwise Gate

Date frozen: 2026-08-09. Written after v36/r32 failed and before any v37 result is generated. v35 and v36 remain immutable negative selector results. Salt Lake remains unopened.

## Fixed diagnosis

The v36 five-fold gate trained each OOF model on 48 target groups, while the final 60-group pairwise model improved over the equally informed fallback in all six development cities. v37 changes the capacity alignment, not the observed v36 thresholds: each OOF model is trained on 59 groups and predicts only the one omitted group.

## Frozen protocol

- Use the same deterministic 60 target simulator counterfactual action groups.
- For every group `g`, fit candidate and fallback on all source cities plus the other 59 target groups, then evaluate both on `g` only.
- Execute the 60 independent fits with 12 workers per target; concurrency cannot change group membership, model configuration, or output order.
- Concatenate exactly one leave-one-group-out prediction per group.
- Freeze the target family decision, refit both families on all 60 groups, and evaluate only on target groups outside the 60-group union.
- Candidate: `causal_antisymmetric_pairwise_advantage`.
- Fallback: `causal_group_normalized_rigid_advantage`.

## Per-target LOGO gate

Select pairwise only if:

1. all 60 groups have exactly one LOGO prediction from a 59-group fit;
2. mean paired gain is at least `0.01`;
3. pairwise is non-worse on at least 50% of LOGO groups;
4. the worst simulator-seed mean gain is at least `-0.02`.

Paired gain remains fallback normalized regret minus pairwise normalized regret. Any failed condition triggers the exact fallback.

## Development-stage success rule

Relative to fitting and deploying the fallback on all 60 target groups:

- city-macro normalized action regret improves by at least 5%;
- at least four of six cities select pairwise and strictly improve;
- maximum city-level absolute regression is at most `0.02`;
- every selected city satisfies all LOGO gates.

Passing freezes LOGO selection and the all-60-group refit for the one-time Salt Lake confirmation. Failure keeps Salt Lake sealed.
