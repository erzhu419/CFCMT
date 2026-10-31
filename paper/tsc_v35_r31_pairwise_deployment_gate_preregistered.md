# TSC v35/r31 Calibration-Only Pairwise Deployment Gate

Date frozen: 2026-08-09. This document is written after the v34/r30 development curve passed and before any v35 calibration result is evaluated. v35 is a development-stage deployment selector; Salt Lake remains unopened.

## Fixed method and information budget

- Target counterfactual budget: 60 complete matched action groups.
- Model-adaptation groups: 36.
- Policy-selection calibration groups: 24, all disjoint from model fitting.
- Evaluation: every target group outside the fixed 60-group union.
- Candidate: `causal_antisymmetric_pairwise_advantage`.
- Exact fallback: `causal_group_normalized_rigid_advantage`.

Both models use the same five source city domains and the same 36 target adaptation groups. Calibration outcomes cannot update either model. The selected policy family is evaluated only after the target-specific decision is frozen.

## Per-target calibration gate

For each calibration action group, compute normalized regret for both families and define paired gain as fallback regret minus pairwise regret. Positive gain favors pairwise. Select pairwise only if all conditions hold:

1. At least 8 complete calibration groups are available.
2. Mean paired gain is at least `0.01`.
3. The one-sided 95% normal lower confidence bound, `mean - 1.645 * sample_sd / sqrt(n)`, is at least `-0.01`.
4. Pairwise is non-worse on at least 50% of calibration groups.
5. Sort groups by snapshot time, divide them into four contiguous balanced blocks, and require the worst block mean gain to be at least `-0.05`.

If any condition fails, deploy the exact fallback. No interpolation weight or threshold may be tuned from v35 evaluation outcomes.

## Development-stage success rule

Relative to deploying the fallback on every target, the calibration-gated policy passes only if:

- city-macro normalized action regret improves by at least 5%;
- at least four of six cities select pairwise and strictly improve;
- maximum city-level absolute regression is at most `0.02`;
- every selected target's recorded calibration decision satisfies all frozen gate conditions.

Passing v35 freezes this selector for the one-time Salt Lake confirmation. Failure keeps Salt Lake sealed and rejects calibration-only deployment of the pooled pairwise model.
