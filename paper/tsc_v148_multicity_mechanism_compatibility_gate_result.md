# V148 Multicity Mechanism-Compatibility Gate Result

## Decision

V148 is rejected under its frozen seven-city development gate. The result does
not authorize V149 or any replacement confirmation target.

The corrected target tasks were `t89229` and `t89263`--`t89268`. All seven
complete-city result contracts passed. The earlier Atlanta task `t89226` ended
before producing a result because of an implementation error and is not
scientific evidence.

## Information boundary

V148 is target-offline few-shot adaptation, not zero-shot transfer. For each
target, the B25 adaptation groups are split into five folds; B20 fits the fixed-
parent rank-0 mechanism residuals and disjoint B5 next-state labels score
mechanism trust. Evaluation groups are excluded from fitting, trust estimation
and source selection.

The five mechanism diagnostics are queue propagation, served movement, red
accumulation, spillback and mobility. They estimate only the mass assigned to a
candidate source residual. They never select an action. The downstream action
model is the V143 rigid causal ranker, not the full multi-mechanism MC-WM
controller.

## Frozen aggregate

The leave-one-target-and-source history gate admitted no source for any of the
seven targets. Atlanta, Cologne, Hangzhou, Ingolstadt, New York, RESCO synthetic
and Salt Lake City therefore all returned exact source-null, which is identical
to the architecture-matched target-only causal arm.

| Comparator | Mean paired effect | One-sided 95% upper bound | Improving cities | Maximum city regression | Gate |
|---|---:|---:|---:|---:|---|
| target-only causal | 0.000000 | 0.000000 | 0/7 | 0.000000 | fail |
| matched source placebo | 0.000000 | 0.000000 | 0/7 | 0.000000 | fail |
| pooled H2O+-style residual | -0.069141 | -0.029837 | 7/7 | 0.000000 | pass |

Negative values favour the V148 selected arm. The apparent H2O+ improvement is
the already-supported target-only rigid causal advantage; it is not evidence
that mechanism compatibility recovered useful cross-city source contribution.

For context, V145 selected a source in six cities but failed target-only safety
(mean -0.002828, one-sided upper bound +0.009293, 4/7 improving, maximum city
regression +0.028104). V146 selected a source in all seven cities and likewise
failed target-only safety (mean -0.001612, one-sided upper bound +0.009510, 3/7
improving, maximum city regression +0.026222). V148 removed those regressions by
abstaining everywhere, but it did not establish source value.

## Claim boundary

V148 supports a conservative negative-transfer result: under the frozen robust
history criterion, labeled local mechanism fit was insufficient to certify any
source city. It does not support a full MC-WM benefit, general source-selection
success, zero-shot transfer or superiority to PhasePressure.

Canonical aggregate:
`cf_h2o/results/cluster/tsc_v148_multicity_mechanism_compatibility_gate_20260901/development_v2_corrected/aggregate.json`.
