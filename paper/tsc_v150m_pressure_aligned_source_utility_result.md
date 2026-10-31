# V150M Pressure-Aligned Source Utility Result

## Decision

V150M is rejected before closed-loop rollout. All seven target cities returned
the exact rigid CFCMT fallback; no runtime source intervention was admitted.

## Diagnosis

The action constraint behaved as specified. On each evaluation reserve, raw
source-mechanism candidates did sometimes select the PhasePressure reference:
the candidate-group counts ranged from 3 in Atlanta to 1,629 in Hangzhou.
However, the B100 nested adaptation folds supplied too few cross-fitted
pressure-aligned disagreements to fit the fixed utility gate. Final
cross-fitted record counts were 0, 15, 10, 4, 0, 1, and 2 for Atlanta, Cologne,
Hangzhou, Ingolstadt, New York, RESCO synthetic, and Salt Lake City,
respectively, all below the predeclared minimum of 48.

This is an information-support failure, not a closed-loop performance result.
Lowering the minimum-record gate after observing these counts would be a
post-hoc protocol change and is not used.

## Next Bounded Hypothesis

V150N retains the binary safe action set but changes the migration object. A
source mechanism no longer has to make PhasePressure the global argmin among
all feasible phases. It supplies only the pairwise preference between the
current rigid action and the PhasePressure reference. This directly matches the
action-contrast estimand and should create a denser, interpretable supervision
set without allowing a third action.

Canonical aggregate:
`cf_h2o/results/paper_artifacts/tsc_v150m_pressure_aligned_source_utility.json`.
