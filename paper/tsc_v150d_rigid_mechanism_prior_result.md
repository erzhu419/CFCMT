# TSC V150D Rigid Mechanism-Prior Integration Result

## Decision

V150D was rejected. Adding the B25-selected source/mechanism correction to rigid
CFCMT changed the seven-city macro mean by `+0.00478435`, where positive is
worse. The city-bootstrap interval was `[-0.00690674, +0.02291960]`; four of
seven city means improved, and Ingolstadt regressed by `+0.05590506`. The frozen
gate therefore retained rigid CFCMT and prohibited source-aware closed-loop
development.

The source-null implementation exactly reproduced rigid CFCMT, so this rejection
does not weaken the established target-only structural result.

## Source Identity

Against the same-source, same-mechanism permutation placebo, the selected source
arm changed the city-macro mean by `-0.00069937`. Its city-bootstrap interval
crossed zero (`[-0.00247931, +0.00098461]`). Four of seven city means beat the
matched placebo, but the result did not establish a stable source-identity effect.

The diagnostic pattern is specific:

- Atlanta, Ingolstadt, New York and Salt Lake City selected candidates whose OOF
  source score was tied with its matched placebo.
- Hangzhou was the clean positive case: the source beat placebo during OOF
  selection and improved untouched evaluation groups.
- RESCO synthetic improved rigid but lost slightly to the matched placebo on
  untouched evaluation groups.
- Ingolstadt supplied the decisive failure: the source/placebo OOF tie preceded
  the largest rigid regression.

V150E was therefore restricted to candidates that beat both target-only and the
same-candidate placebo before admission.

## Evidence

- Aggregate: `cf_h2o/results/paper_artifacts/tsc_v150d_rigid_mechanism_prior_integration.json`
- Per-city results: `cf_h2o/results/cluster/tsc_v150_source_identifiability_20260908/rigid_mechanism_prior_v1/`
- Frozen protocol: `paper/tsc_v150d_rigid_mechanism_prior_integration_protocol.md`
