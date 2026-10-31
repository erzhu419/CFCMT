# TSC V150C Mechanism-Parameter Prior Result

## Decision

V150C passed its six frozen development checks on the seven development-city
bank. The selected mechanism prior improved the same-architecture target-only
model by a city-macro mean of `-0.00408763`; six of seven city means improved,
the 21 seed-unit bootstrap 95% interval was `[-0.00652307, -0.00170442]`, and
the largest city regression was `+0.00049436` in Hangzhou. All seven selectors
admitted a source prior. Zero source strength reproduced target-only
coefficients, evaluation scores, and summaries exactly.

This result authorizes rigid-CFCMT integration under the frozen V150C protocol.
It does not establish a closed-loop or untouched-city source-transfer claim.

## Source-identity sensitivity

The source arm beat its independently selected matched-placebo arm by only
`-0.00025924` on average. The corresponding 21 seed-unit bootstrap interval was
`[-0.00157703, +0.00088915]`, only eight of 21 seed units improved, and only
three of seven city means improved. The frozen protocol required only a
negative placebo mean, so the formal V150C decision remains PASS; nevertheless,
source identity is not yet strongly isolated from the benefit of the mechanism
architecture and regularization.

V150D therefore transfers only the source-induced mechanism score difference
into rigid CFCMT, requires exact rigid fallback, treats city as the inferential
unit, and adds a same-source/same-block placebo confidence gate before any
closed-loop source claim is launched.

## Evidence

- Aggregate: `cf_h2o/results/paper_artifacts/tsc_v150c_mechanism_parameter_prior.json`
- Per-city results: `cf_h2o/results/cluster/tsc_v150_source_identifiability_20260908/mechanism_parameter_prior_v1/`
- Frozen protocol: `paper/tsc_v150c_mechanism_parameter_prior_protocol.md`
