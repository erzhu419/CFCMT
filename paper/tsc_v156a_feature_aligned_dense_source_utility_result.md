# V156A Feature-Aligned Dense Source-Utility Result

## Result

The correction-only replay rejects both the seven-city source-selector claim
and the predeclared city-specific follow-up route. Resolving rigid-model inputs
by stored feature name changes the admitted set from Atlanta and New York in
V150O to Hangzhou alone in V156A. Hangzhou then worsens on the untouched B100
reserve against every required comparator:

| Target | V150O admitted | V156A admitted | Source minus rigid | Source minus placebo | Source minus source-blind |
|---|---:|---:|---:|---:|---:|
| Atlanta | yes | no | 0 | 0 | 0 |
| Hangzhou | no | yes | +0.00942188 | +0.00061303 | +0.00092563 |
| New York | yes | no | 0 | 0 | 0 |
| Other four cities | no | no | 0 | 0 | 0 |

Positive differences mean higher normalized cost. The global gate therefore
fails with only one cross-fitted admission and a reserve regression in that
city. The city-specific gate also has no candidate: Hangzhou fails all three
reserve comparisons, while every rejected city exactly recovers target-only
rigid. Excluding an unsuccessful city cannot retain the old transfer result
because no corrected city passes the complete single-city rule.

## Interpretation

The earlier New York source-specific benefit does not survive the V154 feature
binding correction. Atlanta's earlier admission also disappears. V156A thus
closes the dense pressure-pairwise source-selector family; it does not justify
an untouched-seed city-specific closed loop or a Bologna external test.
PhasePressure remains the untrained rule benchmark, and this result concerns
source identification rather than the validity of the entire traffic-signal
project.

## Execution and provenance

The first launch from the current workspace snapshot stopped before fitting
because its newer occupancy-cache validator rejected the frozen V150O cache;
it produced no scientific result. Tasks `t92470`--`t92476` then completed from
the original V150O snapshot with only
`PairwiseActionAdvantageRegressor._features` changed to resolve stored names.
The cached counterfactual costs were reused and the derived utility records
were recomputed; no SUMO trajectory or runtime model was produced.

All seven source-bank, split, manifest, fit-protocol, prerequisite-result and
snapshot invariants pass. Summed runner time was 2,505.084 seconds. The seven
retrieved result JSON files total 601,208 bytes; the aggregate records
the launch, task IDs and corrected/legacy file identities. Twenty-one V156A
contract tests and two focused feature-binding tests pass. The authoritative
artifact is
`cf_h2o/results/paper_artifacts/tsc_v156a_feature_aligned_dense_source_utility.json`.

## Decision

Retain the negative result and stop further threshold, representation or label
expansion within this dense source-selector family. Do not promote V150O's New
York result or use post-hoc city deletion to claim transfer success.
