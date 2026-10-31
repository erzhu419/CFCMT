# TSC V150E Identity-Verified Mechanism-Prior Result

## Decision

V150E was rejected. The stricter B25 selector admitted a source in five of seven
development cities, but the resulting seven-city macro effect relative to rigid
CFCMT was `+0.00829202`, with a city-bootstrap interval of
`[-0.00684388, +0.03320622]`. Only three city means improved; Atlanta regressed
by `+0.07900808` and New York by `+0.00416202`. Cologne and Ingolstadt rejected
all candidates and recovered rigid CFCMT exactly.

The selected arm's macro effect relative to its same-candidate permutation
placebo was only `-0.00008861`, with interval
`[-0.00164856, +0.00116024]`. Only Atlanta and Hangzhou beat the matched placebo
on untouched evaluation groups. Thus requiring positive mean OOF margins against
both controls is insufficient at B25.

## Interpretation

V150E rules out the simple explanation that V150D failed only because candidates
were allowed to tie their placebo. Mean five-fold OOF gains remain too unstable
to identify deployment-safe source corrections. The evidence points to noisy
action-utility labels and candidate-selection variance rather than a missing
fallback: rejected cities already use an exact rigid fallback.

No source-aware closed-loop run is authorized from V150E. The next diagnostic
separates fixed decision state from future SUMO randomness and estimates each
action-cost label over multiple independent future seeds before another selector
is developed.

## Evidence

- Aggregate: `cf_h2o/results/paper_artifacts/tsc_v150e_identity_verified_mechanism_prior.json`
- Per-city results: `cf_h2o/results/cluster/tsc_v150_source_identifiability_20260908/identity_verified_mechanism_prior_v1/`
- Frozen protocol: `paper/tsc_v150e_identity_verified_mechanism_prior_protocol.md`
