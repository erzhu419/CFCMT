# V152A Hierarchical Mechanism Prior Result

## Decision

`REJECT`. The frozen leave-target-city-out selector admitted no source prior in
any of the seven development cities. Every deployed arm therefore recovered
rigid target-only CFCMT exactly. This is a scientific rejection, not an
execution failure: all target jobs and all information-isolation checks passed.

## Evidence

The forced robust prior improved Atlanta (`-0.005804`), Cologne (`-0.001032`)
and RESCO synthetic (`-0.001815`) relative to rigid CFCMT. It regressed the
other four cities, most strongly Hangzhou (`+0.022101`). The city-macro forced
effect was `+0.002769`, with city-bootstrap interval
`[-0.002079, +0.009796]`; negative values are better.

Each evaluated target considered 30 fixed mechanism-block/strength candidates.
The candidate selector used six pseudo-target cities, with the evaluated target
excluded from every pseudo-target label, source pool and feature-scaling role.
Every pseudo-target prior contained exactly five source cities. No candidate
met the frozen requirements against rigid, matched placebo and the
zero-centred prior in at least five of six pseudo-target cities.

The source-null arm was exactly equal to rigid CFCMT in all seven cities.

## Interpretation

V152A rejects the hypothesis that a target-label-free equal-city median
coefficient prior can be selected safely from the other six development cities.
Several candidates had positive mean meta gain, but the sign was not stable
across cities and did not predict the held-out target reliably. This result does
not negate the established rigid/action-contrast advantage or the target-local
V150K development result.

The next bounded experiment is a few-shot target-adaptation test. It retains the
same source prior, candidate grid and controls, but permits the fixed B100 target
adaptation groups to select the mechanism block and strength by complete
out-of-fold evaluation. Target-only receives the identical B100 labels, and the
untouched target evaluation groups remain unavailable to selection.

## Runtime And Artifacts

Reusable source statistics and vectorized action-group indexing reduced the
Atlanta runtime from `1130.57 s` to `156.66 s` (`7.22x`) with exact equality of
all scientific result fields.

- Aggregate: `cf_h2o/results/paper_artifacts/tsc_v152a_hierarchical_mechanism_prior.json`
- Atlanta optimized result: `cf_h2o/results/cluster/tsc_v152_hierarchical_mechanism_prior_20260908/atlanta_preflight_v2_optimized/atlanta/result.json`
- Six-city matrix: `cf_h2o/results/cluster/tsc_v152_hierarchical_mechanism_prior_20260908/development_v1_optimized/`
- Frozen protocol: `paper/tsc_v152a_hierarchical_mechanism_prior_protocol.md`
