# TSC V150I Sample-Coherent Source-Prior Result

## Status

Partial repair; development gate rejected. All 21 jobs completed under snapshot
`b8410b7f290ab3027bbf`. The aggregate artifact is
`cf_h2o/results/paper_artifacts/tsc_v150i_sample_coherent_source_prior.json`
with SHA-256
`9f4c3e522d06f08f939c01b467761633cce2aebde962caf9ef24032c88bbde0e`.

## Results

Negative paired effects are better.

| Budget | Prior strength | Admitted cities | Macro vs rigid | Macro vs matched placebo |
|---:|---:|---:|---:|---:|
| B25 | `0.100` | 1/7 | `-0.000438` | `-0.000237` |
| B50 | `0.050` | 2/7 | `+0.003765` | `-0.000997` |
| B100 | `0.025` | 1/7 | `-0.005234` | `-0.000008` |

At B100, only Atlanta admitted a source. Its Cologne network-propagation prior
improved the common untouched evaluation set by `-0.036637` versus rigid and by
`-0.0000583` versus its matched placebo. The other six cities returned exact
rigid fallback. Thus B100 was non-degrading in 7/7 cities, unlike V150H, but did
not meet the requirement that at least two cities independently admit useful
source information.

B50 still admitted a harmful Hangzhou queue-service prior for Ingolstadt
(`+0.040703` versus rigid), so the inverse-budget schedule is not sufficient at
all information levels.

## Decision

The objective correction is retained: target evidence should weaken a source
prior when target likelihood is normalized. It materially improved the primary
B100 safety behavior and should not be reverted. However, V150I does not
authorize source-aware closed-loop development because its remaining benefit is
one-city and almost indistinguishable from the matched placebo.

The next test must change the mechanism representation, not tune another prior
strength. It should add only deployment-observable interactions between action
service, protected/permissive right-of-way, shared receivers and neighboring
signal execution, with the same target-only and placebo controls.
