# TSC V150H Source-Identifiability Budget Curve Result

## Status

Reject. All 21 city-budget jobs completed under snapshot
`4cd9b4b91fddbad7959f`. The aggregate artifact is
`cf_h2o/results/paper_artifacts/tsc_v150h_source_identifiability_budget_curve.json`
with SHA-256
`719b99dfc2bd89b0e0a9fdda7cfd1390598cc10e25c4dfcafb0dc657e1b5a60f`.

## Results

All budgets used nested target-labelled groups and the same evaluation groups
outside the B100 reserve. Negative paired effects are better.

| Budget | Admitted cities | Macro vs rigid | City 95% CI | Macro vs matched placebo | City 95% CI |
|---:|---:|---:|---:|---:|---:|
| B25 | 1/7 | `-0.000438` | `[-0.001314, 0]` | `-0.000237` | `[-0.000712, 0]` |
| B50 | 2/7 | `+0.004281` | `[-0.003791, +0.016634]` | `+0.001141` | `[0, +0.002657]` |
| B100 | 2/7 | `-0.004756` | `[-0.015755, +0.001486]` | `+0.000771` | `[-0.000065, +0.002378]` |

At B50, the admitted Hangzhou queue-service prior regressed Ingolstadt by
`+0.038814`. At the primary B100 budget, the Salt Lake City network-propagation
prior improved Atlanta by `-0.036762`, but the Salt Lake City execution prior
regressed Cologne by `+0.003467` and was worse than its matched placebo by
`+0.005549`. Exact fallback protected all non-admitted cities.

## Interpretation

The admission count rose from one to two, but more target labels did not make
the selected source identities reliable on untouched states. The failure is
therefore not explained by B25 sample size alone. It also exposes a concrete
objective mismatch: `group_balanced_weights` normalizes the target likelihood
to unit mass at every budget, while the source-prior penalty remains `0.10`.
Consequently, B100 reduces sampling variance but does not let target evidence
overpower the source prior as intended by the method.

The next bounded repair should make source-prior precision decrease with target
group count while preserving the same target-only comparator, complete-selector
crossfit, matched placebo and common evaluation set. V150H does not authorize a
source-aware closed-loop experiment.
