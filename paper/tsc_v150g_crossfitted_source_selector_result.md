# TSC V150G Complete-Selector Crossfit Result

## Status

Reject. All seven target jobs completed, but the frozen development gate did
not authorize source-aware closed-loop development. The aggregate artifact is
`cf_h2o/results/paper_artifacts/tsc_v150g_crossfitted_source_selector.json`
with SHA-256
`20820aa9ce5bd016b5e650f346c4d3aefd4ea07edc139d19072bf53785fb0fc8`.

## Result

Only Hangzhou passed both levels of selection: the candidate selected on all
B25 groups and the complete-selector leave-one-fold evaluation. Its selected
Atlanta spillback prior improved the untouched evaluation groups by `0.006398`
versus rigid target-only and by `0.004212` versus its matched placebo. The other
six cities returned the bitwise-exact rigid fallback.

Across seven city units, the deployed selector therefore changed only one city:

| Comparison | Macro paired effect | City bootstrap 95% CI | Improving cities |
|---|---:|---:|---:|
| selected source - rigid target-only | `-0.00091399` | `[-0.00274196, 0]` | 1/7 |
| selected source - matched placebo | `-0.00060177` | `[-0.00180531, 0]` | 1/7 |

Negative effects are better. No city regressed because failed selectors used
exact fallback, but the preregistered requirement that at least two cities admit
an identity-verified source was not met.

## Interpretation

V150G removes V150E's selection-layer reuse: each held-out fold is evaluated
using a source/mechanism choice made on the other four folds. The result shows
that the apparent B25 source identities generally do not survive that test.
This is not evidence that source data have no value: V150A repeated relative
source headroom and V150C showed a small same-architecture prior benefit. It is
evidence that B25 does not support a generally deployable 36-candidate identity
selector across these seven development cities.

The next experiment may vary the prespecified target-information budget while
holding the representation, candidate set, complete-selector crossfit and a
common evaluation set fixed. Closed-loop source claims remain unauthorized
until that diagnostic passes a multicity gate.
