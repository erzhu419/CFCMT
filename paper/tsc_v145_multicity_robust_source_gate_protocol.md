# TSC V145 Robust Source-Gate Development Protocol

## Status and purpose

V145 is post-V144 method development. V144 established that every target city
has at least one useful one-source arm, but its oracle identities are not
deployable. Static city-signature nearest-neighbour and pairwise ridge
selectors were rejected during development because they produced material
negative transfer. V145 therefore tests a smaller robust-history gate with an
explicit pressure-reference intervention veto.

A V145 pass is not confirmation. It may only authorize a new protocol whose
target city was untouched by V145 development.

## Information budget

The fitted one-source models retain the frozen B25 target adaptation budget.
V145 is therefore a few-shot target-adaptation experiment, not strict
zero-shot transfer.

For each target, source intervention fractions are computed only on the same
B25 adaptation action groups. The diagnostic compares the model-selected row
with the PhasePressure reference row. It does not read evaluation features,
next-state labels, normalized policy outcomes or counterfactual costs.

When target city `h` is evaluated, every meta-label involving `h` is removed,
including rows where `h` is the source. Target `h` contributes only its B25
fitted-model action diagnostic.

## Frozen selector

For candidate source `q`, the history contains `q -> r` effects for the five
remaining meta cities. A robust source must satisfy all of the following:

- maximum regression versus target-only no greater than 0.005;
- 75th-percentile effect versus target-only no greater than -0.0005;
- 75th-percentile effect versus the matched source placebo no greater than
  -0.0005.

Among robust sources, V145 selects the lowest historical mean effect versus
target-only. If none qualifies, it selects the source with the smallest B25
adaptation intervention fraction. If even that minimum exceeds 0.90, the gate
returns exact source-null. The 0.005 history allowance is a factor-of-two
development safety buffer relative to the 0.01 city-regression gate.

## Development decision

The selected gate must separately improve target-only CFCMT, pooled H2O+ and
the selected source's whole-action-group permutation placebo. Each comparison
requires mean effect at most -0.0005, a negative one-sided 95% city-level upper
bound, at least five improving cities, and no city regression above 0.01.

Uniform CFCMT and PhasePressure remain descriptive comparisons. V145 does not
claim universal superiority over PhasePressure, and a development pass cannot
be substituted for an untouched-city result.
