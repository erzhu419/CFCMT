# TSC V150G Complete-Selector Crossfit Protocol

## Motivation

V150E cross-fitted each candidate model but used the same 25 OOF action groups
to choose among 36 source/mechanism candidates and to decide admission. Its large
Atlanta regression is consistent with winner's curse at the selection layer.
V150F found that future-seed noise is real but generally smaller than variation
between decision states, so simply averaging more simulator futures is not a
complete fix.

## Selector

V150G retains the V150E representation, mechanism priors, prior strength, B25
budget, five folds, matched placebo and untouched evaluation groups. It changes
only admission by cross-fitting the complete selection procedure.

For each held-out B25 fold:

1. use the other four folds to choose one source/mechanism candidate;
2. apply the existing target-only and same-candidate-placebo margins only on
   those four folds;
3. evaluate that fixed selection on the omitted fold;
4. use exact rigid fallback when the four-fold selector rejects every candidate.

The five held-out gains are then combined. Crossfit passes only when the mean
gain is at least `0.0005` against both rigid target-only and matched placebo, and
at least four of five folds are non-degrading against each comparator. The
deployment candidate is selected using all B25 groups, but it is admitted only
if both its full-data identity check and the complete-selector crossfit pass.

## Development Gate

The city is the inferential unit. At least two cities must admit a source; no
city may regress rigid CFCMT; every admitted city must improve both rigid and its
matched placebo; every rejected city must reproduce rigid exactly; and the
seven-city macro mean must improve both controls.

This is a post-V150E development experiment on the same seven cities. A pass can
authorize closed-loop development but cannot establish fresh-city confirmation.
