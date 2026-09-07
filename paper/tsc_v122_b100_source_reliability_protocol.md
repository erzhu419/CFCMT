# V122 B100 Source-Reliability Diagnostic Protocol

## Question

V121 found that a symmetric aggregate of seven B100 source components was worse
than both the architecture-matched target-only arm and a source-permutation
placebo. V122 tests the narrower explanation that a small number of useful source
cities were obscured by negative-transfer sources in that symmetric aggregate.

## Frozen inputs

- Target city: Jinan.
- Adaptation budget: the exact 100 target groups selected by V115.
- Predictions: V120 strict five-fold out-of-fold target-only and one-source-at-a-
  time B100 component predictions.
- Reference policy and estimand: PhasePressure and the V114--V121 pure halted-
  queue waiting estimand.
- Selector labels: none. V122 is a training-set diagnostic only.

## Diagnostic

The analysis reports predictive and action-selection metrics for every source
city separately. It then forms top-k, inverse-MSE and softmax reliability
ensembles. For every held-out fold, source weights are fitted using only the
other four folds. Each source ensemble is blended with the architecture-matched
target-only score at four fixed weights.

A matched group-permutation placebo preserves source-score marginals and action
block dimensions while breaking target alignment. A strategy passes the
development gate only if its out-of-fold policy value improves on both target-
only and its matched placebo by at least 0.0005 normalized waiting units.

## Decision rule

Passing freezes exactly one reliability strategy for an independent V116
selector-cache test. Failure rejects B100 reliability weighting without opening
that selector cache. V122 cannot establish fresh-city efficacy and does not
replace the positive but differently matched V98 result.
