# TSC V133 Historical Action Identifiability Protocol

## Purpose

V125 proves that a pressure-nondegrading action oracle has substantial
counterfactual headroom, while V130--V132 cannot identify stable interventions
from current-state models or generalized-pressure mechanisms. V133 tests
whether the 450-second action effect itself repeats across stochastic simulator
realizations before another model is built.

## Frozen protocol

Each outer fold reserves one of 22 selector seeds. The remaining 21 seeds are
aligned by scenario, control-interval index, TLS identifier and exact candidate
signal state. A candidate history must exist, and remain pressure-nondegrading,
in at least 80% of the development seeds.

The diagnostic mean-lookup arm accepts a candidate when its historical mean
normalized improvement is at least 0.0005. The primary conservative arm also
requires improvement in at least 80% of history seeds and a one-sided 95%
upper t bound below zero. Among admitted actions it chooses the smallest upper
bound. The held-out action label is read only after this lookup and selection
are complete. Missing or unsupported keys fall back exactly to PhasePressure.

## Decision

The primary arm passes only when its 22 outer held-out effects have mean at
most `-0.0005` and upper 95% bound below zero. A primary pass licenses a local
causal-dynamics successor. A mean-only pass indicates historical signal but
insufficient uncertainty control. Failure of both arms means exact
schedule/TLS/action history does not identify the 450-second effect across
realizations; another direct action regressor is then unjustified without
changing the target or adding genuinely predictive dynamics.
