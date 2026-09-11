# TSC V150H Source-Identifiability Budget Curve Protocol

## Question

V150G admitted an identity-verified source in only one of seven cities at B25.
V150H tests the bounded explanation that the complete selector lacks target
information, rather than changing its representation, candidates or thresholds.

## Frozen Comparison

The target-labelled budgets are B25, B50 and B100 action groups. Selection is
scenario-seed balanced and nested: B25 is a subset of B50, which is a subset of
B100. Every arm at every budget is evaluated on the same groups outside the
B100 reserve. Groups reserved for a larger budget are neither training nor
evaluation data for a smaller budget.

At each budget, V150H preserves V150G's rigid CFCMT architecture, 36
source/mechanism candidates, five-fold complete-selector crossfit, matched
placebo, `0.0005` target and identity margins, four-of-five non-degradation rule,
and bitwise-exact source-null fallback. The architecture-matched target-only arm
receives exactly the same target-labelled groups as the source-prior arm.

## Primary Decision

B100 is the prespecified primary budget. It passes only if at least two cities
admit a cross-fitted source, no city regresses rigid CFCMT, every admitted city
improves both rigid and its same-candidate placebo on the common evaluation set,
every rejected city falls back exactly, and the city-macro effects improve both
comparators. B25 and B50 describe the information curve and cannot substitute
for a failed B100 gate.

A pass authorizes a B100 source-aware closed-loop development experiment. It is
not fresh-city confirmation. B25/B50/B100 are target-offline adaptation and are
not described as zero-shot.
