# V127 Causal Pairwise Source-Prior Feasibility Protocol

## Motivation

V125 established substantial pressure-nondegrading oracle headroom. V126 then
showed that a high-dimensional linear ranker cannot recover it and that treating
all source channels as ordinary predictors creates negative transfer. Earlier
TSC development also showed that target-adapted antisymmetric pairwise models
can improve action ranking, while post-hoc deployment selectors are fragile.

## Frozen method

V127 uses the V123 B500 target and source predictions. V123 selected
`atlanta__source_weight_0.75` in all 22 leave-one-selector-seed folds, so V127
freezes Atlanta and exposes only the incremental causal prior
`0.75 * (Atlanta score - target score)`. The target-only arm receives zero in
the same channel. The placebo arm receives the same source-prior vectors after
deterministic within-seed whole-action-group permutation.

Each arm uses the existing antisymmetric pairwise HGB with the registered
causal state and action parents. It cannot read city ID, arbitrary dense local
context, or forbidden residual parents. Exact PhasePressure remains in every
action set, and actions with lower instantaneous service pressure are excluded.

## Information split and gate

For each of 22 held-out selector seeds, five other seeds are reserved for
intervention calibration and the remaining 16 fit the ranker. The held-out seed
is used for neither role. Calibration chooses among frozen retention fractions
from 0.5% to 20%. A profile is enabled only with at least 20 interventions,
negative mean at least 0.0005, negative one-sided 95% upper bound, and negative
seed value in at least four of five calibration seeds. Otherwise deployment is
exact PhasePressure.

The source arm must beat PhasePressure, target-only and source-placebo by at
least 0.0005 mean normalized cost, with all paired 95% bootstrap upper bounds
below zero.

## Claim boundary

The ranker consumes 16 full target counterfactual seed banks per outer fold and
five more for calibration. V127 is therefore an abundant-information
upper-feasibility experiment, not few-shot target adaptation and not fresh-city
confirmation. Passing can justify a target-budget curve; failure rejects this
source-prior ranker family.
