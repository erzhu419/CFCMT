# V125 Source-Intervention Feasibility Result

## Status

V125 completed under immutable snapshot `e16994b5b89eb29b8471` in 158.26 s.
The result SHA-256 is
`8749c85ffb864c32f96ab29d786f85eac25f7db14843bfc493b005b234ef22c6`.
It is a Jinan post-V124 development diagnostic, not confirmation.

## Oracle headroom

The unrestricted eight-action oracle improved all 22 held-out selector seeds.
Its mean normalized waiting-cost difference from PhasePressure was
`-0.40979` (paired 95% bootstrap CI `[-0.41817, -0.40086]`) and it changed
80.70% of decisions.

The pressure-nondegrading oracle also improved all 22 seeds while allowing only
actions whose instantaneous service pressure was no lower than PhasePressure.
Its mean difference was `-0.07709` (95% CI `[-0.08067, -0.07349]`), with
3,359 interventions over 19,851 decision groups (16.92%). Of those selected
interventions, 99.88% had negative observed cost and none had positive cost.

This rules out an action-space glass ceiling: large safe-action headroom exists
inside the current counterfactual bank. The remaining bottleneck is identifying
those actions out of sample.

## Held-out V124 guards

The B50 source candidate intervened 95 times and had mean held-out difference
`+0.0000542` (95% CI `[-0.0000851, +0.0001986]`). Its matched target guard was
also inconclusive at `+0.0000260`.

The B500 source guard is the decisive failure. It had appeared beneficial on
the training folds, but its 13 held-out interventions produced mean difference
`+0.0001315` with 95% CI `[+0.0000224, +0.0002657]`. It therefore harmed
PhasePressure significantly out of sample. All other budget guards either
fell back completely or failed authorization.

## Decision

V124 must not be rescued by lowering its minimum-effect threshold: the only
apparently promising guard overfit and reversed sign out of sample. V125 instead
authorizes development of a new action ranker under the fixed
pressure-nondegrading action set. The next experiment first tests learnability
with abundant target counterfactual labels; only a successful upper-feasibility
result can justify a target-budget curve and conservative runtime integration.
