# TSC v43/r39 full-budget pre-evaluation amendment

Frozen at: `2026-08-09T05:49:04Z`

## Status

The v42/r38 seed-7079 result is retained as a failed confirmation. Its offline
gate decision is `retain_offline_failure_and_prohibit_closed_loop_claim`. Seed
7079 is permanently excluded from v43/r39 model selection and confirmation.

No seed-8171 counterfactual cache existed when this amendment was written.

## Protocol defect being corrected

The v42 pipeline selected only 100 target action groups per city even though
the frozen adaptation cache contains 165 valid Los Angeles groups and 911 valid
Jinan groups. It also deployed an ensemble of cross-validation models, each fit
on only 80 groups, instead of refitting after selection on the complete target
adaptation set. Finally, group-level folds mixed neighboring snapshots from the
same simulator seed across training and validation.

These are data-use and estimation-protocol defects. The correction does not use
the seed-7079 candidate ranking or city-level regret values.

## Frozen v43/r39 correction

1. Use every valid adaptation group from seeds 5057 and 6067; do not subsample
   to B100.
2. Select the already frozen v41 selector with two-fold
   leave-one-adaptation-seed-out validation. Each complete simulator seed is one
   validation fold, and each city is isolated from the other external city.
3. After OOF selection, refit method and baseline families once on the union of
   all adaptation groups. OOF models cannot be deployed.
4. Fit method and same-information baseline families in the same model call,
   with identical source evidence, target groups, static context, and refit
   boundary.
5. Freeze both model artifacts and a joint integrity audit before generating
   any seed-8171 data.
6. Apply the unchanged v42 offline gate on seed 8171. Closed-loop seeds 8081
   and 9091 remain inaccessible unless that gate passes.

The machine-readable protocol is
`cf_h2o/config/traffic_signal_tsc_v29_external_full_budget_confirmation.json`.
