# TSC V129 Shared-Residual Source-Null Feasibility Protocol

## Question

Does the aligned Atlanta prior add absolute and placebo-distinguishable value
when every prior arm receives exactly the same target-only causal residual?

## Change from V128

V128 independently fitted `actual - prior_arm` for target, aligned-source and
source-placebo arms. Abundant target labels could therefore cancel each prior
separately. V129 fits only `actual - target_prior` on the 11 training seeds.
That residual prediction is frozen and added to each candidate prior:

`score_arm = shared_target_residual + prior_arm`.

No source-specific target residual refit is permitted. This is the only method
change. The B500 prior, Atlanta weight 0.75, whole-action-group placebo,
pressure constraint, 10-seed calibration, held-out folds and final development
gate remain unchanged.

## Authorization

V129 may advance only if adaptive source-null:

1. Beats PhasePressure by at least 0.0005 with an upper 95% bound below zero.
2. Beats target-only by at least 0.0005 with an upper 95% bound below zero.
3. Beats the equally adaptive source-placebo null by at least 0.0005 with an
   upper 95% bound below zero.

This remains an abundant-target action-ranking feasibility experiment. A pass
would authorize, but would not substitute for, a total target-information
budget curve and untouched-city confirmation.
