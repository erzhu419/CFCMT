# TSC v41/r37 Seed-Heldout B100 Ensemble Outcome

This document reports the frozen nested seven-city selector with B100 OOF selection and independent seed-4047 ensemble evaluation.

- integrity: `PASS`
- development gate: `PASS`
- cache SHA-256: `14c6dfad87999e55fa86fa942ac0ca2334129f4d1662b3c31a6e4fc7aa74c242`
- source tree SHA-256: `98514081d1e939cfcbbcd0e53b1eb9094b93943bc11c4c030098c054311300b7`
- v40 audit SHA-256: `760ec21f9a63b6ffaa1bb75b53ea7a00c616bde46118100177363ff6c4488a89`

## Frozen gate

- LOCO anchor macro regret: `0.241420659`
- LOCO selected macro regret: `0.217669889`
- LOCO relative improvement: `9.8379%`
- LOCO improved cities: `6/7`
- LOCO maximum city regression: `0.003881324`
- global selector: `scope_all_tau_0_rho_0p1_lambda_0`
- global relative improvement: `11.2105%`
- global improved cities: `6/7`
- decision: `freeze_cross_fitted_anchored_selector_for_external_confirmation`

| Held-out city | Selector | Anchor regret | Selected regret | Improvement | Non-anchor targets |
|---|---|---:|---:|---:|---:|
| resco_synthetic | scope_constant_tau_0_rho_0p05_lambda_0 | 0.277923 | 0.261069 | 0.016854 | 2 |
| cologne | scope_all_tau_0_rho_0p1_lambda_0 | 0.274342 | 0.188614 | 0.085729 | 2 |
| ingolstadt | scope_all_tau_0_rho_0p1_lambda_0 | 0.290468 | 0.258037 | 0.032430 | 2 |
| atlanta | scope_all_tau_0_rho_0p1_lambda_0 | 0.057208 | 0.040318 | 0.016890 | 1 |
| hangzhou | scope_all_tau_0_rho_0p1_lambda_0 | 0.371040 | 0.374921 | -0.003881 | 2 |
| new_york | scope_all_tau_0_rho_0p1_lambda_0 | 0.321456 | 0.305734 | 0.015722 | 1 |
| salt_lake_city | scope_all_tau_0_rho_0p1_lambda_0 | 0.097507 | 0.094996 | 0.002512 | 2 |

## Global target decisions

| Target | City | Candidate | Anchor regret | Selected regret |
|---|---|---|---:|---:|
| grid4x4 | resco_synthetic | local_cap_0p5_z_0 | 0.319228 | 0.213132 |
| arterial4x4 | resco_synthetic | local_cap_0p5_z_0 | 0.236618 | 0.262614 |
| cologne3 | cologne | constant_alpha_0p6 | 0.285390 | 0.187069 |
| cologne8 | cologne | constant_alpha_0p7 | 0.263295 | 0.190158 |
| ingolstadt7 | ingolstadt | constant_alpha_0p6 | 0.227463 | 0.203883 |
| ingolstadt21 | ingolstadt | constant_alpha_0p2 | 0.353473 | 0.312192 |
| atlanta_1x5 | atlanta | constant_alpha_0p7 | 0.057208 | 0.040318 |
| hangzhou_4x4 | hangzhou | constant_alpha_0p7 | 0.353732 | 0.330268 |
| hangzhou_4x4_hetero | hangzhou | constant_alpha_0p8 | 0.388347 | 0.419574 |
| manhattan_28x7 | new_york | constant_alpha_0p8 | 0.321456 | 0.305734 |
| saltlake_400s_200w_q1_weekday_peak | salt_lake_city | constant_alpha_0p3 | 0.109396 | 0.108110 |
| saltlake_state_university_q1_weekday_peak | salt_lake_city | local_cap_0p25_z_0p25 | 0.085619 | 0.081881 |

## Claim boundary

This is development evidence for simulator-supported target offline adaptation using 100 complete matched action groups and an independently held-out simulator seed. It is not zero-shot, passive real-world adaptation, or external-city confirmation.
