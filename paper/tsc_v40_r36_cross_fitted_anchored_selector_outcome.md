# TSC v40/r36 Cross-Fitted Anchored Selector Outcome

This document reports the frozen nested seven-city selector using only B60 OOF evidence for each target decision.

- integrity: `PASS`
- development gate: `FAIL`
- cache SHA-256: `14c6dfad87999e55fa86fa942ac0ca2334129f4d1662b3c31a6e4fc7aa74c242`
- source tree SHA-256: `1b006f4da6d122f158be832c3d90ef1bbd6aa6cc22254a557ed26a69d6b50224`
- v39b evaluation audit SHA-256: `91058c229f1bbbaae1de7639af07598a95913d4ac2594d5f2e6f3cb0f3089796`

## Frozen gate

- LOCO anchor macro regret: `0.234172246`
- LOCO selected macro regret: `0.236865537`
- LOCO relative improvement: `-1.1501%`
- LOCO improved cities: `0/7`
- LOCO maximum city regression: `0.017393607`
- global selector: `always_anchor`
- global relative improvement: `0.0000%`
- global improved cities: `0/7`
- decision: `reject_v40_cross_fitted_anchored_selector`

| Held-out city | Selector | Anchor regret | Selected regret | Improvement | Non-anchor targets |
|---|---|---:|---:|---:|---:|
| resco_synthetic | always_anchor | 0.303230 | 0.303230 | 0.000000 | 0 |
| cologne | always_anchor | 0.201061 | 0.201061 | 0.000000 | 0 |
| ingolstadt | always_anchor | 0.260662 | 0.260662 | 0.000000 | 0 |
| atlanta | always_anchor | 0.088993 | 0.088993 | 0.000000 | 0 |
| hangzhou | scope_all_tau_0_rho_0_lambda_2 | 0.352801 | 0.354260 | -0.001459 | 2 |
| new_york | scope_all_tau_0_rho_0_lambda_2 | 0.331792 | 0.331792 | 0.000000 | 1 |
| salt_lake_city | scope_all_tau_0_rho_0p1_lambda_0p5 | 0.100667 | 0.118060 | -0.017394 | 2 |

## Global target decisions

| Target | City | Candidate | Anchor regret | Selected regret |
|---|---|---|---:|---:|
| grid4x4 | resco_synthetic | constant_alpha_0 | 0.339415 | 0.339415 |
| arterial4x4 | resco_synthetic | constant_alpha_0 | 0.267045 | 0.267045 |
| cologne1 | cologne | constant_alpha_0 | 0.109361 | 0.109361 |
| cologne3 | cologne | constant_alpha_0 | 0.258126 | 0.258126 |
| cologne8 | cologne | constant_alpha_0 | 0.235696 | 0.235696 |
| ingolstadt1 | ingolstadt | constant_alpha_0 | 0.154014 | 0.154014 |
| ingolstadt7 | ingolstadt | constant_alpha_0 | 0.282285 | 0.282285 |
| ingolstadt21 | ingolstadt | constant_alpha_0 | 0.345687 | 0.345687 |
| atlanta_1x5 | atlanta | constant_alpha_0 | 0.088993 | 0.088993 |
| hangzhou_4x4 | hangzhou | constant_alpha_0 | 0.358276 | 0.358276 |
| hangzhou_4x4_hetero | hangzhou | constant_alpha_0 | 0.347326 | 0.347326 |
| manhattan_28x7 | new_york | constant_alpha_0 | 0.331792 | 0.331792 |
| saltlake_400s_200w_q1_weekday_peak | salt_lake_city | constant_alpha_0 | 0.093151 | 0.093151 |
| saltlake_state_university_q1_weekday_peak | salt_lake_city | constant_alpha_0 | 0.108182 | 0.108182 |

## Claim boundary

This is target-simulator adaptation with 60 complete matched action groups and nested development-city selection. It is not zero-shot, passive real-world adaptation, or external-city confirmation.
