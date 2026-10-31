# Traffic Signal Transfer Feasibility Phase 0

This is an analytic queueing benchmark for deciding whether traffic signal control is a better cross-city transfer environment than bus holding. It is not a SUMO experiment.

Transfer protocol: leave-one-network-out; zero-shot methods use no target next-state labels; the few-shot CFCMT variant uses only passive target transitions under the spillback-pressure behavior controller.

## Aggregate Policy Metrics

| method | mean_cost | mean_regret | p90_regret | oracle_action_accuracy | win_rate_vs_spillback_pressure |
| --- | --- | --- | --- | --- | --- |
| oracle_mpc | 268.1977 | 0.0000 | 0.0000 | 1.0000 | 0.7551 |
| cfcmt_fewshot_bias_mpc | 268.3847 | 0.1870 | 0.6999 | 0.8231 | 0.6694 |
| cfcmt_global_mpc | 268.3926 | 0.1949 | 0.7127 | 0.8171 | 0.6617 |
| sim_mpc | 268.6394 | 0.4417 | 0.9473 | 0.7511 | 0.6726 |
| fixed_time | 268.8736 | 0.6760 | 1.3322 | 0.7320 | 0.6591 |
| cfcmt_weighted_mpc | 269.4068 | 1.2092 | 4.7037 | 0.7203 | 0.5849 |
| h2oplus_dense_mpc | 269.6027 | 1.4050 | 4.8381 | 0.6166 | 0.6134 |
| spillback_pressure | 271.9847 | 3.7871 | 8.9699 | 0.2449 | 0.0000 |
| queue_pressure | 273.5092 | 5.3115 | 11.2412 | 0.1846 | 0.0803 |

## Aggregate Model Metrics

| method | next_queue_mae |
| --- | --- |
| cfcmt_fewshot_bias_mpc | 1.2669 |
| cfcmt_global_mpc | 1.6176 |
| h2oplus_dense_mpc | 2.8692 |
| cfcmt_weighted_mpc | 3.3633 |
| sim_mpc | 3.4999 |

## Target-Level Winners

| target | best_non_oracle | best_cost | best_regret | spillback_pressure_cost | cfcmt_weighted_cost | cfcmt_fewshot_cost |
| --- | --- | --- | --- | --- | --- | --- |
| grid_balanced | fixed_time | 165.7923 | 0.0328 | 169.9044 | 166.3767 | 165.8797 |
| arterial_peak | cfcmt_weighted_mpc | 270.2153 | 0.2652 | 270.9239 | 270.2153 | 270.4705 |
| downtown_spillback | fixed_time | 405.1296 | 0.0023 | 409.7212 | 405.3129 | 405.2176 |
| suburban_asymmetric | h2oplus_dense_mpc | 104.1477 | 0.0478 | 108.0840 | 104.3447 | 104.1584 |
| event_reversal | cfcmt_fewshot_bias_mpc | 396.1974 | 0.1458 | 401.2902 | 400.7846 | 396.1974 |

## Interpretation

- If CFCMT weighted/few-shot beats spillback-pressure on regret and cost, the TSC direction is worth a SUMO Phase 1 implementation.
- If spillback-pressure remains best, TSC may have the same rule-ceiling problem as bus holding and should not replace the paper's main environment without a stronger task design.
- The few-shot row is not zero-shot: it uses target passive next-state labels and should be reported as target offline adaptation.
