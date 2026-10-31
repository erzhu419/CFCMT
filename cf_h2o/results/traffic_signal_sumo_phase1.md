# Traffic Signal SUMO Phase 1

This is a microscopic SUMO feasibility benchmark. It is not yet a full TSC paper experiment.

Protocol: generated 3x3 grid, center intersection B1 controlled by libsumo, leave-one-scenario-out transfer. Zero-shot CFCMT and H2O+ use source SUMO transitions only; the few-shot CFCMT variant uses passive target transitions from a short behavior-policy run.

## Aggregate Policy Metrics

| policy | mean_cost | mean_queue | p90_queue | throughput_ratio | ns_action_share |
| --- | --- | --- | --- | --- | --- |
| sim_mpc | 23.7120 | 22.0601 | 33.8560 | 0.9347 | 0.4691 |
| cfcmt_global_mpc | 24.3528 | 22.5121 | 33.8295 | 0.9350 | 0.4364 |
| cfcmt_trust_mpc | 24.4363 | 22.7086 | 33.0584 | 0.9303 | 0.4473 |
| spillback_pressure | 24.7034 | 22.9909 | 35.1287 | 0.9342 | 0.4655 |
| queue_pressure | 25.4569 | 23.6113 | 35.7796 | 0.9355 | 0.4545 |
| h2oplus_dense_mpc | 25.4968 | 23.5898 | 34.7854 | 0.9310 | 0.4291 |
| cfcmt_fewshot_selector_mpc | 25.5651 | 23.8230 | 36.2710 | 0.9311 | 0.4400 |
| cfcmt_fewshot_bias_mpc | 25.9214 | 23.9779 | 36.6935 | 0.9326 | 0.4255 |
| fixed_time | 34.5244 | 30.5987 | 48.0126 | 0.9352 | 0.4909 |

## Target-Level Results

| target | best_policy | best_cost | fixed_time | spillback_pressure | h2oplus_dense_mpc | cfcmt_global_mpc | cfcmt_trust_mpc | cfcmt_fewshot_bias_mpc | cfcmt_fewshot_selector_mpc | selector |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_grid | cfcmt_fewshot_selector_mpc | 10.9624 | 14.3570 | 11.2988 | 11.3550 | 11.4827 | 11.1050 | 11.2304 | 10.9624 | cfcmt_trust_mpc |
| ns_arterial_peak | cfcmt_global_mpc | 13.4244 | 18.5727 | 13.8929 | 14.1959 | 13.4244 | 13.8274 | 13.6757 | 13.7916 | cfcmt_fewshot_bias_mpc |
| ew_arterial_peak | spillback_pressure | 16.2025 | 42.6843 | 16.2025 | 18.0496 | 17.9341 | 16.7476 | 17.4424 | 18.0093 | cfcmt_fewshot_bias_mpc |
| downtown_spillback | sim_mpc | 56.2189 | 57.4510 | 60.6702 | 64.5397 | 60.0334 | 58.8408 | 63.0154 | 61.7687 | cfcmt_global_mpc |
| event_reversal | cfcmt_global_mpc | 18.8892 | 39.5571 | 21.4524 | 19.3440 | 18.8892 | 21.6607 | 24.2433 | 23.2933 | cfcmt_fewshot_bias_mpc |

## Notes

- This Phase 1 benchmark uses a small generated grid, not real city topology.
- The few-shot row is target offline adaptation, not zero-shot.
- There is no SUMO oracle row because cloning the simulator for per-state counterfactual actions is not part of this Phase 1 script.
