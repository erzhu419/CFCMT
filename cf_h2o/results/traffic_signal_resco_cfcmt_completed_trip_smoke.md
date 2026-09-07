# RESCO CFCMT Phase-Transfer Benchmark

Protocol: real RESCO SUMO scenarios, real `tlLogic` green phases as actions, leave-one-scenario-out source/target split.

Zero-shot residual policies use only source transitions and target static network/demand summaries; few-shot variants use a short passive target transition slice.

## Aggregate Metrics

| policy | mean_queue | seed_std_mean_queue | mean_tls_queue | p90_queue | mean_vehicle_waiting_time | p90_vehicle_waiting_time | mean_active_vehicles | mean_completed_travel_time | p90_completed_travel_time | mean_completed_waiting_time | p90_completed_waiting_time | completed_trips | throughput_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2oplus_dense_phase_mpc | 2.1858 | nan | 2.7500 | 4.1401 | 0.4210 | 0.5701 | 16.2000 | 28.1944 | 39.1000 | 1.3333 | 1.6000 | 7.5000 | 0.2702 |
| sim_generic_phase_mpc | 2.2287 | nan | 2.8634 | 4.1401 | 0.4223 | 0.5755 | 16.2000 | 28.1944 | 39.1000 | 1.3333 | 1.6000 | 7.5000 | 0.2702 |
| sim_source_avg_phase_mpc | 2.2287 | nan | 2.8634 | 4.1401 | 0.4223 | 0.5755 | 16.2000 | 28.1944 | 39.1000 | 1.3333 | 1.6000 | 7.5000 | 0.2702 |
| sim_target_static_phase_mpc | 2.2287 | nan | 2.8634 | 4.1401 | 0.4223 | 0.5755 | 16.2000 | 28.1944 | 39.1000 | 1.3333 | 1.6000 | 7.5000 | 0.2702 |
| cfcmt_phase_fewshot_selector_mpc | 2.2287 | nan | 2.8634 | 4.1401 | 0.4223 | 0.5755 | 16.2000 | 28.1944 | 39.1000 | 1.3333 | 1.6000 | 7.5000 | 0.2702 |
| cfcmt_phase_global_mpc | 2.2287 | nan | 2.8634 | 4.1401 | 0.4221 | 0.5745 | 16.2818 | 28.9444 | 39.6000 | 1.3333 | 1.6000 | 7.5000 | 0.2702 |
| cfcmt_phase_trust_mpc | 2.2287 | nan | 2.8634 | 4.1401 | 0.4221 | 0.5745 | 16.2818 | 28.9444 | 39.6000 | 1.3333 | 1.6000 | 7.5000 | 0.2702 |
| cfcmt_phase_fewshot_bias_mpc | 2.2287 | nan | 2.8634 | 4.1401 | 0.4221 | 0.5745 | 16.2818 | 28.9444 | 39.6000 | 1.3333 | 1.6000 | 7.5000 | 0.2702 |
| cfcmt_phase_pressure_guard_mpc | 2.3561 | nan | 3.2296 | 4.4923 | 0.4397 | 0.7192 | 16.2364 | 28.4444 | 39.9000 | 1.3333 | 1.6000 | 7.0000 | 0.2326 |
| max_pressure | 2.5973 | nan | 3.4548 | 4.9598 | 0.4683 | 1.0826 | 16.2364 | 28.4444 | 39.9000 | 1.3333 | 1.6000 | 7.0000 | 0.2326 |
| phase_pressure | 2.5973 | nan | 3.4548 | 4.9598 | 0.4683 | 1.0826 | 16.2364 | 28.4444 | 39.9000 | 1.3333 | 1.6000 | 7.0000 | 0.2326 |
| phase_spillback_pressure | 2.5973 | nan | 3.4548 | 4.9598 | 0.4683 | 1.0826 | 16.2364 | 28.4444 | 39.9000 | 1.3333 | 1.6000 | 7.0000 | 0.2326 |
| cfcmt_phase_passive_policy_selector_mpc | 2.5973 | nan | 3.4548 | 4.9598 | 0.4683 | 1.0826 | 16.2364 | 28.4444 | 39.9000 | 1.3333 | 1.6000 | 7.0000 | 0.2326 |
| fixed_program | 3.8759 | nan | 4.7798 | 7.5879 | 0.8970 | 3.0442 | 16.4977 | 31.7857 | 41.6000 | 2.7857 | 7.0000 | 7.0000 | 0.2674 |

## Paired Target Bootstrap

Delta is `cfcmt_phase_pressure_guard_mpc - baseline` on target mean queue; negative values favor CFCMT.

| baseline | mean_delta | ci_low | ci_high | win_targets | target_count |
| --- | --- | --- | --- | --- | --- |
| max_pressure | -0.2412 | -0.4825 | 0.0000 | 1 | 2 |
| phase_pressure | -0.2412 | -0.4825 | 0.0000 | 1 | 2 |
| sim_target_static_phase_mpc | 0.1274 | -0.0859 | 0.3406 | 1 | 2 |
| h2oplus_dense_phase_mpc | 0.1703 | 0.0000 | 0.3406 | 0 | 2 |
| phase_spillback_pressure | -0.2412 | -0.4825 | 0.0000 | 1 | 2 |
| fixed_program | -1.5198 | -2.8650 | -0.1746 | 2 | 2 |

## Paired Seed-Target Bootstrap

Delta is `cfcmt_phase_pressure_guard_mpc - baseline` over all seed-target pairs; negative values favor CFCMT.

| baseline | mean_delta | ci_low | ci_high | win_pairs | pair_count |
| --- | --- | --- | --- | --- | --- |

## Target Metrics

| target | best_policy | best_queue | selector | policy_selector | fixed_program | max_pressure | phase_pressure | phase_spillback_pressure | sim_generic_phase_mpc | sim_source_avg_phase_mpc | sim_target_static_phase_mpc | h2oplus_dense_phase_mpc | cfcmt_phase_global_mpc | cfcmt_phase_trust_mpc | cfcmt_phase_pressure_guard_mpc | cfcmt_phase_fewshot_bias_mpc | cfcmt_phase_fewshot_selector_mpc | cfcmt_phase_passive_policy_selector_mpc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cologne1 | sim_generic_phase_mpc | 2.3053 | sim_generic_phase_mpc | max_pressure | 2.8205 | 2.6459 | 2.6459 | 2.6459 | 2.3053 | 2.3053 | 2.3053 | 2.3053 | 2.3053 | 2.3053 | 2.6459 | 2.3053 | 2.3053 | 2.6459 |
| cologne3 | h2oplus_dense_phase_mpc | 2.0662 | cfcmt_phase_global_mpc | max_pressure | 4.9313 | 2.5487 | 2.5487 | 2.5487 | 2.1521 | 2.1521 | 2.1521 | 2.0662 | 2.1521 | 2.1521 | 2.0662 | 2.1521 | 2.1521 | 2.5487 |

## Source Transition Counts

| scenario | transitions |
| --- | --- |
| cologne1 | 6 |
| cologne3 | 18 |

## Notes

- `sim_generic_phase_mpc` uses a fixed generic static prior; `sim_source_avg_phase_mpc` uses only source-scenario average static summaries; `sim_target_static_phase_mpc` uses target route/network static summaries but no target next-state labels.
- `h2oplus_dense_phase_mpc` is a dense residual predictor inspired by H2O+ dynamics-gap correction.
- `cfcmt_phase_*` uses a sparse phase-local residual parent set plus optional source-trust scaling or few-shot target passive adaptation.
- `cfcmt_phase_passive_policy_selector_mpc` uses passive target snapshots to select among pressure, simulator, H2O+-style dense, and CFCMT policies under the few-shot-validated predictor; it does not inspect target closed-loop rollouts.
- The benchmark is not a faithful RESCO RL leaderboard result; it is a cross-network transfer stress test over real SUMO networks and phase programs.
