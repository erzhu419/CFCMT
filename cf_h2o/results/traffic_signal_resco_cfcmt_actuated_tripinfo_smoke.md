# RESCO CFCMT Phase-Transfer Benchmark

Protocol: real RESCO SUMO scenarios, real `tlLogic` green phases as actions, leave-one-scenario-out source/target split.

Zero-shot residual policies use only source transitions and target static network/demand summaries; few-shot variants use a short passive target transition slice.

## Aggregate Metrics

| policy | mean_queue | seed_std_mean_queue | mean_tls_queue | p90_queue | mean_vehicle_waiting_time | p90_vehicle_waiting_time | mean_active_vehicles | mean_completed_travel_time | p90_completed_travel_time | mean_completed_waiting_time | p90_completed_waiting_time | completed_trips | throughput_ratio | mean_tripinfo_duration | p90_tripinfo_duration | mean_tripinfo_waiting_time | p90_tripinfo_waiting_time | mean_tripinfo_time_loss | p90_tripinfo_time_loss | mean_tripinfo_depart_delay | p90_tripinfo_depart_delay | tripinfo_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2oplus_dense_phase_mpc | 2.1858 | nan | 2.7500 | 4.1401 | 0.4210 | 0.5701 | 16.2000 | 28.1944 | 39.1000 | 1.3333 | 1.6000 | 7.5000 | 0.2702 | 27.1944 | 38.1000 | 0.3333 | 0.6000 | 5.9792 | 8.3680 | 0.0556 | 0.1000 | 7.5000 |
| cfcmt_phase_pressure_guard_mpc | 2.3561 | nan | 3.2296 | 4.4923 | 0.4397 | 0.7192 | 16.2364 | 28.4444 | 39.9000 | 1.3333 | 1.6000 | 7.0000 | 0.2326 | 27.4444 | 38.9000 | 0.3333 | 0.6000 | 5.7700 | 7.8060 | 0.0556 | 0.1000 | 7.0000 |
| max_pressure | 2.5973 | nan | 3.4548 | 4.9598 | 0.4683 | 1.0826 | 16.2364 | 28.4444 | 39.9000 | 1.3333 | 1.6000 | 7.0000 | 0.2326 | 27.4444 | 38.9000 | 0.3333 | 0.6000 | 5.7667 | 7.8060 | 0.0556 | 0.1000 | 7.0000 |
| actuated_program | 3.2750 | nan | 3.7651 | 5.3931 | 0.8828 | 3.0059 | 16.6672 | 34.0000 | 43.0000 | 2.5714 | 6.4000 | 7.0000 | 0.2674 | 33.0000 | 42.0000 | 1.5714 | 5.4000 | 8.1814 | 15.1670 | 0.2857 | 0.8000 | 7.0000 |
| fixed_program | 3.8759 | nan | 4.7798 | 7.5879 | 0.8970 | 3.0442 | 16.4977 | 31.7857 | 41.6000 | 2.7857 | 7.0000 | 7.0000 | 0.2674 | 30.7857 | 40.6000 | 1.7857 | 6.0000 | 8.2736 | 15.5500 | 0.2857 | 0.8000 | 7.0000 |

## Paired Target Bootstrap

Delta is `cfcmt_phase_pressure_guard_mpc - baseline` on target mean queue; negative values favor CFCMT.

| baseline | mean_delta | ci_low | ci_high | win_targets | target_count |
| --- | --- | --- | --- | --- | --- |
| max_pressure | -0.2412 | -0.4825 | 0.0000 | 1 | 2 |
| phase_pressure | nan | nan | nan | 0 | 0 |
| sim_target_static_phase_mpc | nan | nan | nan | 0 | 0 |
| h2oplus_dense_phase_mpc | 0.1703 | 0.0000 | 0.3406 | 0 | 2 |
| phase_spillback_pressure | nan | nan | nan | 0 | 0 |
| fixed_program | -1.5198 | -2.8650 | -0.1746 | 2 | 2 |

## Paired Seed-Target Bootstrap

Delta is `cfcmt_phase_pressure_guard_mpc - baseline` over all seed-target pairs; negative values favor CFCMT.

| baseline | mean_delta | ci_low | ci_high | win_pairs | pair_count |
| --- | --- | --- | --- | --- | --- |

## Target Metrics

| target | best_policy | best_queue | selector | policy_selector | fixed_program | actuated_program | max_pressure | h2oplus_dense_phase_mpc | cfcmt_phase_pressure_guard_mpc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cologne1 | h2oplus_dense_phase_mpc | 2.3053 | sim_generic_phase_mpc | max_pressure | 2.8205 | 2.8205 | 2.6459 | 2.3053 | 2.6459 |
| cologne3 | h2oplus_dense_phase_mpc | 2.0662 | cfcmt_phase_global_mpc | max_pressure | 4.9313 | 3.7295 | 2.5487 | 2.0662 | 2.0662 |

## Source Transition Counts

| scenario | transitions |
| --- | --- |
| cologne1 | 6 |
| cologne3 | 18 |

## Notes

- `sim_generic_phase_mpc` uses a fixed generic static prior; `sim_source_avg_phase_mpc` uses only source-scenario average static summaries; `sim_target_static_phase_mpc` uses target route/network static summaries but no target next-state labels.
- `actuated_program` evaluates a generated native SUMO actuated version of the same RESCO network and route demand; it does not call the learned/controller phase override.
- `h2oplus_dense_phase_mpc` is a dense residual predictor inspired by H2O+ dynamics-gap correction.
- `cfcmt_phase_*` uses a sparse phase-local residual parent set plus optional source-trust scaling or few-shot target passive adaptation.
- `cfcmt_phase_passive_policy_selector_mpc` uses passive target snapshots to select among pressure, simulator, H2O+-style dense, and CFCMT policies under the few-shot-validated predictor; it does not inspect target closed-loop rollouts.
- The benchmark is not a faithful RESCO RL leaderboard result; it is a cross-network transfer stress test over real SUMO networks and phase programs.
