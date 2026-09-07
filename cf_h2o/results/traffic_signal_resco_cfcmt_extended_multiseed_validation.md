# RESCO CFCMT Phase-Transfer Benchmark

Protocol: real RESCO SUMO scenarios, real `tlLogic` green phases as actions, leave-one-scenario-out source/target split.

Zero-shot residual policies use only source transitions and target static network/demand summaries; few-shot variants use a short passive target transition slice.

## Aggregate Metrics

| policy | mean_queue | seed_std_mean_queue | mean_tls_queue | p90_queue | mean_vehicle_waiting_time | p90_vehicle_waiting_time | mean_active_vehicles | mean_completed_travel_time | p90_completed_travel_time | mean_completed_waiting_time | p90_completed_waiting_time | completed_trips | throughput_ratio | mean_tripinfo_duration | p90_tripinfo_duration | mean_tripinfo_waiting_time | p90_tripinfo_waiting_time | mean_tripinfo_time_loss | p90_tripinfo_time_loss | mean_tripinfo_depart_delay | p90_tripinfo_depart_delay | tripinfo_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_pressure | 12.9068 | 0.1546 | 14.5889 | 19.5279 | 14.4671 | 47.3056 | 58.3304 | 89.0535 | 133.1083 | 9.1777 | 22.3250 | 300.6250 | 0.8267 | 87.5891 | 131.3375 | 8.0605 | 21.0750 | 24.5171 | 47.7396 | 3.2964 | 8.9063 | 306.2917 |
| cfcmt_phase_pressure_guard_mpc | 12.9180 | 0.1919 | 14.6028 | 19.5973 | 14.4874 | 47.4903 | 58.2947 | 88.9497 | 132.7750 | 9.1280 | 22.1375 | 300.2917 | 0.8256 | 87.4863 | 131.0333 | 8.0114 | 20.8833 | 24.4127 | 47.2685 | 3.3064 | 8.9271 | 305.9583 |
| phase_pressure | 13.2446 | 0.1816 | 14.8995 | 19.3911 | 16.8264 | 46.4349 | 57.5121 | 88.0038 | 132.7208 | 8.5728 | 20.6000 | 296.0833 | 0.8199 | 86.5371 | 131.0375 | 7.4544 | 19.4875 | 23.4628 | 47.7057 | 2.6887 | 8.4688 | 301.7500 |
| h2oplus_dense_phase_mpc | 17.2279 | 3.6084 | 18.8089 | 28.9325 | 20.2121 | 51.3646 | 61.1936 | 88.1811 | 129.5893 | 9.7262 | 23.3601 | 282.2083 | 0.7653 | 86.6585 | 127.9690 | 8.5768 | 21.9952 | 24.5488 | 48.1127 | 2.6978 | 8.6556 | 288.0833 |
| phase_spillback_pressure | 18.0660 | 1.6679 | 19.8785 | 28.2371 | 17.9010 | 52.9680 | 60.9412 | 101.8025 | 157.0667 | 20.6279 | 44.3667 | 288.6250 | 0.7742 | 100.2665 | 155.2708 | 19.4579 | 42.8292 | 37.2376 | 72.4931 | 2.6506 | 8.4816 | 294.2917 |
| cfcmt_phase_fewshot_selector_mpc | 18.4102 | 3.7846 | 19.9792 | 31.8288 | 20.2575 | 52.8214 | 62.0303 | 88.5859 | 132.7315 | 10.3330 | 26.3875 | 281.7917 | 0.7642 | 87.0560 | 131.0982 | 9.1799 | 25.0226 | 25.0525 | 51.4850 | 2.8521 | 9.2096 | 287.6667 |
| cfcmt_phase_passive_policy_selector_mpc | 18.4164 | 3.8516 | 19.9987 | 32.0107 | 19.6790 | 50.3001 | 61.8592 | 88.6671 | 132.9024 | 10.2557 | 26.2423 | 283.4167 | 0.7685 | 87.1494 | 131.0226 | 9.1134 | 25.0179 | 25.0603 | 51.5687 | 2.9492 | 9.1935 | 289.1667 |
| sim_generic_phase_mpc | 20.3592 | 0.1062 | 22.0617 | 32.4161 | 18.6182 | 48.5752 | 62.3696 | 99.3047 | 162.1833 | 20.9454 | 55.1042 | 285.7500 | 0.7430 | 97.8509 | 160.5333 | 19.8715 | 54.0417 | 35.3071 | 78.6640 | 2.7483 | 8.7300 | 291.4583 |
| sim_target_static_phase_mpc | 20.4023 | 0.1758 | 22.1022 | 32.5538 | 18.5088 | 47.4673 | 62.3910 | 99.4201 | 162.1833 | 20.9781 | 55.1250 | 285.7917 | 0.7429 | 97.9633 | 160.5167 | 19.9032 | 54.1375 | 35.3928 | 78.7581 | 2.8326 | 8.7521 | 291.5000 |
| sim_source_avg_phase_mpc | 20.7008 | 0.3845 | 22.3833 | 33.3681 | 18.6434 | 47.1982 | 62.6004 | 102.1163 | 162.2167 | 24.0176 | 55.2625 | 285.2083 | 0.7405 | 100.6584 | 160.5083 | 22.9531 | 54.2458 | 38.3345 | 78.7712 | 2.8172 | 8.6067 | 290.9167 |
| cfcmt_phase_trust_mpc | 21.5663 | 0.7542 | 23.0683 | 36.3767 | 20.2170 | 47.3767 | 63.0802 | 87.4931 | 130.2446 | 10.0339 | 22.6958 | 279.1250 | 0.7293 | 85.9499 | 128.5923 | 8.8794 | 21.3780 | 24.8904 | 47.4842 | 2.8647 | 9.2510 | 285.0417 |
| cfcmt_phase_global_mpc | 24.9354 | 2.6462 | 26.3823 | 43.9184 | 21.2921 | 51.2577 | 65.9777 | 90.5920 | 142.9571 | 12.7556 | 34.8923 | 270.9167 | 0.7180 | 88.9932 | 141.0577 | 11.5601 | 33.4500 | 28.1175 | 61.2831 | 2.7818 | 9.1254 | 276.8333 |
| cfcmt_phase_fewshot_bias_mpc | 24.9360 | 2.6456 | 26.3800 | 43.9184 | 21.2920 | 51.2571 | 65.9765 | 90.6044 | 142.9667 | 12.7605 | 34.9018 | 270.8750 | 0.7182 | 89.0054 | 141.0577 | 11.5649 | 33.4738 | 28.1289 | 61.3347 | 2.7813 | 9.1302 | 276.7917 |
| actuated_program | 30.1825 | 0.3536 | 30.3411 | 47.6873 | 4.6200 | 15.0782 | 76.1919 | 121.2029 | 185.2000 | 34.1729 | 69.6917 | 290.5833 | 0.7580 | 119.7813 | 183.8292 | 33.0017 | 68.4083 | 57.0805 | 102.7690 | 2.9132 | 8.1198 | 294.2500 |
| fixed_program | 42.6732 | 0.2709 | 43.8127 | 69.6852 | 6.9422 | 21.9508 | 85.3546 | 144.3444 | 221.5958 | 56.2211 | 102.9292 | 270.4167 | 0.7127 | 142.3116 | 219.1958 | 54.6333 | 101.0833 | 79.8759 | 140.4191 | 3.4433 | 10.3208 | 275.8750 |

## Paired Target Bootstrap

Delta is `cfcmt_phase_pressure_guard_mpc - baseline` on target mean queue; negative values favor CFCMT.

| baseline | mean_delta | ci_low | ci_high | win_targets | target_count |
| --- | --- | --- | --- | --- | --- |
| max_pressure | 0.0112 | -0.0324 | 0.0660 | 1 | 8 |
| phase_pressure | -0.3266 | -0.8926 | 0.0583 | 5 | 8 |
| actuated_program | -17.2645 | -25.0067 | -10.1984 | 8 | 8 |
| sim_target_static_phase_mpc | -7.4843 | -19.9899 | -0.1642 | 5 | 8 |
| h2oplus_dense_phase_mpc | -4.3099 | -10.2487 | -0.2689 | 8 | 8 |
| phase_spillback_pressure | -5.1480 | -11.4883 | -0.2848 | 8 | 8 |
| fixed_program | -29.7552 | -41.8576 | -19.2473 | 8 | 8 |

## Paired Seed-Target Bootstrap

Delta is `cfcmt_phase_pressure_guard_mpc - baseline` over all seed-target pairs; negative values favor CFCMT.

| baseline | mean_delta | ci_low | ci_high | win_pairs | pair_count |
| --- | --- | --- | --- | --- | --- |
| max_pressure | 0.0112 | -0.0457 | 0.0787 | 3 | 24 |
| phase_pressure | -0.3266 | -0.7588 | 0.0227 | 12 | 24 |
| actuated_program | -17.2645 | -21.8067 | -12.9979 | 24 | 24 |
| sim_target_static_phase_mpc | -7.4843 | -14.3783 | -1.7428 | 18 | 24 |
| h2oplus_dense_phase_mpc | -4.3099 | -9.6002 | -0.9210 | 18 | 24 |
| phase_spillback_pressure | -5.1480 | -9.2895 | -1.7008 | 20 | 24 |
| fixed_program | -29.7552 | -36.5483 | -23.3073 | 24 | 24 |

## Target Metrics

| target | best_policy | best_queue | selector | policy_selector | fixed_program | actuated_program | max_pressure | phase_pressure | phase_spillback_pressure | sim_generic_phase_mpc | sim_source_avg_phase_mpc | sim_target_static_phase_mpc | h2oplus_dense_phase_mpc | cfcmt_phase_global_mpc | cfcmt_phase_trust_mpc | cfcmt_phase_pressure_guard_mpc | cfcmt_phase_fewshot_bias_mpc | cfcmt_phase_fewshot_selector_mpc | cfcmt_phase_passive_policy_selector_mpc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid4x4 | sim_generic_phase_mpc | 8.7975 | multi_seed | multi_seed | 25.0374 | 21.8213 | 8.8850 | 8.8850 | 8.8909 | 8.7975 | 8.7975 | 8.7975 | 9.0607 | 8.9736 | 8.9271 | 8.8850 | 8.9736 | 9.0687 | 9.0687 |
| arterial4x4 | max_pressure | 39.8138 | multi_seed | multi_seed | 100.2401 | 70.5620 | 39.8138 | 41.9602 | 65.1852 | 88.9778 | 90.9106 | 88.9778 | 63.3304 | 95.2945 | 95.2945 | 39.8138 | 95.2945 | 63.3304 | 63.3304 |
| cologne1 | cfcmt_phase_passive_policy_selector_mpc | 9.6156 | multi_seed | multi_seed | 36.5116 | 27.6266 | 9.7276 | 10.4032 | 10.1342 | 9.7979 | 9.9173 | 9.7328 | 10.1781 | 31.3517 | 10.6538 | 9.9036 | 31.3565 | 9.7979 | 9.6156 |
| cologne3 | cfcmt_phase_global_mpc | 6.3767 | multi_seed | multi_seed | 25.6248 | 18.8230 | 6.4832 | 6.4592 | 7.0037 | 6.5215 | 6.5215 | 6.5215 | 6.4293 | 6.3767 | 6.4453 | 6.3968 | 6.3767 | 6.4293 | 6.4293 |
| cologne8 | max_pressure | 7.1810 | multi_seed | multi_seed | 23.1953 | 17.2377 | 7.1810 | 7.2582 | 20.6207 | 7.5129 | 7.5129 | 7.5129 | 7.5156 | 7.6547 | 7.5611 | 7.1810 | 7.6547 | 7.6547 | 7.6547 |
| ingolstadt1 | sim_generic_phase_mpc | 4.8473 | multi_seed | multi_seed | 19.1277 | 6.5623 | 4.9442 | 4.9335 | 5.0687 | 4.8473 | 4.8473 | 4.8473 | 5.1071 | 5.1092 | 5.0978 | 4.9442 | 5.1092 | 5.0978 | 4.9194 |
| ingolstadt7 | max_pressure | 7.9895 | multi_seed | multi_seed | 41.4151 | 23.2207 | 7.9895 | 8.2090 | 8.1023 | 9.5721 | 10.0209 | 9.9821 | 9.2388 | 8.3921 | 8.3793 | 7.9895 | 8.3921 | 9.5721 | 9.9821 |
| ingolstadt21 | phase_pressure | 17.8484 | multi_seed | multi_seed | 70.2335 | 55.6065 | 18.2298 | 17.8484 | 19.5220 | 26.8466 | 27.0781 | 26.8466 | 26.9630 | 36.3307 | 30.1718 | 18.2298 | 36.3307 | 36.3307 | 36.3307 |

## Source Transition Counts

| scenario | transitions |
| --- | --- |
| arterial4x4 | 880 |
| cologne1 | 60 |
| cologne3 | 180 |
| cologne8 | 480 |
| grid4x4 | 880 |
| ingolstadt1 | 60 |
| ingolstadt21 | 1260 |
| ingolstadt7 | 420 |

## Notes

- `sim_generic_phase_mpc` uses a fixed generic static prior; `sim_source_avg_phase_mpc` uses only source-scenario average static summaries; `sim_target_static_phase_mpc` uses target route/network static summaries but no target next-state labels.
- `actuated_program` evaluates a generated native SUMO actuated version of the same RESCO network and route demand; it does not call the learned/controller phase override.
- `h2oplus_dense_phase_mpc` is a dense residual predictor inspired by H2O+ dynamics-gap correction.
- `cfcmt_phase_*` uses a sparse phase-local residual parent set plus optional source-trust scaling or few-shot target passive adaptation.
- `cfcmt_phase_passive_policy_selector_mpc` uses passive target snapshots to select among pressure, simulator, H2O+-style dense, and CFCMT policies under the few-shot-validated predictor; it does not inspect target closed-loop rollouts.
- The benchmark is not a faithful RESCO RL leaderboard result; it is a cross-network transfer stress test over real SUMO networks and phase programs.
