# RESCO CFCMT Phase-Transfer Benchmark

Protocol: real RESCO SUMO scenarios, real `tlLogic` green phases as actions, leave-one-scenario-out source/target split.

Zero-shot residual policies use only source transitions and target static network/demand summaries; few-shot variants use a short passive target transition slice.

## Aggregate Metrics

| policy | mean_queue | seed_std_mean_queue | mean_tls_queue | p90_queue | mean_vehicle_waiting_time | p90_vehicle_waiting_time | mean_active_vehicles | mean_completed_travel_time | p90_completed_travel_time | mean_completed_waiting_time | p90_completed_waiting_time | completed_trips | throughput_ratio | mean_tripinfo_duration | p90_tripinfo_duration | mean_tripinfo_waiting_time | p90_tripinfo_waiting_time | mean_tripinfo_time_loss | p90_tripinfo_time_loss | mean_tripinfo_depart_delay | p90_tripinfo_depart_delay | tripinfo_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_pressure | 32.2343 | 0.9901 | 34.3103 | 62.1800 | 39.0123 | 131.4616 | 80.3234 | 105.3056 | 151.4750 | 21.8582 | 31.3042 | 2161.8750 | 0.9492 | 104.2291 | 150.2417 | 20.8528 | 30.2250 | 38.5434 | 61.4375 | 12.5393 | 10.2125 | 2167.5417 |
| max_pressure | 42.7872 | 0.7314 | 44.9300 | 71.6277 | 29.7900 | 95.7911 | 87.3857 | 107.9864 | 159.6583 | 21.5565 | 39.4542 | 2163.5417 | 0.9448 | 106.9182 | 158.5167 | 20.5521 | 38.4000 | 40.7331 | 68.5168 | 32.0923 | 62.8620 | 2169.2083 |
| cfcmt_phase_pressure_guard_mpc | 43.5106 | 0.5699 | 45.6455 | 74.2496 | 30.5052 | 96.8623 | 88.7722 | 109.7045 | 162.4750 | 23.0473 | 41.7875 | 2163.5833 | 0.9448 | 108.6326 | 161.2583 | 22.0400 | 40.7375 | 42.4485 | 71.9155 | 33.3048 | 67.5370 | 2169.2500 |
| phase_spillback_pressure | 43.6554 | 1.5028 | 45.8533 | 68.3469 | 42.0771 | 150.5116 | 88.5714 | 158.0773 | 300.2917 | 67.3439 | 174.8625 | 2101.2500 | 0.9175 | 156.9780 | 299.0333 | 66.3515 | 173.9625 | 90.6855 | 211.9775 | 19.5865 | 38.3398 | 2106.9167 |
| sim_source_avg_phase_mpc | 44.2873 | 0.0873 | 46.3898 | 63.9731 | 43.6505 | 130.3929 | 88.7348 | 263.0371 | 435.0417 | 174.5099 | 302.0292 | 2099.4167 | 0.8754 | 261.9484 | 433.9708 | 173.8499 | 301.4792 | 192.8206 | 333.2415 | 45.4390 | 127.9979 | 2105.1250 |
| sim_generic_phase_mpc | 44.8783 | 1.2535 | 46.9800 | 65.5599 | 45.9883 | 146.7800 | 90.7013 | 259.6020 | 437.2917 | 170.6755 | 304.0042 | 2096.8750 | 0.8756 | 258.5129 | 436.1667 | 169.9993 | 303.3042 | 189.2271 | 335.4298 | 44.2123 | 126.1179 | 2102.5833 |
| sim_target_static_phase_mpc | 44.8973 | 1.2335 | 46.9938 | 65.4094 | 45.9867 | 146.9726 | 90.6361 | 259.3439 | 437.2667 | 170.5664 | 303.7708 | 2098.7083 | 0.8754 | 258.2547 | 436.1958 | 169.8900 | 303.1208 | 188.9974 | 335.1476 | 45.0202 | 126.2079 | 2104.4167 |
| cfcmt_phase_trust_mpc | 46.9466 | 1.0388 | 48.9825 | 71.2773 | 51.4110 | 148.2050 | 98.2696 | 300.4460 | 461.3458 | 209.2529 | 341.3500 | 2042.0417 | 0.8688 | 299.3283 | 460.1042 | 208.6518 | 340.4708 | 229.4935 | 373.4788 | 47.4390 | 131.1158 | 2047.9583 |
| h2oplus_dense_phase_mpc | 49.5058 | 3.9510 | 51.5351 | 74.9880 | 48.5172 | 125.6407 | 96.7482 | 222.5329 | 410.4958 | 131.9737 | 284.1875 | 2061.4583 | 0.8827 | 221.4173 | 408.9667 | 131.1706 | 283.2833 | 150.3275 | 316.9657 | 34.9868 | 93.7287 | 2067.3333 |
| cfcmt_phase_passive_policy_selector_mpc | 51.6698 | 2.6939 | 53.7443 | 79.6228 | 45.6927 | 123.1200 | 97.1336 | 223.0114 | 409.6500 | 131.7394 | 280.9333 | 2081.7917 | 0.8838 | 221.9110 | 408.3417 | 130.9531 | 280.2000 | 150.7422 | 313.4747 | 36.1671 | 85.6683 | 2087.5417 |
| cfcmt_phase_fewshot_selector_mpc | 51.7018 | 2.7563 | 53.7480 | 79.6367 | 49.8363 | 129.8162 | 98.5362 | 225.5618 | 410.9792 | 134.3961 | 282.1875 | 2062.7917 | 0.8823 | 224.4287 | 409.5000 | 133.5787 | 281.0500 | 153.3217 | 315.1118 | 40.8453 | 108.3795 | 2068.6667 |
| cfcmt_phase_fewshot_bias_mpc | 52.2719 | 7.0711 | 54.2632 | 82.3491 | 53.8950 | 153.8639 | 99.3504 | 306.9239 | 475.1292 | 215.3979 | 350.7917 | 2033.7083 | 0.8685 | 305.7615 | 473.4708 | 214.7565 | 350.0042 | 236.0619 | 388.4365 | 55.1470 | 155.8329 | 2039.6250 |
| cfcmt_phase_global_mpc | 52.4237 | 7.2056 | 54.4030 | 82.1637 | 53.7572 | 153.7980 | 99.4903 | 307.4375 | 476.2583 | 215.8668 | 352.3125 | 2033.3333 | 0.8685 | 306.2731 | 474.5708 | 215.2234 | 351.4750 | 236.5704 | 389.4137 | 55.5246 | 156.3745 | 2039.2500 |
| actuated_program | 58.3468 | 1.1370 | 58.3160 | 99.4055 | 6.3557 | 20.2140 | 110.9396 | 165.6482 | 285.9542 | 67.1662 | 150.3250 | 2342.9167 | 0.9714 | 164.5754 | 284.8833 | 66.1326 | 149.2542 | 98.1604 | 199.5743 | 41.3109 | 160.4408 | 2346.5833 |
| fixed_program | 84.6088 | 0.6458 | 85.9953 | 133.0312 | 8.1372 | 25.1472 | 137.8017 | 222.3477 | 368.7167 | 114.4737 | 217.7667 | 2233.4583 | 0.9385 | 221.2093 | 367.6042 | 113.3988 | 216.6083 | 154.3776 | 281.2557 | 36.6823 | 130.6593 | 2238.9167 |

## Paired Target Bootstrap

Delta is `cfcmt_phase_pressure_guard_mpc - baseline` on target mean queue; negative values favor CFCMT.

| baseline | mean_delta | ci_low | ci_high | win_targets | target_count |
| --- | --- | --- | --- | --- | --- |
| max_pressure | 0.7235 | -0.0983 | 2.2687 | 1 | 8 |
| phase_pressure | 11.2763 | -1.1392 | 34.0498 | 4 | 8 |
| actuated_program | -14.8361 | -26.1103 | -2.8833 | 7 | 8 |
| sim_target_static_phase_mpc | -1.3866 | -5.7031 | 1.8546 | 5 | 8 |
| h2oplus_dense_phase_mpc | -5.9951 | -14.7118 | -0.0205 | 6 | 8 |
| phase_spillback_pressure | -0.1448 | -11.6868 | 10.4748 | 4 | 8 |
| fixed_program | -41.0981 | -69.0496 | -19.4909 | 8 | 8 |

## Paired Seed-Target Bootstrap

Delta is `cfcmt_phase_pressure_guard_mpc - baseline` over all seed-target pairs; negative values favor CFCMT.

| baseline | mean_delta | ci_low | ci_high | win_pairs | pair_count |
| --- | --- | --- | --- | --- | --- |
| max_pressure | 0.7235 | -0.0694 | 2.2268 | 1 | 24 |
| phase_pressure | 11.2763 | 0.5624 | 23.8420 | 15 | 24 |
| actuated_program | -14.8361 | -21.5846 | -7.3945 | 20 | 24 |
| sim_target_static_phase_mpc | -1.3866 | -4.4389 | 1.2557 | 17 | 24 |
| h2oplus_dense_phase_mpc | -5.9951 | -12.7723 | -0.9680 | 20 | 24 |
| phase_spillback_pressure | -0.1448 | -6.4773 | 6.3281 | 16 | 24 |
| fixed_program | -41.0981 | -56.9522 | -27.4205 | 24 | 24 |

## Target Metrics

| target | best_policy | best_queue | selector | policy_selector | fixed_program | actuated_program | max_pressure | phase_pressure | phase_spillback_pressure | sim_generic_phase_mpc | sim_source_avg_phase_mpc | sim_target_static_phase_mpc | h2oplus_dense_phase_mpc | cfcmt_phase_global_mpc | cfcmt_phase_trust_mpc | cfcmt_phase_pressure_guard_mpc | cfcmt_phase_fewshot_bias_mpc | cfcmt_phase_fewshot_selector_mpc | cfcmt_phase_passive_policy_selector_mpc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid4x4 | phase_spillback_pressure | 20.0318 | multi_seed | multi_seed | 52.3748 | 49.3779 | 20.0456 | 20.1572 | 20.0318 | 20.1635 | 20.1503 | 20.1680 | 22.3611 | 22.9448 | 22.3069 | 20.0456 | 22.8483 | 23.0512 | 23.1476 |
| arterial4x4 | phase_pressure | 178.9225 | multi_seed | multi_seed | 393.6939 | 251.2473 | 267.9500 | 178.9225 | 238.9213 | 266.8237 | 267.7984 | 266.8237 | 301.1737 | 268.5176 | 268.5176 | 267.9500 | 268.5176 | 301.1737 | 301.1737 |
| cologne1 | sim_target_static_phase_mpc | 6.7024 | multi_seed | multi_seed | 28.1036 | 19.5531 | 7.5738 | 7.8133 | 7.2705 | 6.8105 | 6.7835 | 6.7024 | 7.2760 | 41.8554 | 7.5260 | 7.3117 | 40.7369 | 6.7835 | 6.9750 |
| cologne3 | max_pressure | 5.3713 | multi_seed | multi_seed | 22.8246 | 20.9466 | 5.3713 | 5.4606 | 6.3083 | 5.5227 | 5.5014 | 5.5014 | 7.7362 | 11.5514 | 5.5763 | 11.4211 | 11.5514 | 7.7362 | 7.7362 |
| cologne8 | phase_pressure | 9.8643 | multi_seed | multi_seed | 31.1408 | 29.0301 | 9.9030 | 9.8643 | 41.7872 | 11.7093 | 11.7000 | 11.7093 | 11.7831 | 21.6325 | 15.7728 | 9.9030 | 21.6325 | 21.6325 | 21.6325 |
| ingolstadt1 | max_pressure | 3.0899 | multi_seed | multi_seed | 11.6812 | 6.1194 | 3.0899 | 3.2758 | 3.2031 | 3.6234 | 3.6409 | 3.6409 | 4.4273 | 4.6003 | 4.5324 | 3.0899 | 4.6003 | 4.5801 | 3.6751 |
| ingolstadt7 | phase_pressure | 7.8430 | multi_seed | multi_seed | 43.3625 | 26.6196 | 7.8461 | 7.8430 | 8.3635 | 9.0371 | 9.2551 | 9.3986 | 8.5939 | 8.6682 | 8.4513 | 7.8461 | 8.6682 | 9.0371 | 9.3986 |
| ingolstadt21 | max_pressure | 20.5177 | multi_seed | multi_seed | 93.6888 | 63.8803 | 20.5177 | 24.5377 | 23.3576 | 35.3362 | 29.4687 | 35.2337 | 32.6947 | 39.6196 | 42.8895 | 20.5177 | 39.6196 | 39.6196 | 39.6196 |

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
