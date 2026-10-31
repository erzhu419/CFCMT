# RESCO CFCMT Phase-Transfer Benchmark

Protocol: real RESCO SUMO scenarios, real `tlLogic` green phases as actions, leave-one-scenario-out source/target split.

Zero-shot residual policies use only source transitions and target static network/demand summaries; few-shot variants use a short passive target transition slice.

## Aggregate Metrics

| policy | mean_queue | seed_std_mean_queue | mean_tls_queue | p90_queue | throughput_ratio |
| --- | --- | --- | --- | --- | --- |
| cfcmt_phase_pressure_guard_mpc | 9.8866 | 0.0532 | 10.6668 | 15.4934 | 0.5184 |
| max_pressure | 9.9513 | 0.0546 | 10.8149 | 15.5440 | 0.5219 |
| phase_pressure | 10.2713 | 0.0542 | 11.0773 | 15.9281 | 0.5160 |
| phase_spillback_pressure | 11.4141 | 0.3787 | 12.4052 | 17.4783 | 0.4934 |
| h2oplus_dense_phase_mpc | 11.7763 | 0.0993 | 12.5596 | 19.0013 | 0.4808 |
| sim_generic_phase_mpc | 11.8664 | 0.0842 | 12.6286 | 18.5419 | 0.4935 |
| sim_target_static_phase_mpc | 11.9295 | 0.0050 | 12.6742 | 18.6680 | 0.4915 |
| cfcmt_phase_trust_mpc | 12.0373 | 0.1161 | 12.8576 | 19.3885 | 0.4796 |
| cfcmt_phase_fewshot_bias_mpc | 12.0532 | 0.0641 | 12.8383 | 19.6489 | 0.4747 |
| cfcmt_phase_global_mpc | 12.0540 | 0.0652 | 12.8419 | 19.6695 | 0.4747 |
| sim_source_avg_phase_mpc | 12.1458 | 0.3010 | 12.8756 | 19.0682 | 0.4865 |
| cfcmt_phase_fewshot_selector_mpc | 12.1810 | 0.4821 | 12.9390 | 19.5875 | 0.4767 |
| cfcmt_phase_passive_policy_selector_mpc | 12.2815 | 0.3759 | 13.0518 | 19.8138 | 0.4751 |
| fixed_program | 24.3492 | 0.2050 | 25.3352 | 46.5811 | 0.3411 |

## Paired Target Bootstrap

Delta is `cfcmt_phase_pressure_guard_mpc - baseline` on target mean queue; negative values favor CFCMT.

| baseline | mean_delta | ci_low | ci_high | win_targets | target_count |
| --- | --- | --- | --- | --- | --- |
| max_pressure | -0.0647 | -0.1720 | -0.0005 | 4 | 8 |
| phase_pressure | -0.3847 | -0.8739 | -0.0648 | 6 | 8 |
| sim_target_static_phase_mpc | -2.0429 | -4.5322 | -0.2420 | 5 | 8 |
| h2oplus_dense_phase_mpc | -1.8897 | -5.3065 | 0.0585 | 5 | 8 |
| phase_spillback_pressure | -1.5275 | -3.2598 | -0.0028 | 5 | 8 |
| fixed_program | -14.4626 | -19.7704 | -9.7235 | 8 | 8 |

## Paired Seed-Target Bootstrap

Delta is `cfcmt_phase_pressure_guard_mpc - baseline` over all seed-target pairs; negative values favor CFCMT.

| baseline | mean_delta | ci_low | ci_high | win_pairs | pair_count |
| --- | --- | --- | --- | --- | --- |
| max_pressure | -0.0647 | -0.1950 | 0.0294 | 4 | 16 |
| phase_pressure | -0.3847 | -0.7477 | -0.0989 | 8 | 16 |
| sim_target_static_phase_mpc | -2.0429 | -3.8594 | -0.5274 | 11 | 16 |
| h2oplus_dense_phase_mpc | -1.8897 | -4.3414 | -0.0998 | 10 | 16 |
| phase_spillback_pressure | -1.5275 | -2.7519 | -0.4163 | 9 | 16 |
| fixed_program | -14.4626 | -18.2706 | -10.9570 | 16 | 16 |

## Target Metrics

| target | best_policy | best_queue | selector | policy_selector | fixed_program | max_pressure | phase_pressure | phase_spillback_pressure | sim_generic_phase_mpc | sim_source_avg_phase_mpc | sim_target_static_phase_mpc | h2oplus_dense_phase_mpc | cfcmt_phase_global_mpc | cfcmt_phase_trust_mpc | cfcmt_phase_pressure_guard_mpc | cfcmt_phase_fewshot_bias_mpc | cfcmt_phase_fewshot_selector_mpc | cfcmt_phase_passive_policy_selector_mpc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid4x4 | max_pressure | 6.5359 | multi_seed | multi_seed | 12.9668 | 6.5359 | 6.5359 | 6.5672 | 6.5567 | 6.5567 | 6.5567 | 6.6919 | 6.6552 | 6.6552 | 6.5359 | 6.6552 | 6.6552 | 6.6552 |
| arterial4x4 | max_pressure | 28.3550 | multi_seed | multi_seed | 34.2568 | 28.3550 | 30.3460 | 33.5786 | 37.9351 | 39.6661 | 37.9351 | 41.7664 | 41.9435 | 41.9435 | 28.3550 | 41.9435 | 41.7664 | 41.7664 |
| cologne1 | sim_generic_phase_mpc | 5.3836 | multi_seed | multi_seed | 15.4224 | 5.5613 | 5.8123 | 5.5613 | 5.3836 | 5.3836 | 5.3836 | 6.4431 | 6.4784 | 5.8614 | 5.5038 | 6.4784 | 5.7908 | 5.8863 |
| cologne3 | h2oplus_dense_phase_mpc | 5.3073 | multi_seed | multi_seed | 16.3563 | 5.4255 | 5.4255 | 5.4022 | 5.3882 | 5.3882 | 5.3882 | 5.3073 | 5.3219 | 5.4967 | 5.4245 | 5.3219 | 5.3219 | 5.3219 |
| cologne8 | cfcmt_phase_pressure_guard_mpc | 7.3608 | multi_seed | multi_seed | 25.2885 | 7.3904 | 7.5852 | 12.6837 | 7.9787 | 7.9787 | 7.9787 | 8.6007 | 8.7194 | 8.7194 | 7.3608 | 8.7194 | 8.7194 | 8.7194 |
| ingolstadt1 | sim_generic_phase_mpc | 9.0782 | multi_seed | multi_seed | 23.1803 | 9.3190 | 9.3190 | 9.3190 | 9.0782 | 9.0782 | 9.0782 | 9.3190 | 9.3190 | 9.3190 | 9.3190 | 9.3190 | 9.1208 | 9.3190 |
| ingolstadt7 | phase_spillback_pressure | 9.4259 | multi_seed | multi_seed | 38.1544 | 10.6964 | 10.7528 | 9.4259 | 13.3122 | 13.8167 | 13.8167 | 10.4551 | 11.2268 | 11.2268 | 10.2672 | 11.2268 | 13.3122 | 13.8167 |
| ingolstadt21 | h2oplus_dense_phase_mpc | 5.6268 | multi_seed | multi_seed | 29.1685 | 6.3268 | 6.3937 | 8.7752 | 9.2986 | 9.2986 | 9.2986 | 5.6268 | 6.7676 | 7.0762 | 6.3268 | 6.7613 | 6.7613 | 6.7676 |

## Source Transition Counts

| scenario | transitions |
| --- | --- |
| arterial4x4 | 272 |
| cologne1 | 18 |
| cologne3 | 54 |
| cologne8 | 144 |
| grid4x4 | 272 |
| ingolstadt1 | 18 |
| ingolstadt21 | 378 |
| ingolstadt7 | 126 |

## Notes

- `sim_generic_phase_mpc` uses a fixed generic static prior; `sim_source_avg_phase_mpc` uses only source-scenario average static summaries; `sim_target_static_phase_mpc` uses target route/network static summaries but no target next-state labels.
- `h2oplus_dense_phase_mpc` is a dense residual predictor inspired by H2O+ dynamics-gap correction.
- `cfcmt_phase_*` uses a sparse phase-local residual parent set plus optional source-trust scaling or few-shot target passive adaptation.
- `cfcmt_phase_passive_policy_selector_mpc` uses passive target snapshots to select among pressure, simulator, H2O+-style dense, and CFCMT policies under the few-shot-validated predictor; it does not inspect target closed-loop rollouts.
- The benchmark is not a faithful RESCO RL leaderboard result; it is a cross-network transfer stress test over real SUMO networks and phase programs.
