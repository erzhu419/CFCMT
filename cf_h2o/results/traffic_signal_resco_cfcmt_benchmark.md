# RESCO CFCMT Phase-Transfer Benchmark

Protocol: real RESCO SUMO scenarios, real `tlLogic` green phases as actions, leave-one-scenario-out source/target split.

Zero-shot residual policies use only source transitions and target static network/demand summaries; few-shot variants use a short passive target transition slice.

## Aggregate Metrics

| policy | mean_queue | mean_tls_queue | p90_queue | throughput_ratio |
| --- | --- | --- | --- | --- |
| cfcmt_phase_global_mpc | 7.1265 | 8.3903 | 13.0394 | 0.9096 |
| cfcmt_phase_fewshot_bias_mpc | 7.1265 | 8.3903 | 13.0394 | 0.9096 |
| cfcmt_phase_trust_mpc | 7.1955 | 8.4994 | 13.0380 | 0.9071 |
| cfcmt_phase_passive_policy_selector_mpc | 7.2132 | 8.5479 | 12.7504 | 0.9116 |
| cfcmt_phase_fewshot_selector_mpc | 7.2715 | 8.6122 | 13.0218 | 0.9101 |
| phase_pressure | 7.3115 | 8.6737 | 12.9773 | 0.9035 |
| sim_generic_phase_mpc | 7.3127 | 8.7065 | 12.9074 | 0.9092 |
| sim_source_avg_phase_mpc | 7.3127 | 8.7065 | 12.9074 | 0.9092 |
| sim_target_static_phase_mpc | 7.3127 | 8.7065 | 12.9074 | 0.9092 |
| h2oplus_dense_phase_mpc | 7.6559 | 9.0522 | 13.4804 | 0.9048 |
| phase_spillback_pressure | 8.9068 | 10.5282 | 15.9597 | 0.8958 |
| fixed_program | 28.4898 | 29.5628 | 47.9729 | 0.8527 |

## Paired Target Bootstrap

Delta is `cfcmt_phase_global_mpc - baseline` on target mean queue; negative values favor CFCMT.

| baseline | mean_delta | ci_low | ci_high | win_targets | target_count |
| --- | --- | --- | --- | --- | --- |
| phase_pressure | -0.1850 | -0.8988 | 0.3295 | 3 | 5 |
| sim_target_static_phase_mpc | -0.1862 | -0.2823 | -0.0491 | 4 | 5 |
| h2oplus_dense_phase_mpc | -0.5293 | -1.2475 | -0.0607 | 4 | 5 |
| phase_spillback_pressure | -1.7802 | -5.0996 | 0.3135 | 4 | 5 |
| fixed_program | -21.3632 | -27.5960 | -16.2052 | 5 | 5 |

## Target Metrics

| target | best_policy | best_queue | selector | policy_selector | fixed_program | phase_pressure | phase_spillback_pressure | sim_generic_phase_mpc | sim_source_avg_phase_mpc | sim_target_static_phase_mpc | h2oplus_dense_phase_mpc | cfcmt_phase_global_mpc | cfcmt_phase_trust_mpc | cfcmt_phase_fewshot_bias_mpc | cfcmt_phase_fewshot_selector_mpc | cfcmt_phase_passive_policy_selector_mpc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cologne1 | cfcmt_phase_global_mpc | 8.7463 | sim_source_avg_phase_mpc | sim_target_static_phase_mpc | 33.9162 | 10.3150 | 9.6576 | 9.0416 | 9.0416 | 9.0416 | 10.6843 | 8.7463 | 9.3498 | 8.7463 | 9.3330 | 9.0416 |
| cologne3 | cfcmt_phase_trust_mpc | 6.3591 | h2oplus_dense_phase_mpc | h2oplus_dense_phase_mpc | 24.9694 | 6.5080 | 6.9183 | 6.6330 | 6.6330 | 6.6330 | 6.6732 | 6.4614 | 6.3591 | 6.4614 | 6.6732 | 6.6732 |
| cologne8 | phase_pressure | 7.1975 | cfcmt_phase_fewshot_bias_mpc | cfcmt_phase_global_mpc | 24.6805 | 7.1975 | 15.7010 | 7.7627 | 7.7627 | 7.7627 | 7.5543 | 7.5084 | 7.5084 | 7.5084 | 7.5084 | 7.5084 |
| ingolstadt1 | h2oplus_dense_phase_mpc | 3.8174 | cfcmt_phase_trust_mpc | h2oplus_dense_phase_mpc | 17.9042 | 3.9164 | 4.0671 | 4.1008 | 4.1008 | 4.1008 | 3.8174 | 3.8174 | 3.8174 | 3.8174 | 3.8174 | 3.8174 |
| ingolstadt7 | phase_spillback_pressure | 8.1898 | sim_generic_phase_mpc | sim_target_static_phase_mpc | 40.9786 | 8.6207 | 8.1898 | 9.0254 | 9.0254 | 9.0254 | 9.5503 | 9.0992 | 8.9431 | 9.0992 | 9.0254 | 9.0254 |

## Source Transition Counts

| scenario | transitions |
| --- | --- |
| cologne1 | 60 |
| cologne3 | 180 |
| cologne8 | 480 |
| ingolstadt1 | 60 |
| ingolstadt7 | 420 |

## Notes

- `sim_generic_phase_mpc` uses a fixed generic static prior; `sim_source_avg_phase_mpc` uses only source-scenario average static summaries; `sim_target_static_phase_mpc` uses target route/network static summaries but no target next-state labels.
- `h2oplus_dense_phase_mpc` is a dense residual predictor inspired by H2O+ dynamics-gap correction.
- `cfcmt_phase_*` uses a sparse phase-local residual parent set plus optional source-trust scaling or few-shot target passive adaptation.
- `cfcmt_phase_passive_policy_selector_mpc` uses passive target snapshots to select among pressure, simulator, H2O+-style dense, and CFCMT policies under the few-shot-validated predictor; it does not inspect target closed-loop rollouts.
- The benchmark is not a faithful RESCO RL leaderboard result; it is a cross-network transfer stress test over real SUMO networks and phase programs.
