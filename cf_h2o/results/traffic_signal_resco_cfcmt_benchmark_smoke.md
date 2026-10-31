# RESCO CFCMT Phase-Transfer Benchmark

Protocol: real RESCO SUMO scenarios, real `tlLogic` green phases as actions, leave-one-scenario-out source/target split.

Zero-shot residual policies use only source transitions and target static network/demand summaries; few-shot variants use a short passive target transition slice.

## Aggregate Metrics

| policy | mean_queue | mean_tls_queue | p90_queue | throughput_ratio |
| --- | --- | --- | --- | --- |
| cfcmt_phase_global_mpc | 6.5984 | 6.7824 | 10.9661 | 0.7684 |
| cfcmt_phase_trust_mpc | 6.5984 | 6.7824 | 10.9661 | 0.7684 |
| cfcmt_phase_fewshot_bias_mpc | 6.5984 | 6.7824 | 10.9661 | 0.7684 |
| cfcmt_phase_passive_policy_selector_mpc | 6.5984 | 6.7824 | 10.9661 | 0.7684 |
| h2oplus_dense_phase_mpc | 6.5984 | 6.7824 | 10.9661 | 0.7684 |
| sim_generic_phase_mpc | 6.9987 | 7.2361 | 11.6139 | 0.7635 |
| sim_source_avg_phase_mpc | 6.9987 | 7.2361 | 11.6139 | 0.7635 |
| sim_target_static_phase_mpc | 6.9987 | 7.2361 | 11.6139 | 0.7635 |
| cfcmt_phase_fewshot_selector_mpc | 6.9987 | 7.2361 | 11.6139 | 0.7635 |
| phase_spillback_pressure | 7.0584 | 7.5374 | 11.6141 | 0.7736 |
| phase_pressure | 7.2265 | 7.6965 | 11.9920 | 0.7736 |
| fixed_program | 18.6446 | 19.0603 | 33.8111 | 0.6016 |

## Target Metrics

| target | best_policy | best_queue | selector | policy_selector | fixed_program | phase_pressure | phase_spillback_pressure | sim_generic_phase_mpc | sim_source_avg_phase_mpc | sim_target_static_phase_mpc | h2oplus_dense_phase_mpc | cfcmt_phase_global_mpc | cfcmt_phase_trust_mpc | cfcmt_phase_fewshot_bias_mpc | cfcmt_phase_fewshot_selector_mpc | cfcmt_phase_passive_policy_selector_mpc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cologne1 | sim_generic_phase_mpc | 4.8851 | sim_generic_phase_mpc | sim_target_static_phase_mpc | 14.5233 | 6.1411 | 5.8051 | 4.8851 | 4.8851 | 4.8851 | 4.8851 | 4.8851 | 4.8851 | 4.8851 | 4.8851 | 4.8851 |
| ingolstadt1 | phase_pressure | 8.3118 | sim_target_static_phase_mpc | phase_spillback_pressure | 22.7659 | 8.3118 | 8.3118 | 9.1124 | 9.1124 | 9.1124 | 8.3118 | 8.3118 | 8.3118 | 8.3118 | 9.1124 | 8.3118 |

## Source Transition Counts

| scenario | transitions |
| --- | --- |
| cologne1 | 18 |
| ingolstadt1 | 18 |

## Notes

- `sim_generic_phase_mpc` uses a fixed generic static prior; `sim_source_avg_phase_mpc` uses only source-scenario average static summaries; `sim_target_static_phase_mpc` uses target route/network static summaries but no target next-state labels.
- `h2oplus_dense_phase_mpc` is a dense residual predictor inspired by H2O+ dynamics-gap correction.
- `cfcmt_phase_*` uses a sparse phase-local residual parent set plus optional source-trust scaling or few-shot target passive adaptation.
- `cfcmt_phase_passive_policy_selector_mpc` uses passive target snapshots to select among pressure, simulator, H2O+-style dense, and CFCMT policies under the few-shot-validated predictor; it does not inspect target closed-loop rollouts.
- The benchmark is not a faithful RESCO RL leaderboard result; it is a cross-network transfer stress test over real SUMO networks and phase programs.
