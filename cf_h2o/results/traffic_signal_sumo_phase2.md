# Traffic Signal SUMO Phase 2

This is a multi-intersection microscopic SUMO benchmark. It is still synthetic, but it uses OD-style routes that cross multiple controlled signals.

Protocol: generated 4x4 grid, four center intersections controlled by libsumo, leave-one-scenario-out transfer. The optional oracle row is a short-horizon SUMO counterfactual MPC computed with saveState/loadState.

## Aggregate Policy Metrics

| policy | mean_cost | mean_queue | p90_queue | throughput_ratio | ns_action_share |
| --- | --- | --- | --- | --- | --- |
| sim_mpc | 127.6722 | 119.2837 | 149.6997 | 0.8533 | 0.4518 |
| sim_generic_mpc | 128.5864 | 120.2246 | 149.5695 | 0.8535 | 0.4536 |
| cfcmt_trust_mpc | 128.6887 | 120.2475 | 150.5167 | 0.8533 | 0.4455 |
| cfcmt_fewshot_selector_mpc | 128.6965 | 120.3496 | 149.9905 | 0.8521 | 0.4445 |
| cfcmt_fewshot_bias_mpc | 128.7996 | 120.4280 | 149.8803 | 0.8501 | 0.4445 |
| cfcmt_global_mpc | 129.0055 | 120.5991 | 151.0125 | 0.8532 | 0.4473 |
| spillback_pressure | 130.0079 | 121.4632 | 152.7532 | 0.8538 | 0.4627 |
| sim_source_avg_mpc | 130.0294 | 121.5828 | 147.3013 | 0.8493 | 0.4536 |
| h2oplus_dense_mpc | 131.5991 | 122.8948 | 154.5703 | 0.8499 | 0.4455 |
| local_pressure | 131.8001 | 123.3163 | 153.8749 | 0.8515 | 0.4600 |
| fixed_offset | 160.0160 | 145.1386 | 186.0040 | 0.8337 | 0.5000 |

## Target-Level Results

| target | best_policy | best_cost | fixed_offset | spillback_pressure | sim_generic_mpc | sim_source_avg_mpc | sim_mpc | h2oplus_dense_mpc | cfcmt_global_mpc | cfcmt_trust_mpc | cfcmt_fewshot_selector_mpc | oracle_mpc | selector |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_corridors | sim_source_avg_mpc | 108.7872 | 131.3664 | 109.3322 | 112.7133 | 108.7872 | 111.5221 | 113.4207 | 114.1952 | 110.3029 | 116.8175 | 151.5283 | cfcmt_global_mpc |
| ns_commute | sim_generic_mpc | 104.6778 | 122.7224 | 114.1551 | 104.6778 | 107.7787 | 116.5051 | 113.4318 | 116.2195 | 111.2804 | 105.4415 | nan | cfcmt_trust_mpc |
| ew_commute | sim_mpc | 111.1699 | 160.0075 | 116.1229 | 124.5646 | 126.6490 | 111.1699 | 121.1322 | 113.0592 | 115.8711 | 122.1582 | nan | cfcmt_fewshot_bias_mpc |
| downtown_bottleneck | sim_mpc | 160.3576 | 203.0048 | 166.2426 | 162.2231 | 165.0792 | 160.3576 | 172.7237 | 163.8046 | 170.2389 | 162.7108 | nan | cfcmt_trust_mpc |
| event_reversal | cfcmt_trust_mpc | 135.7503 | 182.9788 | 144.1866 | 138.7533 | 141.8527 | 138.8061 | 137.2871 | 137.7491 | 135.7503 | 136.3545 | nan | cfcmt_global_mpc |

## Notes

- Simulator information budget: `sim_generic_mpc` uses a fixed generic demand prior, `sim_source_avg_mpc` uses source-domain average demand summaries, and `sim_mpc` uses the target static demand/bottleneck summary.
- The aggregate excludes oracle unless it was evaluated for every target; by default oracle is run on the first target only as a counterfactual sanity check.
- The oracle row is a short-horizon myopic counterfactual controller, not a full-horizon performance upper bound.
- Few-shot selector uses passive target transitions only; it does not inspect target policy rollouts.
- This benchmark tests network-level closed-loop behavior but does not yet use a real-world TSC benchmark topology.
