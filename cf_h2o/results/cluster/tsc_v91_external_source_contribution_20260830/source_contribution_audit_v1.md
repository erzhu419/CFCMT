# V91 fresh source-contribution confirmation

- evidence integrity: `PASS`
- joint scientific gate: `False`
- decision: `reject_fresh_source_contribution_confirmation`
- fresh seeds: 64
- CFCMT macro waiting: 155.144562 s
- target-only macro waiting: 149.092079 s
- mean relative delta: 4.2545%
- paired-seed 95% interval: [-0.7748%, 13.2108%]
- improved seeds: 29/64
- one-sided sign-test p: 0.809134
- worst seed: 267.5565%
- city mean deltas: {"jinan": -0.030446469768574826, "los_angeles": 0.0786534191826514}
- collision incidents: {"causal_target_only_mpc": 15, "cfcmt_selected_mpc": 25}

## Gate details

- mean_relative_delta: `False`
- bootstrap_upper: `False`
- each_city_mean: `False`
- worst_seed: `False`
- improved_seed_fraction: `False`
- one_sided_sign_test: `False`
- aggregate_teleports: `True`
- aggregate_collision_safety: `False`
- complete_identity_matrix: `True`
- no_failed_seed_exclusion: `True`
- method_active: `True`

## Boundary

V91 independently tests whether the frozen v43 source-plus-target CFCMT controller improves over the already frozen target-only family on new simulator seeds in the same Los Angeles and Jinan benchmark geometries. It can establish seed-level source contribution in these targets, not population-level transfer to unseen cities or field effectiveness.
