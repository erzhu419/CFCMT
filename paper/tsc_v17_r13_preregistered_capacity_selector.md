# TSC v17 r13 Target-Capacity Selector Protocol

Date fixed: 2026-08-08, after inspecting r12 development budgets 0, 8, 16, 32, and 60, and before any r13 rollout.

## Development finding that motivates r13

The r12 hierarchy proved that the target-only expert contains useful target information, but it also exposed a structural error in the combination rule. Independently calibrated mechanism, target-specialist, and rigid-core confidence bounds were compared directly at every state. Those bounds share nominal units but not a common cross-family calibration distribution. At budget 32, rigid-core actions that passed target calibration reintroduced source bias in several target networks; at budget 60, the full guard remained slightly worse and more geographically concentrated than the standalone target-only guard. The mechanism expert was almost always disabled and did not explain the observed high-budget gain.

## Frozen r13 method

r13 replaces per-state cross-family priority competition with a target-level capacity choice made only from disjoint target calibration groups.

1. Without a fitted target head, the rigid causal core is the anchor.
2. With a fitted positive-weight target head, the target-only causal specialist replaces the source-only core as the anchor. The core cannot re-enter the few-shot deployment path.
3. The mechanism expert can replace an available anchor only when its robust held-out calibration score improves on the anchor by at least 0.01 normalized action-cost units.
4. If the calibrated anchor is unavailable, the controller falls back to the selected pressure prior; the mechanism expert cannot deploy alone.
5. Exactly one learned expert is enabled for each target and control mode. Its existing guard or regularizer remains unchanged, and all rejected actions fall back to the pressure prior.

This selector uses no evaluation-seed labels, no target closed-loop policy rollouts, and no calibration-group refit. It changes only the deployment capacity choice; source counterfactuals, target adaptation groups, target calibration groups, phase actions, pressure prior, and safety coordination are unchanged.

## Frozen controlled protocol

- Hierarchy protocol: `disjoint-target-calibrated-single-expert-mechanism-target-rigid-pressure-v2`
- Capacity-selection protocol: `disjoint-target-group-anchor-first-single-expert-v1`
- SUMO/libsumo: 1.22.0
- Networks: 16 networks in 6 city groups
- Development evaluation seeds: 131, 337, 911
- Target group budgets: 0, 8, 16, 32, 60, 120
- Target calibration seed: 4047, disjoint from evaluation
- Counterfactual and source-rule caches: unchanged audited r12 inputs
- Target closed-loop selection seeds: none
- Development workers: 96 per budget job; this is a runtime-only change
- Snapshot SHA-256: `3855769c6f6a48b3f64a438268bebaae7ec0c904932a32e06c22d65722e488ff`
- Source-tree SHA-256: `349bc698337ea75e70ed9412f5ae0075ce3e1b126a2791e1192b249be4141e6e`

## Acceptance gates

Every result must pass source hash, protocol, seed separation, finite-value, rollout-completeness, fixed-policy invariance, and guarded-policy collision/teleport audits.

Budgets 60 and 120 must satisfy the unchanged r12 primary gates for `cfcmt_mechanism_contrast_guard`:

- negative mean and median paired city effects relative to `selected_source_prior`;
- at least 4 of 6 city groups improve;
- worst-city degradation is no greater than 0.5%;
- top-gain city share is below 90%;
- mean effect is no worse than target-only guard by more than 0.2%;
- zero collision and teleport events for deployable guarded rows.

Budget 32 is a secondary transition-regime check. A mechanism expert activation is credible only if the serialized selector diagnostics show the frozen 0.01 calibration margin and its held-out evaluation does not violate the city-level safety gate. If r13 merely reproduces target-only performance, the correct interpretation is that target-specialist transfer is the supported method component and MC-WM mechanism refinement remains an unsupported ablation on this benchmark.

The 600-second matrix is developmental. No formal paper claim will use these seeds; the long-horizon confirmatory matrix must be frozen with fresh evaluation seeds after r13 is accepted.
