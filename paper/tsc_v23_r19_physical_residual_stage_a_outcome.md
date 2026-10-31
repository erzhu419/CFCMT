# TSC v23/r19 Physical-Residual Stage-A Outcome

Date: 2026-08-09

## Frozen protocol and integrity

The preregistered Stage-A screen used one representative network from each of
six city groups, zero target adaptation groups, and complete city-group
holdout. The authoritative rank-0 result is
`cf_h2o/results/cluster/tsc_v23_physical_offline_probe/six_city_budget000_rank0_process_v2.json`.
All four follow-up diagnostics used the same immutable 768-file, 32,394-row
counterfactual cache and did not run or modify SUMO.

Every target evaluated all retained action groups; no target groups were used
for fitting or excluded from evaluation. All family metrics are finite and
share exactly the same group set within each target. The six target group
counts are 648, 76, 90, 609, 631, and 572.

The occupancy-unit audit was repeated against the exact HPC SUMO/libsumo
1.22.0 runtime. For an occupied 272.8 m lane containing four 5 m vehicles,
`getLastStepOccupancy` returned `0.07331378299120234`, exactly
`20 / 272.8`, rather than `100 * 20 / 272.8`. The physical-residual transform
therefore correctly preserves occupancy as a ratio in this runtime.

## Preregistered Stage-A decision

| Target | Rigid regret | Full physical regret | Absolute change | Relative improvement |
|---|---:|---:|---:|---:|
| RESCO grid4x4 | 0.345286 | 0.244438 | -0.100848 | +29.21% |
| Cologne1 | 0.285399 | 0.298180 | +0.012782 | -4.48% |
| Ingolstadt1 | 0.199679 | 0.156925 | -0.042754 | +21.41% |
| Atlanta 1x5 | 0.215572 | 0.513117 | +0.297545 | -138.03% |
| Hangzhou 4x4 | 0.367733 | 0.357718 | -0.010016 | +2.72% |
| Manhattan 28x7 | 0.333154 | 0.322300 | -0.010854 | +3.26% |
| **City macro mean** | **0.291137** | **0.315446** | **+0.024309** | **-8.35%** |

The full rank-0 physical-residual model improves four of six cities but makes
the macro mean 8.35% worse and increases Atlanta regret by 0.2975. It therefore
fails both the preregistered 10% macro-improvement condition and the maximum
0.05 absolute-regression condition. Stage A is **FAIL**. The model is not
eligible for Stage B, latent augmentation, or closed-loop confirmation.

## Post-failure diagnostics

Fixed blending of the full physical score into rigid did not rescue the gate.
The best tested blend, weight 0.5, improved the macro mean by 3.96% and improved
five cities, but Atlanta still regressed by 0.05075 and the 10% mean gate
failed. A target-label-free prediction-disagreement cap also failed; its best
setting improved the macro mean by only 1.68%, with a 0.05392 Atlanta
regression. These results rule out treating the failure as only excessive
score magnitude or generic OOD uncertainty.

Exact single-mechanism attribution is stored in
`cf_h2o/results/cluster/tsc_v23_physical_offline_probe/six_city_budget000_mechanism_attribution_process_v5.json`.

| Variant | Macro regret | Improvement vs rigid | Cities improved | Max absolute regression |
|---|---:|---:|---:|---:|
| Rigid | 0.291137 | 0.00% | 0/6 | 0.000000 |
| Served movement only | 0.294655 | -1.21% | 2/6 | 0.069338 |
| Mobility only | 0.274374 | +5.76% | 5/6 | 0.011636 |
| Full five-mechanism stack | 0.315446 | -8.35% | 4/6 | 0.297545 |

Mobility alone improves Atlanta by 13.64%, whereas the full stack worsens it by
138.03%. Served movement alone is identical to rigid in Atlanta because its
source-city stack gate disables that component. Thus neither mobility nor the
served-movement residual alone explains the catastrophic interaction. The
remaining candidates are the queue-propagation, red-accumulation, and
spillback analytic-prior columns admitted by the joint stack, plus their
joint linear extrapolation.

## Completed component attribution and next admissible experiment

The exact follow-up is complete and independently audited in
`paper/tsc_v24_r20_component_attribution_outcome.md`. Six shards and all 42
new family results passed SHA-256, zero-target-budget, complete-action-group,
and finite-metric checks. Queue-only, red-only, and spillback-only each cause a
large Atlanta regression. Adding any one of those mechanisms to mobility also
recreates the failure. Mobility-only is the sole robust diagnostic (5.76%
macro improvement, five of six cities improved, 0.0116 maximum regression),
but still fails the preregistered 10% macro-improvement threshold.

No current physical family is promoted. The next admissible experiment is the
pre-specified estimand repair: retain the 60 s first-action-then-pressure
rollout cost for action ranking, but collect local mechanism labels immediately
after the first 10 s control interval and compute their analytic priors at the
same 10 s horizon. This tests the temporal-consistency hypothesis without
changing the Stage-A gate.
