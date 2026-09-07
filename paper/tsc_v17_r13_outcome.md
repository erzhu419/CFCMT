# TSC v17 r13 Outcome: Target Capacity Without Mechanism Gain

Date evaluated: 2026-08-08. This report was written after all six preregistered development budgets completed and before implementing r14.

## Integrity result

The six-budget matrix at 0, 8, 16, 32, 60, and 120 target groups passed the strict result audit. Every budget contains 1,344 expected policy-seed-network rollouts, uses SUMO/libsumo 1.22.0, matches source-tree SHA-256 `349bc698337ea75e70ed9412f5ae0075ce3e1b126a2791e1192b249be4141e6e`, preserves source/evaluation/calibration seed separation, and has zero collision or teleport events in deployable guarded rows.

The legacy matrix reporter selected `cfcmt_mechanism_contrast_regularized` as its automatic primary because the v13 JSON did not explicitly declare a primary policy. That automatic summary is not the preregistered r13 test. Future version-4 specifications require an explicit primary policy and audit it at every budget.

## Preregistered guard result

All effects below are paired city-group relative changes in `mean_system_vehicles_per_controlled_lane` versus `selected_source_prior`; negative is better.

| Target groups | Mean | Median | Improved cities | Worst city | Top-gain share | Target-specialist networks | Mechanism selections |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000% | 0.0000% | 0/6 | 0.0000% | 0.00% | 0/16 | 0 |
| 8 | -0.6133% | 0.0000% | 1/6 | +0.1329% | 100.00% | 3/16 | 0 |
| 16 | -1.1701% | 0.0000% | 1/6 | +0.1918% | 100.00% | 4/16 | 0 |
| 32 | -2.1106% | -0.0387% | 3/6 | 0.0000% | 94.68% | 6/16 | 0 |
| 60 | -2.7143% | -0.5935% | 5/6 | 0.0000% | 87.11% | 9/16 | 0 |
| 120 | -2.6287% | -0.7165% | 5/6 | +0.0051% | 84.39% | 11/16 | 0 |

Budgets 60 and 120 pass the preregistered breadth, concentration, worst-city, and safety gates. They also satisfy the intended data ladder: the number of networks allowed to deploy a target specialist rises from 3 to 11 as the target information budget increases.

## Mechanism falsification result

At every budget and every network, `cfcmt_mechanism_contrast_guard` is numerically identical to `causal_target_only_contrast_guard`. The target-level selector never selected the mechanism expert. When the target-only guard passed held-out calibration, r13 deployed it; otherwise it fell back to the pressure prior. Thus r13 does not support a claim that MC-WM mechanism refinement improves target adaptation.

This is not a safety failure or an evaluation-label leak. It is a method-identification failure: the nominal full policy reduces exactly to a lower-capacity ablation. The supported r13 result is conservative target-specialist activation, not causal mechanism fusion. The paper must not present r13 as evidence for a mechanism contribution.

## Consequence

r13 is retained as a negative ablation and as evidence that post-hoc expert selection cannot rescue a mechanism model whose target residual is weaker than a pure target specialist. r14 must combine the invariant source mechanism and target specialist inside one cross-fitted predictor, then calibrate that predictor as one family. A successful r14 must show a strict, geographically non-concentrated gain over target-only under the same target-information budget; otherwise the mechanism claim is rejected.
