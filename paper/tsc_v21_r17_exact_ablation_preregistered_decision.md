# TSC v21/r17 Exact-Mechanism Ablation Decision Log

Recorded before any v21 development result existed: 2026-08-08 (Asia/Shanghai).

## Frozen inputs

- Development configuration: `cf_h2o/config/traffic_signal_tsc_v17_exact_mechanism_ablation_development.json`
- Matrix protocol tag: `v21`
- Snapshot SHA-256: `7b5bb44cf0fbfd16b40da1d6aea591066e40d372ddfb435d4be0c3d71e9c6cfd`
- Source-tree SHA-256: `e53c8ec17cee2e687025c06146478ca81e2677bb6c27bf39fa38e976e26d2793`
- Counterfactual/source-rule cache identity: `tsc_v20r16_benchmark_aware_dev600_20260808`
- Development evaluation seeds: `131`, `337`, and `911`
- Target counterfactual-group budgets: `0`, `8`, `16`, `32`, `60`, and `120`
- Primary policy: `cfcmt_fused_contrast_guard`
- Exact source-mechanism ablation: `cfcmt_fused_rigid_contrast_guard`

All six physical nodes passed the same snapshot hash, source hash, SUMO/libsumo
1.22.0, 192-CPU, and 16-scenario preflight before the matrix was submitted.

## Question isolated by r17

The fused-rigid policy is identical to the fused policy in its rigid source
core, target-specialist architecture, target-group split, out-of-fold target
weight selection, uncertainty propagation, guard interface, action set, and
evaluation seeds. Its source mechanism-stack weight is forced to zero after
the same source fit. Therefore, the paired fused-minus-fused-rigid effect is
the policy-level contribution of the selected source mechanism path under the
frozen protocol. It is not a generic causal-versus-dense comparison.

## Result-independent decision rules

1. Proceed directly to a fresh-seed, 3,600-second confirmatory matrix only if
   integrity, paired collision-incident non-inferiority, efficacy breadth, and
   the preregistered mechanism-contribution gate all pass.
2. If fused is worse than fused-rigid or target-only on targets where the
   source mechanism stack is active, add a target-adaptation-only mechanism
   applicability gate. The gate must be selected from target adaptation groups
   by grouped out-of-fold regret and must be evaluated in a new development
   matrix. No target evaluation rollout may enter this gate.
3. If the primary policy fails collision/service non-inferiority, use disjoint
   target closed-loop policy-selection seeds and compare each residual policy
   to its actual `selected_source_prior` reference. These simulator rollouts
   must be reported as an additional target-information budget; they may not be
   hidden inside the counterfactual-group count.
4. A failed city or network may not be removed. A failed metric may not be
   replaced. Thresholds may not be chosen from the development evaluation
   rollouts and then reported as confirmatory evidence on those same rollouts.
5. Any method change after inspecting seeds `131/337/911` makes those seeds
   development-only. Final claims then require new evaluation seeds, a new
   immutable snapshot, and a full independent audit.

## Evidence already known when this log was written

The incomplete r20 diagnostic matrix had valid results through budget 60. At
budget 60 the primary reduced city-first mean system load by 1.63% relative to
the selected source prior, but 97.3% of positive gain came from Atlanta, the
worst city worsened by 0.558%, and collision incidents were 45 versus 40 for
the paired source prior. Atlanta target-only also outperformed fused by 3.18%.
These observations motivate the exact ablation and safety checks above; they
are not accepted method evidence and do not change the frozen r21 thresholds.
