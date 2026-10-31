# TSC v18 r14 Cross-Fitted Source-Mechanism/Target-Specialist Fusion Protocol

Date fixed: 2026-08-08, after completing the r13 six-budget audit and before implementing or evaluating the r14 family.

## Motivation fixed from r13

r13 made a target-level choice between independently calibrated experts. It was safe and improved broadly at 60 and 120 target groups, but the mechanism expert was selected zero times and the nominal full CFCMT guard was exactly equal to target-only at all budgets. The failure is architectural: choosing between complete policies cannot establish that invariant source mechanisms complement target history.

r14 tests one new module only: cross-fitted convex fusion of the frozen source mechanism predictor and a low-capacity target specialist. Source data, target group construction, calibration split, pressure prior, phase executor, coordination graph, action space, metrics, and evaluation seeds remain unchanged for the development test.

## Frozen r14 model

The new family is `cfcmt_fused`.

1. The source branch is the existing causal mechanism advantage model: a rigid causal action core plus auxiliary queue, spillback, clearance, and system-load mechanisms. Mechanism stacking is selected only by nested leave-one-source-city cross-fitting.
2. The target branch is the same low-capacity causal target specialist used by the target-only ablation. It is trained only on target adaptation groups and receives no source rows.
3. Target adaptation groups are partitioned by action-group GroupKFold. Every target-specialist prediction used to choose a fusion weight is out-of-fold with respect to that group.
4. Candidate specialist weights are 0, 0.10, 0.25, 0.50, 0.75, and 1.00, capped by `n_target_groups / (n_target_groups + 8)`. The cap therefore relaxes monotonically as target history increases.
5. A positive weight is selected only if cross-fitted mean action regret improves by at least 0.002 normalized action units, no more than 35% of groups are harmed, and worst-group regret is no more than 0.05 above the source-mechanism baseline. Ties select the smaller target weight.
6. The final target specialist is fitted on all adaptation groups after weight selection. Held-out target calibration groups are not refitted into either branch.
7. Prediction uncertainty combines source and target uncertainty plus source-target disagreement. Reference actions remain exactly zero by construction.
8. The disjoint target calibration groups select the guard or regularizer for the already-fused family. They do not choose the fusion weight.

The deployable r14 primary is `cfcmt_fused_contrast_guard`. It is a single calibrated predictor and falls back directly to the selected source pressure prior when rejected. It does not use the r13 per-family hierarchy. `cfcmt_mechanism`, `causal_target_only`, and `causal_rigid_advantage` remain explicit ablations.

## Controlled development protocol

- Networks: the existing audited 16-network, six-city-group manifest. Salt Lake is excluded until its separate data-admission gates pass.
- SUMO/libsumo: 1.22.0 with the unchanged no-teleport, write-unfinished protocol.
- Development evaluation seeds: 131, 337, and 911.
- Target group budgets: 0, 8, 16, 32, 60, and 120.
- Target calibration seed: 4047, disjoint from evaluation.
- Source counterfactual and rule-policy caches: immutable r13 inputs.
- Primary metric: fixed-horizon `mean_system_vehicles_per_controlled_lane`.
- Runtime parallelism may change; statistical and information protocols may not.

## Acceptance gates

Every budget must pass the existing source hash, seed separation, finite value, rollout completeness, fixed-policy invariance, collision, and teleport audits.

At budgets 60 and 120, `cfcmt_fused_contrast_guard` must have negative mean and median effects versus `selected_source_prior`, improve at least 4 of 6 city groups, degrade the worst city by no more than 0.5%, and keep the top-gain city share below 90%.

The mechanism contribution is accepted only if at least one of budgets 16, 32, 60, or 120 improves mean paired city effect over target-only by at least 0.1 percentage point without reducing the number of improved city groups or violating the worst-city gate. At least one target must select a strictly intermediate fusion weight, proving that the primary is not merely target-only under another name.

Failure of this contribution gate rejects r14 as the paper's full mechanism method. It may not be rescued by evaluation-label policy selection, target closed-loop tuning, or relabeling the target-only result.
