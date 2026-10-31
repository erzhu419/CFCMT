# TSC v19 r15 Junction-Collision-Audited Fused CFCMT Protocol

Date fixed: 2026-08-08, after discovering the collision-monitoring defect and before building any r15 cache or running any r15 model evaluation.

## Protocol correction

The previous SUMO launch protocol disabled teleportation and wrote unfinished trips, but it did not enable junction collision checks. A controlled 600 s Salt Lake replay of the same native signal program reported zero collisions under the previous launch and five collisions after adding `--collision.check-junctions true`. Therefore, all pre-r15 zero-collision statements mean only that no collisions were exposed by SUMO's default detector. They are not accepted as junction-safety evidence.

r15 adds `--collision.action warn` and `--collision.check-junctions true` to every source collection, source-rule rollout, calibration rollout, and evaluation rollout. The protocol identity, counterfactual cache version, and source-rule cache version are changed. No pre-r15 cache is admissible under r15.

## Frozen method and information protocol

The method is the r14 `cfcmt_fused` model: nested source-city cross-fitted causal mechanisms fused with a group-out-of-fold low-capacity target specialist. Target specialist weights are 0, 0.10, 0.25, 0.50, 0.75, and 1.00, capped by `n/(n+8)`. Selection requires at least 0.002 mean normalized action-regret gain, at most 35% harmed groups, and no more than 0.05 worst-group regret above the source-mechanism branch. Calibration groups remain disjoint from adaptation groups and choose only the deployment guard.

The 16-network, six-city-group manifest, source seeds, target budgets, calibration seed, evaluation seeds, action space, pressure prior, metric, and city-first aggregation are unchanged. The explicit primary is `cfcmt_fused_contrast_guard`.

## Sequential gates

1. Rebuild and hash-audit all source counterfactual and source-rule caches under the r15 launch protocol.
2. Require zero junction collisions and zero teleports from deployable source behavior and rule policies. A failure stops model evaluation until the responsible network or controller is corrected.
3. Run the six-budget development matrix only after both cache audits pass.
4. Apply the r14 efficacy and contribution gates unchanged: at budgets 60 and 120, negative mean and median effect versus selected source prior, at least four of six improved city groups, worst-city degradation at most 0.5%, and top-city gain share below 90%. Mechanism contribution additionally requires at least 0.1 percentage point mean gain over target-only at one of budgets 16, 32, 60, or 120, no fewer city wins, and at least one strictly intermediate target fusion weight.

Failure is reported as a failed method or protocol gate. Evaluation labels may not tune the method, fusion weights, guard, or acceptance thresholds.

## Pre-cache safety amendment

Before any r15 cache was built, the corrected detector found one repeated junction-collision incident in the Salt Lake State x University `phase_pressure` admission run. The collision involved a vehicle still traversing a downstream internal junction segment after the fixed yellow plus all-red interval. The executor is therefore frozen with full-junction occupancy clearance: it discovers every internal lane sharing the controlled-link junction prefix and extends all-red in one-second increments until all such lanes are empty. Source behavior and every counterfactual branch fail fast on any junction collision or teleport. The phase-pressure score also excludes right-on-red `s` states from controllable green benefit. A same-seed 3600 s replay after the amendment had zero collisions and teleports, with 54 one-second occupancy extensions. This amendment precedes all r15 cache, source-rule, model, and evaluation jobs.

## Protocol rejection and successor

The r15 zero-raw-collision gate was rejected before cache admission. After the Salt Lake amendment, the corrected detector showed that Cologne1's unmodified native fixed program also produces junction collision records, and matched step-length checks did not remove them monotonically. Consequently, raw SUMO benchmark events cannot be treated as controller-attributable failures without a reference condition.

No r15 cache or model result is accepted for the submission. The successor r16/v20 protocol is specified in `cf_h2o/config/traffic_signal_tsc_v16_benchmark_aware_development.json`. It retains collision detection, keeps teleport as a hard failure, censors complete matched counterfactual action groups on any branch collision, reports deduplicated incident signatures, and applies paired collision-incident noninferiority against `selected_source_prior` to the primary deployable policy. The method architecture and efficacy gates remain frozen; only the invalid safety estimand and cache identities are replaced before rerunning the experiment.
