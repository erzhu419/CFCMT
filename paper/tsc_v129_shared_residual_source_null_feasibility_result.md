# TSC V129 Shared-Residual Source-Null Feasibility Result

## Decision

Reject. V129 does not authorize a target-budget curve, source integration or
untouched-city confirmation.

## Execution record

- Initial task `t88665` completed all 22 folds but failed before publishing
  because the final call supplied an unsupported keyword to the project's
  already-strict atomic JSON writer.
- Automatic identical retry `t88697` was force-cancelled.
- Corrected task: `t88699`
- Node: `node003`
- Runtime: 565.65 s
- Peak RAM: 8,501 MB
- Snapshot: `197b16ca6e729db2af6b`
- Signature:
  `CFCMT/v129/shared-target-residual-source-null-b500-v2-execution-safe`
- Result:
  `cf_h2o/results/cluster/tsc_v129_shared_residual_source_null_feasibility_20260901/development_v2_execution_safe/result.json`
- Result SHA-256:
  `d0724b1e211944713bc424bfe701885acd2a67478836c2a1e45e492794bfb237`

The execution correction changed neither the method nor any scientific gate.
Only the 517,481-byte result JSON was retrieved.

## Main results

| Arm | Mean normalized delta | 95% interval | Enabled folds | Interventions |
|---|---:|---:|---:|---:|
| target-only | +0.0002073 | [-0.0000984, +0.0006630] | 4/22 | 40 |
| source-aligned | +0.0000346 | [0, +0.0001038] | 1/22 | 8 |
| source-placebo | +0.0001161 | [0, +0.0003313] | 2/22 | 25 |
| adaptive source-null | +0.0002419 | [-0.0000686, +0.0007044] | 5/22 | 48 |
| adaptive placebo-null | +0.0003233 | [-0.0000297, +0.0008094] | 6/22 | 65 |

Shared residuals made the prior arms distinguishable, unlike V128. However,
the aligned source was selected in only one fold, held-out seed `47023`, where
its eight interventions were harmful by `+0.0007615`. Adaptive source-null was
worse than target-only by `+0.0000346`; it did not beat the placebo null with a
strictly negative upper bound.

## Interpretation

V129 confirms that separate residual cancellation was a real V128 design
problem, but not the only problem. Even after removing it, the target residual
and source selector do not identify interventions that generalize across
traffic seeds. The one selected source fold changed from a ten-seed calibration
gain of `-0.0007834` to a harmful held-out outcome. This is action-identification
instability, not a source-weight tuning problem.

## Authorized successor

Before any further source fusion, V130 must compare high-precision target-side
intervention models under the same 11-train/10-calibration/1-held-out seed
partition and pressure constraint. At minimum it should test a causal binary
improvement classifier and a group-normalized sign-balanced advantage
regressor. Source predictions must not enter this screen. Only a model that
beats PhasePressure under the held-out absolute gate may proceed to a separate
source-veto/placebo experiment.
