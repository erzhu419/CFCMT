# TSC V128 Anchored Source-Null Residual Feasibility Result

## Decision

Reject. V128 does not justify a target-budget curve or untouched-city
confirmation.

## Frozen execution

- Scheduler task: `t88451`
- Node: `node003`
- Runtime: 1,334.66 s
- Peak RAM: 9,212 MB
- Snapshot: `e4c93d6ecc6687c4c839`
- Signature: `CFCMT/v128/anchored-source-null-reference-residual-b500-v1`
- Result:
  `cf_h2o/results/cluster/tsc_v128_anchored_source_null_residual_feasibility_20260901/development_v1/result.json`
- Result SHA-256:
  `f7fd449d2ac62338815ccb62ad2e0bab662fbdec40afcd9467d9548070cb4155`

Only the 658,880-byte result JSON was retrieved. The approximately 210 MB
upstream prediction artifact remains on the server.

## Main results

| Arm | Mean normalized delta | 95% interval | Enabled folds | Interventions |
|---|---:|---:|---:|---:|
| target-only | +0.0002073 | [-0.0000984, +0.0006630] | 4/22 | 40 |
| source-aligned | -0.0000322 | [-0.0000967, 0] | 1/22 | 6 |
| source-placebo | -0.0000269 | [-0.0000806, 0] | 1/22 | 10 |
| adaptive source-null | +0.0002073 | [-0.0000984, +0.0006630] | 4/22 | 40 |
| adaptive placebo-null | +0.0001804 | [-0.0001310, +0.0006428] | 5/22 | 50 |

The aligned source was selected over target-only in 0/22 folds. Adaptive
source-null minus target-only was exactly zero. Adaptive source-null minus the
adaptive placebo control was harmful by `+0.0000269` on average. The frozen
development gate therefore failed all source-contribution requirements.

## Diagnostic interpretation

V128 fits a separate target-labelled residual model for target-only,
source-aligned and source-placebo priors. With abundant target labels, each
residual can learn `actual - its_own_prior`; this can cancel the very prior
perturbation whose source value the experiment is intended to measure. The
near-identical adaptive result and zero source selection are therefore an
architectural diagnostic, not evidence that the earlier V98/V123 relative
source effects were fabricated.

The target-only safety gate also overgeneralized: three of its four enabled
held-out folds were harmful, despite passing the ten-seed calibration rule.
Consequently, loosening calibration or promoting the small unadjusted
source-aligned mean is not authorized.

## Authorized successor

V129 must fit one target-only residual per outer fold and freeze that residual
prediction across target-only, aligned-source and source-placebo priors. No
source-specific target residual refit is allowed. The same source-null,
placebo, pressure and held-out gates remain in force. This directly tests
whether a source prior adds value after a shared target correction rather than
allowing target supervision to erase each prior separately.
