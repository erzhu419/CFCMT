# TSC v26/r22 Rigid-Anchored Residual Outcome

Generated: 2026-08-08T19:02:15.951184+00:00

## Integrity

**PASS**: 6 shards and 30 independent results passed SHA-256, complete city holdout, zero-target-label, residual-estimand, pair-excluded-fit, and frozen-cache checks.

- snapshot SHA-256: `6d44e1f6d22eea229628a8c521be5b1bf10da75c35eea722ecdaaaddc60b8fa9`
- model source-tree SHA-256: `c8503170bb81b5b784a34fdb6ca4b41cad0a5b3947e316b16264cf43860f5790`
- cache aggregate SHA-256: `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`
- evaluated action groups: 2,626

## Frozen Decision

Stage: **FAIL**
Selected family: `none`
Diagnostic best family: `cfcmt_one_step_rigid_residual_mobility_only`

| Family | Macro regret | Improvement vs rigid | Cities improved | Max regression | Gate |
|---|---:|---:|---:|---:|---:|
| Rigid | 0.291137 | 0.00% | 0/6 | 0.000000 | reference |
| Rigid + queue residual | 0.284563 | +2.26% | 4/6 | 0.011347 | FAIL |
| Rigid + mobility residual | 0.280850 | +3.53% | 4/6 | 0.005595 | FAIL |
| Rigid + mobility and queue residuals | 0.281908 | +3.17% | 5/6 | 0.007332 | FAIL |
| Rigid + full five-mechanism residual | 0.290441 | +0.24% | 4/6 | 0.065532 | FAIL |

Salt Lake data were not read and cannot alter this decision.
