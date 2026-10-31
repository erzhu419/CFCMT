# TSC v28/r24 Policy-Conditioned Rollout-Value Outcome

Generated: 2026-08-08T19:59:40.680619+00:00

## Integrity

**PASS**: 6 shards and 6 independent results passed SHA-256, complete-city holdout, zero-target-label, terminal-value-only, pair-excluded-fit, and frozen-cache checks.

- snapshot SHA-256: `0915b12927610d8978a05851a34d2e787512bc296c96d610f6c47fdd3710f86a`
- v26 reference SHA-256: `05fc6e94757b74221b781878fa6bdae3b4142ee4d866aceea13332b5e4932154`
- cache aggregate SHA-256: `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`
- evaluated action groups: 2,626

## Frozen Decision

Stage: **FAIL**
Selected family: `none`

| Family | Macro regret | Improvement | Cities | Max regression | Gate |
|---|---:|---:|---:|---:|---:|
| Rigid | 0.291137 | 0.00% | 0/6 | 0.000000 | reference |
| Rollout-value residual | 0.291623 | -0.17% | 0/6 | 0.001899 | FAIL |

Salt Lake data were not read and cannot alter this decision.
