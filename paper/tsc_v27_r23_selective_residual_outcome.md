# TSC v27/r23 Selective Residual Outcome

Generated: 2026-08-08T19:38:24.518263+00:00

## Integrity

**PASS**: 6 shards and 36 independent results passed SHA-256, complete-city holdout, zero-target-label, source-only arbiter, pair-excluded-fit, and frozen-cache checks.

- snapshot SHA-256: `9f022b1f5314b84032b825cc79b6bdb996ae3b952774b0a16036e780d49910f0`
- v26 reference SHA-256: `05fc6e94757b74221b781878fa6bdae3b4142ee4d866aceea13332b5e4932154`
- cache aggregate SHA-256: `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`
- evaluated action groups: 2,626

## Frozen Decision

Stage: **FAIL**
Selected family: `none`
Diagnostic best family: `cfcmt_one_step_selective_residual_mobility_queue_q50`

| Family | q | Macro regret | Improvement | Cities | Max regression | Gate |
|---|---:|---:|---:|---:|---:|---:|
| Rigid | - | 0.291137 | 0.00% | 0/6 | 0.000000 | reference |
| Selective mobility residual, q=0.50 | 0.50 | 0.290319 | +0.28% | 1/6 | 0.004402 | FAIL |
| Selective mobility residual, q=0.75 | 0.75 | 0.289586 | +0.53% | 1/6 | 0.000396 | FAIL |
| Selective mobility residual, q=0.90 | 0.90 | 0.289409 | +0.59% | 2/6 | 0.000000 | FAIL |
| Selective mobility + queue residual, q=0.50 | 0.50 | 0.284085 | +2.42% | 4/6 | 0.003883 | FAIL |
| Selective mobility + queue residual, q=0.75 | 0.75 | 0.286706 | +1.52% | 2/6 | 0.001642 | FAIL |
| Selective mobility + queue residual, q=0.90 | 0.90 | 0.290939 | +0.07% | 1/6 | 0.000000 | FAIL |

Salt Lake data were not read and cannot alter this decision.
