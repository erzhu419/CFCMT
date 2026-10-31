# TSC v33/r29 Source-Gated Pairwise Outcome

Generated: 2026-08-08T21:10:28.612913+00:00

## Integrity

**PASS**: 6 shards and 6 independent results passed SHA-256, complete-city holdout, zero-target-label, nested source-city OOF, causal-parent, and frozen-cache checks.

- snapshot SHA-256: `10d2a72deca550b80de9e9a09e9fa242116128dad3f29b5998091a5dd768aefd`
- v26 reference SHA-256: `05fc6e94757b74221b781878fa6bdae3b4142ee4d866aceea13332b5e4932154`
- cache aggregate SHA-256: `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`
- evaluated action groups: 2,626
- target folds with source-approved pairwise gate: 3/6

## Frozen Decision

Stage: **FAIL**
Selected family: `none`

| Family | Macro regret | Improvement | Cities | Max regression | Gate |
|---|---:|---:|---:|---:|---:|
| Rigid reference | 0.291137 | 0.00% | 0/6 | 0.000000 | reference |
| Source-gated pairwise | 0.290161 | +0.34% | 1/6 | 0.002315 | FAIL |

Salt Lake data were not read and cannot alter this decision.
