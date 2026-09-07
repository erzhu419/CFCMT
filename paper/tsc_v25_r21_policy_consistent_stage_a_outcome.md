# TSC v25/r21 Policy-Consistent Stage-A Outcome

Generated: 2026-08-08T18:25:04.322773+00:00

## Integrity

**PASS**: 6 shards and 66 independent results passed SHA-256, complete city-group holdout, zero-target-label, finite-metric, and frozen-cache checks.

- snapshot SHA-256: `8f60f48e0aac184941bf76bec91e04836ea73ea263b389ca1eaa42366a2d70be`
- source tree SHA-256: `997f072412977fb40420c3ecceeba48016b098142e71ded1bb32d265df625ffa`
- cache aggregate SHA-256: `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`
- evaluated action groups: 2,626

## Frozen Decision

Stage A: **FAIL**
Selected family: `none`
Diagnostic best family: `cfcmt_one_step_physical_mobility_only`

| Family | Macro regret | Improvement vs rigid | Cities improved | Max regression | Gate |
|---|---:|---:|---:|---:|---:|
| Rigid | 0.291137 | 0.00% | 0/6 | 0.000000 | reference |
| Queue only | 0.312231 | -7.25% | 5/6 | 0.262966 | FAIL |
| Red accumulation only | 0.312140 | -7.21% | 3/6 | 0.297198 | FAIL |
| Spillback only | 0.291137 | +0.00% | 0/6 | 0.000000 | FAIL |
| Served movement only | 0.340321 | -16.89% | 2/6 | 0.303692 | FAIL |
| Mobility only | 0.283836 | +2.51% | 3/6 | 0.083428 | FAIL |
| Mobility + queue | 0.294278 | -1.08% | 5/6 | 0.196266 | FAIL |
| Mobility + red | 0.313672 | -7.74% | 3/6 | 0.296862 | FAIL |
| Mobility + spillback | 0.291576 | -0.15% | 4/6 | 0.130661 | FAIL |
| Mobility + served | 0.336016 | -15.42% | 2/6 | 0.304792 | FAIL |
| Full five mechanisms | 0.294744 | -1.24% | 4/6 | 0.073890 | FAIL |

The selector is mechanical. Salt Lake data were not read and cannot alter this decision.
