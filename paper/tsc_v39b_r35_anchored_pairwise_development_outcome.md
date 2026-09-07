# TSC v39b/r35 Anchored Pairwise Development Outcome

This document reports the frozen seven-city development gate. Network metrics are averaged within each city before city-macro selection.

- integrity: `PASS`
- development gate: `FAIL`
- combined cache SHA-256: `14c6dfad87999e55fa86fa942ac0ca2334129f4d1662b3c31a6e4fc7aa74c242`
- expanded baseline SHA-256: `f9223a7001078d9a6df0e1a2ad4bfe1d2bc59fc2547621f092e8be702596dbfa`
- source tree SHA-256: `fdc26c1a843ee5f82a340693030bd8a8f9f3082c35227f034440fc87a85e9829`

## Frozen gate

- LOCO anchor macro regret: `0.234172246`
- LOCO selected macro regret: `0.236404785`
- LOCO relative improvement: `-0.9534%`
- LOCO improved cities: `0/7`
- LOCO maximum city regression: `0.014168339`
- global candidate: `constant_alpha_0`
- global relative improvement: `0.0000%`
- global improved cities: `0/7`
- decision: `reject_v39_anchored_pairwise`

| Held-out city | LOCO candidate | Anchor regret | Selected regret | Improvement |
|---|---|---:|---:|---:|
| resco_synthetic | constant_alpha_0 | 0.303230 | 0.303230 | 0.000000 |
| cologne | constant_alpha_0 | 0.201061 | 0.201061 | 0.000000 |
| ingolstadt | constant_alpha_0 | 0.260662 | 0.260662 | 0.000000 |
| atlanta | constant_alpha_0 | 0.088993 | 0.088993 | 0.000000 |
| hangzhou | local_cap_1_z_0p25 | 0.352801 | 0.354260 | -0.001459 |
| new_york | local_cap_1_z_0p25 | 0.331792 | 0.331792 | 0.000000 |
| salt_lake_city | constant_alpha_0p9 | 0.100667 | 0.114835 | -0.014168 |

## Claim boundary

This is seven-city development on 14 capacity-admitted networks using 60 complete matched simulator action groups per target. Los Angeles and Nanchang remain unopened external holdouts unless this gate passes.
