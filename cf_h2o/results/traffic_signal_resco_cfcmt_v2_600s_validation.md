# CFCMT V2 RESCO Leave-One-Network-Out Benchmark

All externally controlled policies use explicit minimum-green, yellow, and all-red execution.
The target contributes static topology summaries and online state only; no target transition labels are collected.

| Policy | Mean queue | P90 queue | Trip duration | Trip waiting | Throughput |
|---|---:|---:|---:|---:|---:|
| phase_pressure | 19.2977 | 27.8399 | 91.7564 | 13.6950 | 0.7944 |
| dense_guard | 20.5680 | 30.9893 | 95.5062 | 15.5840 | 0.7918 |
| max_pressure | 20.5686 | 30.9771 | 95.5062 | 15.5837 | 0.7920 |
| simulator_guard | 20.5686 | 30.9771 | 95.5062 | 15.5837 | 0.7920 |
| cfcmt_guard | 20.5690 | 30.9768 | 95.5009 | 15.5819 | 0.7917 |
| spillback_pressure | 22.1865 | 34.2711 | 105.0777 | 24.8562 | 0.7556 |
| cfcmt_mpc | 28.4394 | 46.5470 | 107.9613 | 29.5570 | 0.7133 |
| dense_mpc | 30.6364 | 45.0031 | 94.1349 | 15.8091 | 0.7490 |
| fixed_program | 41.5064 | 67.8550 | 138.7650 | 52.5127 | 0.7236 |
| simulator_mpc | 52.0006 | 78.0023 | 102.3589 | 34.2183 | 0.6123 |

## Hierarchical Bootstrap

| Baseline | Delta | 95% CI | Wins |
|---|---:|---:|---:|
| fixed_program | -20.9374 | [-33.7682, -12.2280] | 8/8 |
| max_pressure | 0.0004 | [0.0000, 0.0015] | 0/8 |
| phase_pressure | 1.2712 | [0.0583, 2.8493] | 2/8 |
| spillback_pressure | -1.6176 | [-6.8281, 1.4883] | 2/8 |
| simulator_mpc | -31.4316 | [-49.2118, -14.6009] | 7/8 |
| dense_mpc | -10.0675 | [-31.1862, 1.7491] | 4/8 |
| cfcmt_mpc | -7.8704 | [-21.2638, -0.0869] | 6/8 |
| simulator_guard | 0.0004 | [0.0000, 0.0015] | 0/8 |
| dense_guard | 0.0010 | [0.0000, 0.0038] | 0/8 |
