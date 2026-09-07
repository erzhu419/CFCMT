# LibSignal SUMO Baseline Adapter

Protocol: calls LibSignal `run.py` with LibSignal's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| agent | network | runs | travel_time | reward | queue | delay | throughput | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqn | sumo1x1 | 1 | 60.1148 | -70.9250 | 48.1000 | 0.6644 | 122.0000 | 8.8893 |
| dqn | sumo1x3 | 1 | 67.3892 | -81.5233 | 15.5556 | 0.4685 | 334.0000 | 14.3430 |
| dqn | sumo4x4 | 1 | 230.7391 | -48.7383 | 3.1042 | 0.1238 | 23.0000 | 39.5264 |
| frap | sumo1x1 | 1 | 76.6529 | -3.1610 | 25.7000 | 0.5671 | 363.0000 | 16.7853 |
| frap | sumo1x3 | 1 | 82.2418 | -13.7470 | 20.7389 | 0.4290 | 244.0000 | 25.1379 |
| frap | sumo4x4 | 1 | 196.1429 | -3.8563 | 2.9427 | 0.1276 | 28.0000 | 47.2911 |
| mplight | sumo1x1 | 1 | 62.7521 | -5.9835 | 48.7000 | 0.6721 | 117.0000 | 26.1301 |
| mplight | sumo1x3 | 1 | 91.5825 | -1.1533 | 2.9278 | 0.3269 | 491.0000 | 31.3268 |
| mplight | sumo4x4 | 1 | 123.6822 | -0.2071 | 0.1948 | 0.0164 | 107.0000 | 45.9902 |
| presslight | sumo1x1 | 1 | 25.3333 | -89.0617 | 87.8333 | 0.7886 | 9.0000 | 9.5788 |
| presslight | sumo1x3 | 1 | 86.1058 | -93.1433 | 30.2056 | 0.7149 | 189.0000 | 13.7208 |
| presslight | sumo4x4 | 1 | 169.4211 | -34.8750 | 3.2625 | 0.1351 | 19.0000 | 35.7800 |

## Runs

| ok | agent | network | seed | travel_time | reward | queue | delay | throughput | elapsed_sec | cached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | dqn | sumo1x3 | None | 67.3892 | -81.5233 | 15.5556 | 0.4685 | 334.0000 | 14.3430 | True |
| True | dqn | sumo1x1 | None | 60.1148 | -70.9250 | 48.1000 | 0.6644 | 122.0000 | 8.8893 | True |
| True | frap | sumo1x1 | None | 76.6529 | -3.1610 | 25.7000 | 0.5671 | 363.0000 | 16.7853 | True |
| True | mplight | sumo1x1 | None | 62.7521 | -5.9835 | 48.7000 | 0.6721 | 117.0000 | 26.1301 | True |
| True | presslight | sumo1x3 | None | 86.1058 | -93.1433 | 30.2056 | 0.7149 | 189.0000 | 13.7208 | True |
| True | presslight | sumo1x1 | None | 25.3333 | -89.0617 | 87.8333 | 0.7886 | 9.0000 | 9.5788 | True |
| True | dqn | sumo4x4 | None | 230.7391 | -48.7383 | 3.1042 | 0.1238 | 23.0000 | 39.5264 | True |
| True | mplight | sumo1x3 | None | 91.5825 | -1.1533 | 2.9278 | 0.3269 | 491.0000 | 31.3268 | True |
| True | presslight | sumo4x4 | None | 169.4211 | -34.8750 | 3.2625 | 0.1351 | 19.0000 | 35.7800 | True |
| True | mplight | sumo4x4 | None | 123.6822 | -0.2071 | 0.1948 | 0.0164 | 107.0000 | 45.9902 | True |
| True | frap | sumo1x3 | None | 82.2418 | -13.7470 | 20.7389 | 0.4290 | 244.0000 | 25.1379 | True |
| True | frap | sumo4x4 | None | 196.1429 | -3.8563 | 2.9427 | 0.1276 | 28.0000 | 47.2911 | True |

## Notes

- The adapter parses the final metric line printed by LibSignal.
- Use larger episode/step budgets before citing RL numbers as trained baselines.
- The local LibSignal checkout has SUMO-only compatibility patches for optional CityFlow/CoLight imports.
