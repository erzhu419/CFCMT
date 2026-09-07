# LibSignal SUMO Baseline Adapter

Protocol: calls LibSignal `run.py` with LibSignal's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| agent | network | runs | travel_time | reward | queue | delay | throughput | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqn | sumo4x4 | 1 | 230.7391 | -48.7383 | 3.1042 | 0.1238 | 23.0000 | 39.5264 |
| frap | sumo4x4 | 1 | 196.1429 | -3.8563 | 2.9427 | 0.1276 | 28.0000 | 47.2911 |
| mplight | sumo4x4 | 1 | 123.6822 | -0.2071 | 0.1948 | 0.0164 | 107.0000 | 45.9902 |
| presslight | sumo4x4 | 1 | 169.4211 | -34.8750 | 3.2625 | 0.1351 | 19.0000 | 35.7800 |

## Runs

| ok | agent | network | seed | travel_time | reward | queue | delay | throughput | elapsed_sec | cached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | presslight | sumo4x4 | None | 169.4211 | -34.8750 | 3.2625 | 0.1351 | 19.0000 | 35.7800 | False |
| True | dqn | sumo4x4 | None | 230.7391 | -48.7383 | 3.1042 | 0.1238 | 23.0000 | 39.5264 | False |
| True | mplight | sumo4x4 | None | 123.6822 | -0.2071 | 0.1948 | 0.0164 | 107.0000 | 45.9902 | False |
| True | frap | sumo4x4 | None | 196.1429 | -3.8563 | 2.9427 | 0.1276 | 28.0000 | 47.2911 | False |

## Notes

- The adapter parses the final metric line printed by LibSignal.
- Use larger episode/step budgets before citing RL numbers as trained baselines.
- The local LibSignal checkout has SUMO-only compatibility patches for optional CityFlow/CoLight imports.
