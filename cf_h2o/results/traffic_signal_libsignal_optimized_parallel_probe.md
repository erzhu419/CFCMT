# LibSignal SUMO Baseline Adapter

Protocol: calls LibSignal `run.py` with LibSignal's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| agent | network | runs | travel_time | reward | queue | delay | throughput | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqn | sumo1x1 | 1 | 17.1250 | -165.0925 | 110.5667 | 0.8805 | 8.0000 | 5.0139 |
| dqn | sumo1x3 | 1 | 118.4705 | -212.1891 | 28.1917 | 0.4861 | 593.0000 | 5.9328 |
| dqn | sumo4x4 | 1 | 357.7045 | -102.9867 | 6.5005 | 0.2120 | 88.0000 | 7.9238 |
| mplight | sumo1x1 | 1 | 109.4491 | -4.6942 | 38.0333 | 0.6729 | 579.0000 | 5.2328 |
| mplight | sumo1x3 | 1 | 96.4280 | -20.6388 | 41.1556 | 0.7107 | 493.0000 | 7.5251 |
| mplight | sumo4x4 | 1 | 142.8688 | -0.6006 | 0.5323 | 0.0356 | 282.0000 | 7.3311 |
| presslight | sumo1x1 | 1 | 88.4379 | -36.1583 | 31.0000 | 0.5560 | 507.0000 | 4.5066 |
| presslight | sumo1x3 | 1 | 157.3544 | -134.5658 | 43.8694 | 0.7518 | 316.0000 | 6.0997 |
| presslight | sumo4x4 | 1 | 383.4348 | -69.3025 | 6.5078 | 0.1931 | 92.0000 | 8.0418 |

## Runs

| ok | agent | network | seed | travel_time | reward | queue | delay | throughput | elapsed_sec | cached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | presslight | sumo1x1 | 131 | 88.4379 | -36.1583 | 31.0000 | 0.5560 | 507.0000 | 4.5066 | False |
| True | dqn | sumo1x1 | 131 | 17.1250 | -165.0925 | 110.5667 | 0.8805 | 8.0000 | 5.0139 | False |
| True | mplight | sumo1x1 | 131 | 109.4491 | -4.6942 | 38.0333 | 0.6729 | 579.0000 | 5.2328 | False |
| True | dqn | sumo1x3 | 131 | 118.4705 | -212.1891 | 28.1917 | 0.4861 | 593.0000 | 5.9328 | False |
| True | presslight | sumo1x3 | 131 | 157.3544 | -134.5658 | 43.8694 | 0.7518 | 316.0000 | 6.0997 | False |
| True | mplight | sumo1x3 | 131 | 96.4280 | -20.6388 | 41.1556 | 0.7107 | 493.0000 | 7.5251 | False |
| True | dqn | sumo4x4 | 131 | 357.7045 | -102.9867 | 6.5005 | 0.2120 | 88.0000 | 7.9238 | False |
| True | presslight | sumo4x4 | 131 | 383.4348 | -69.3025 | 6.5078 | 0.1931 | 92.0000 | 8.0418 | False |
| True | mplight | sumo4x4 | 131 | 142.8688 | -0.6006 | 0.5323 | 0.0356 | 282.0000 | 7.3311 | False |

## Notes

- The adapter parses the final metric line printed by LibSignal.
- Use larger episode/step budgets before citing RL numbers as trained baselines.
- The local LibSignal checkout has SUMO-only compatibility patches for optional CityFlow/CoLight imports.
