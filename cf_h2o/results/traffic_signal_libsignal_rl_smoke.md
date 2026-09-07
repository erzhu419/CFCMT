# LibSignal SUMO Baseline Adapter

Protocol: calls LibSignal `run.py` with LibSignal's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| agent | network | runs | travel_time | reward | queue | delay | throughput | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqn | sumo1x1 | 1 | 22.8333 | -1.1750 | 1.3333 | 0.1378 | 6.0000 | 4.5885 |
| frap | sumo1x1 | 1 | 27.5000 | -0.3750 | 3.8333 | 0.2045 | 2.0000 | 4.5893 |
| mplight | sumo1x1 | 1 | 18.8571 | -0.0542 | 0.8333 | 0.1235 | 7.0000 | 4.5864 |
| presslight | sumo1x1 | 1 | 25.0000 | -6.1333 | 3.8333 | 0.2045 | 2.0000 | 4.5441 |

## Runs

| ok | agent | network | seed | travel_time | reward | queue | delay | throughput | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | presslight | sumo1x1 | None | 25.0000 | -6.1333 | 3.8333 | 0.2045 | 2.0000 | 4.5441 |
| True | mplight | sumo1x1 | None | 18.8571 | -0.0542 | 0.8333 | 0.1235 | 7.0000 | 4.5864 |
| True | dqn | sumo1x1 | None | 22.8333 | -1.1750 | 1.3333 | 0.1378 | 6.0000 | 4.5885 |
| True | frap | sumo1x1 | None | 27.5000 | -0.3750 | 3.8333 | 0.2045 | 2.0000 | 4.5893 |

## Notes

- The adapter parses the final metric line printed by LibSignal.
- Use larger episode/step budgets before citing RL numbers as trained baselines.
- The local LibSignal checkout has SUMO-only compatibility patches for optional CityFlow/CoLight imports.
