# LibSignal SUMO Baseline Adapter

Protocol: calls LibSignal `run.py` with LibSignal's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| agent | network | runs | travel_time | reward | queue | delay | throughput | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqn | sumo1x1 | 1 | 32.4000 | -129.6125 | 87.4000 | 0.7476 | 10.0000 | 8.3711 |
| frap | sumo1x1 | 1 | 44.9664 | -0.6623 | 5.9333 | 0.3809 | 387.0000 | 8.4214 |
| mplight | sumo1x1 | 1 | 141.1295 | -5.7202 | 46.4000 | 0.6604 | 193.0000 | 10.8604 |
| presslight | sumo1x1 | 1 | 29.5631 | -59.8033 | 56.4833 | 0.4791 | 206.0000 | 7.3831 |

## Runs

| ok | agent | network | seed | travel_time | reward | queue | delay | throughput | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | presslight | sumo1x1 | None | 29.5631 | -59.8033 | 56.4833 | 0.4791 | 206.0000 | 7.3831 |
| True | dqn | sumo1x1 | None | 32.4000 | -129.6125 | 87.4000 | 0.7476 | 10.0000 | 8.3711 |
| True | frap | sumo1x1 | None | 44.9664 | -0.6623 | 5.9333 | 0.3809 | 387.0000 | 8.4214 |
| True | mplight | sumo1x1 | None | 141.1295 | -5.7202 | 46.4000 | 0.6604 | 193.0000 | 10.8604 |

## Notes

- The adapter parses the final metric line printed by LibSignal.
- Use larger episode/step budgets before citing RL numbers as trained baselines.
- The local LibSignal checkout has SUMO-only compatibility patches for optional CityFlow/CoLight imports.
