# LibSignal SUMO Baseline Adapter

Protocol: calls LibSignal `run.py` with LibSignal's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| agent | network | runs | travel_time | reward | queue | delay | throughput | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqn | sumo1x1 | 1 | 25.0000 | -4.5000 | 3.8333 | 0.2045 | 2.0000 | 3.6478 |
| fixedtime | sumo1x1 | 1 | 39.4000 | -7.4500 | 2.5000 | 1.5158 | 5.0000 | 2.5973 |

## Runs

| ok | agent | network | seed | travel_time | reward | queue | delay | throughput | elapsed_sec | cached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | fixedtime | sumo1x1 | None | 39.4000 | -7.4500 | 2.5000 | 1.5158 | 5.0000 | 2.5973 | True |
| True | dqn | sumo1x1 | None | 25.0000 | -4.5000 | 3.8333 | 0.2045 | 2.0000 | 3.6478 | True |

## Notes

- The adapter parses the final metric line printed by LibSignal.
- Use larger episode/step budgets before citing RL numbers as trained baselines.
- The local LibSignal checkout has SUMO-only compatibility patches for optional CityFlow/CoLight imports.
