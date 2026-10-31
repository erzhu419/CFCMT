# LibSignal SUMO Baseline Adapter

Protocol: calls LibSignal `run.py` with LibSignal's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| agent | network | runs | travel_time | reward | queue | delay | throughput | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqn | sumo1x1 | 1 | 19.2000 | -1.5750 | 1.8333 | 0.1570 | 5.0000 | 2.9093 |

## Runs

| ok | agent | network | seed | travel_time | reward | queue | delay | throughput | elapsed_sec | cached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | dqn | sumo1x1 | 131 | 19.2000 | -1.5750 | 1.8333 | 0.1570 | 5.0000 | 2.9093 | False |

## Notes

- The adapter parses the final metric line printed by LibSignal.
- Use larger episode/step budgets before citing RL numbers as trained baselines.
- The local LibSignal checkout has SUMO-only compatibility patches for optional CityFlow/CoLight imports.
