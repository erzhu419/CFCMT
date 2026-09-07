# Official RESCO RL Baseline Adapter

Protocol: calls the official RESCO `main.py` with RESCO's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| algorithm | scenario | runs | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | test_max_queues | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDQN | cologne1 | 1 | 30.5648 | 52.0978 | 14.9087 | 10.0943 | 4.5409 | 11.4557 |

## Runs

| ok | algorithm | scenario | seed | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | elapsed_sec | cached | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | IDQN | cologne1 | None | 30.5648 | 52.0978 | 14.9087 | 10.0943 | 11.4557 | True | cf_h2o/results/resco_cache_probe_logs/cologne1+idqn_testing@1_libsumo@True_gui@False_7a03d183d45a2d5a.json |

## Notes

- `testing` episodes are summarized after official RESCO switches the agent into testing mode.
- Official RESCO `timeLoss` adds SUMO `departDelay` inside its parser.
- Use larger episode budgets before citing these numbers as trained RL baselines.
