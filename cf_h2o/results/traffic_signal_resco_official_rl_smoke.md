# Official RESCO RL Baseline Adapter

Protocol: calls the official RESCO `main.py` with RESCO's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| algorithm | scenario | runs | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | test_max_queues | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDQN | cologne1 | 1 | 27.5833 | 49.0759 | 11.9583 | 9.2621 | 4.1387 | 8.2889 |
| MPLight | cologne1 | 1 | 1126.8395 | 221.1241 | 143.3648 | 95.5368 | 27.0277 | 48.5700 |

## Runs

| ok | algorithm | scenario | seed | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | elapsed_sec | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | IDQN | cologne1 | None | 27.5833 | 49.0759 | 11.9583 | 9.2621 | 8.2889 | cf_h2o/results/resco_official_rl_smoke_adapter_valid/cologne1+idqn_testing@1_libsumo@True_gui@False_7a03d183e6a8cac0.json |
| True | MPLight | cologne1 | None | 1126.8395 | 221.1241 | 143.3648 | 95.5368 | 48.5700 | cf_h2o/results/resco_official_rl_smoke_adapter_valid/cologne1+mplight_testing@1_libsumo@True_gui@False_8439f12451bbf32c.json |

## Notes

- `testing` episodes are summarized after official RESCO switches the agent into testing mode.
- Official RESCO `timeLoss` adds SUMO `departDelay` inside its parser.
- Use larger episode budgets before citing these numbers as trained RL baselines.
