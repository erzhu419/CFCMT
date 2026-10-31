# Official RESCO RL Baseline Adapter

Protocol: calls the official RESCO `main.py` with RESCO's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| algorithm | scenario | runs | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | test_max_queues | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDQN | cologne1 | 1 | 1542.1472 | 290.7062 | 284.0174 | 113.7295 | 33.3370 | 12.5783 |
| MPLight | cologne1 | 1 | 940.6510 | 185.4119 | 165.6035 | 45.2219 | 15.7032 | 46.8097 |

## Runs

| ok | algorithm | scenario | seed | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | elapsed_sec | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | IDQN | cologne1 | None | 1542.1472 | 290.7062 | 284.0174 | 113.7295 | 12.5783 | cf_h2o/results/resco_official_rl_parallel_smoke/cologne1+idqn_testing@1_libsumo@True_gui@False_7a03d18339a33543.json |
| True | MPLight | cologne1 | None | 940.6510 | 185.4119 | 165.6035 | 45.2219 | 46.8097 | cf_h2o/results/resco_official_rl_parallel_smoke/cologne1+mplight_testing@1_libsumo@True_gui@False_8439f124eb94104c.json |

## Notes

- `testing` episodes are summarized after official RESCO switches the agent into testing mode.
- Official RESCO `timeLoss` adds SUMO `departDelay` inside its parser.
- Use larger episode budgets before citing these numbers as trained RL baselines.
