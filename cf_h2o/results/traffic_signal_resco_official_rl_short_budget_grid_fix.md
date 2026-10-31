# Official RESCO RL Baseline Adapter

Protocol: calls the official RESCO `main.py` with RESCO's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| algorithm | scenario | runs | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | test_max_queues | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDQN | arterial4x4 | 1 | 1283.3901 | 491.7176 | 410.7446 | 19.2564 | 9.7540 | 559.9849 |
| IDQN | grid4x4 | 1 | 1326.1294 | 1372.1965 | 1311.1718 | 31.6107 | 11.0044 | 385.7567 |
| MPLight | arterial4x4 | 1 | 2152.9834 | 545.0747 | 529.2746 | 23.1763 | 11.2415 | 579.5639 |
| MPLight | grid4x4 | 1 | 45.2978 | 157.0771 | 17.1059 | 0.7754 | 0.5489 | 302.5416 |

## Runs

| ok | algorithm | scenario | seed | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | elapsed_sec | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | MPLight | grid4x4 | None | 45.2978 | 157.0771 | 17.1059 | 0.7754 | 302.5416 | cf_h2o/results/resco_official_rl_short_budget_grid_fix/routegrid4x4rouxml+mplight_testing@2_libsumo@True_gui@False_e59ccf1112625c97.json |
| True | IDQN | grid4x4 | None | 1326.1294 | 1372.1965 | 1311.1718 | 31.6107 | 385.7567 | cf_h2o/results/resco_official_rl_short_budget_grid_fix/routegrid4x4rouxml+idqn_testing@2_libsumo@True_gui@False_76f0fa4910418a58.json |
| True | IDQN | arterial4x4 | None | 1283.3901 | 491.7176 | 410.7446 | 19.2564 | 559.9849 | cf_h2o/results/resco_official_rl_short_budget_grid_fix/routearterial4x4rouxml+idqn_testing@2_libsumo@True_gui@False_90b0fd4f309c60de.json |
| True | MPLight | arterial4x4 | None | 2152.9834 | 545.0747 | 529.2746 | 23.1763 | 579.5639 | cf_h2o/results/resco_official_rl_short_budget_grid_fix/routearterial4x4rouxml+mplight_testing@2_libsumo@True_gui@False_2a3cb90876522ac8.json |

## Notes

- `testing` episodes are summarized after official RESCO switches the agent into testing mode.
- Official RESCO `timeLoss` adds SUMO `departDelay` inside its parser.
- Use larger episode budgets before citing these numbers as trained RL baselines.
