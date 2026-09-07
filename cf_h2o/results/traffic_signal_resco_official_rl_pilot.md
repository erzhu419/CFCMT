# Official RESCO RL Baseline Adapter

Protocol: calls the official RESCO `main.py` with RESCO's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| algorithm | scenario | runs | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | test_max_queues | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDQN | cologne1 | 1 | 1330.3218 | 207.6625 | 196.2754 | 61.8044 | 15.7240 | 11.8371 |
| IDQN | cologne3 | 1 | 48.8530 | 83.3036 | 30.3277 | 5.7157 | 3.0435 | 19.3888 |
| IDQN | ingolstadt1 | 1 | 585.4419 | 94.5868 | 77.5478 | 2.5936 | 1.4979 | 8.5445 |
| MPLight | cologne1 | 1 | 1148.3123 | 226.1717 | 146.3127 | 96.9639 | 27.5215 | 45.4114 |
| MPLight | cologne3 | 1 | 374.5863 | 208.8813 | 162.5035 | 24.0791 | 14.6694 | 52.6598 |
| MPLight | ingolstadt1 | 1 | 549.4858 | 90.1952 | 72.5058 | 2.6075 | 1.5173 | 36.5132 |

## Runs

| ok | algorithm | scenario | seed | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | elapsed_sec | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | IDQN | cologne1 | None | 1330.3218 | 207.6625 | 196.2754 | 61.8044 | 11.8371 | cf_h2o/results/resco_official_rl_pilot/cologne1+idqn_testing@1_libsumo@True_gui@False_7a03d183a3fa695d.json |
| True | MPLight | cologne1 | None | 1148.3123 | 226.1717 | 146.3127 | 96.9639 | 45.4114 | cf_h2o/results/resco_official_rl_pilot/cologne1+mplight_testing@1_libsumo@True_gui@False_8439f1249b6f3177.json |
| True | IDQN | cologne3 | None | 48.8530 | 83.3036 | 30.3277 | 5.7157 | 19.3888 | cf_h2o/results/resco_official_rl_pilot/cologne3+idqn_testing@1_libsumo@True_gui@False_0a87607861241b7b.json |
| True | MPLight | cologne3 | None | 374.5863 | 208.8813 | 162.5035 | 24.0791 | 52.6598 | cf_h2o/results/resco_official_rl_pilot/cologne3+mplight_testing@1_libsumo@True_gui@False_93132af9b7a4a466.json |
| True | IDQN | ingolstadt1 | None | 585.4419 | 94.5868 | 77.5478 | 2.5936 | 8.5445 | cf_h2o/results/resco_official_rl_pilot/ingolstadt1+idqn_testing@1_libsumo@True_gui@False_8e5c1f6ab3a1028d.json |
| True | MPLight | ingolstadt1 | None | 549.4858 | 90.1952 | 72.5058 | 2.6075 | 36.5132 | cf_h2o/results/resco_official_rl_pilot/ingolstadt1+mplight_testing@1_libsumo@True_gui@False_000d5cf6321c9a39.json |

## Notes

- `testing` episodes are summarized after official RESCO switches the agent into testing mode.
- Official RESCO `timeLoss` adds SUMO `departDelay` inside its parser.
- Use larger episode budgets before citing these numbers as trained RL baselines.
