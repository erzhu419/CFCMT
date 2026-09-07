# Official RESCO RL Baseline Adapter

Protocol: calls the official RESCO `main.py` with RESCO's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| algorithm | scenario | runs | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | test_max_queues | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDQN | arterial4x4 | 1 | 1594.0870 | 490.7727 | 441.1291 | 19.8389 | 8.8299 | 3312.9989 |
| IDQN | cologne1 | 1 | 39.8515 | 60.0228 | 20.3359 | 15.2402 | 6.3836 | 199.4958 |
| IDQN | cologne3 | 1 | 30.0605 | 64.6053 | 13.2531 | 2.1540 | 1.2346 | 469.3995 |
| IDQN | cologne8 | 1 | 28.9295 | 93.5005 | 11.1554 | 1.1057 | 0.7106 | 1054.8147 |
| IDQN | grid4x4 | 1 | 75.7384 | 185.1734 | 51.3496 | 1.5779 | 0.8313 | 2055.4555 |
| IDQN | ingolstadt1 | 1 | 18.9953 | 38.1277 | 5.7383 | 1.5786 | 0.9359 | 157.5136 |
| IDQN | ingolstadt21 | 1 | 260.6003 | 358.2011 | 181.1352 | 3.6985 | 1.8987 | 3061.7333 |
| IDQN | ingolstadt7 | 1 | 39.1509 | 76.0159 | 12.4953 | 1.5929 | 0.9993 | 1057.2678 |
| MPLight | arterial4x4 | 1 | 1296.3204 | 504.1346 | 400.6747 | 19.0896 | 9.7499 | 2993.0279 |
| MPLight | cologne1 | 1 | 239.8674 | 145.9315 | 102.7664 | 56.1817 | 20.2092 | 1991.8017 |
| MPLight | cologne3 | 1 | 323.4930 | 246.1450 | 188.9992 | 34.4504 | 15.5035 | 2281.9767 |
| MPLight | grid4x4 | 1 | 42.2869 | 154.2215 | 14.6767 | 0.6879 | 0.4926 | 2290.5227 |
| MPLight | ingolstadt1 | 1 | 512.5157 | 91.2463 | 70.9314 | 2.5207 | 1.3359 | 1714.8796 |
| MPLight | ingolstadt7 | 1 | 261.4806 | 132.8456 | 74.7673 | 4.5061 | 2.1087 | 1589.5534 |

## Runs

| ok | algorithm | scenario | seed | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | elapsed_sec | cached | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | IDQN | cologne8 | 131 | 28.9295 | 93.5005 | 11.1554 | 1.1057 | 1054.8147 | True | cf_h2o/results/resco_official_rl_mid_budget/cologne8+idqn_testing@5_libsumo@True_gui@False_seed@131_6cab02c0966ef1c9.json |
| True | IDQN | arterial4x4 | 131 | 1594.0870 | 490.7727 | 441.1291 | 19.8389 | 3312.9989 | True | cf_h2o/results/resco_official_rl_mid_budget/routearterial4x4rouxml+idqn_testing@5_libsumo@True_gui@False_seed@131_9de60e99db328514.json |
| True | MPLight | arterial4x4 | 131 | 1296.3204 | 504.1346 | 400.6747 | 19.0896 | 2993.0279 | True | cf_h2o/results/resco_official_rl_mid_budget/routearterial4x4rouxml+mplight_testing@5_libsumo@True_gui@False_seed@131_5706022b4cb178fa.json |
| True | MPLight | cologne3 | 131 | 323.4930 | 246.1450 | 188.9992 | 34.4504 | 2281.9767 | True | cf_h2o/results/resco_official_rl_mid_budget/cologne3+mplight_testing@5_libsumo@True_gui@False_seed@131_e3e2a9e9dd3443a8.json |
| True | IDQN | cologne1 | 131 | 39.8515 | 60.0228 | 20.3359 | 15.2402 | 199.4958 | True | cf_h2o/results/resco_official_rl_mid_budget/cologne1+idqn_testing@5_libsumo@True_gui@False_seed@131_feb2d80d7f4350b7.json |
| True | MPLight | grid4x4 | 131 | 42.2869 | 154.2215 | 14.6767 | 0.6879 | 2290.5227 | True | cf_h2o/results/resco_official_rl_mid_budget/routegrid4x4rouxml+mplight_testing@5_libsumo@True_gui@False_seed@131_16ae6739fa254747.json |
| True | IDQN | grid4x4 | 131 | 75.7384 | 185.1734 | 51.3496 | 1.5779 | 2055.4555 | True | cf_h2o/results/resco_official_rl_mid_budget/routegrid4x4rouxml+idqn_testing@5_libsumo@True_gui@False_seed@131_c20f6b81d1b1a9bc.json |
| True | MPLight | cologne1 | 131 | 239.8674 | 145.9315 | 102.7664 | 56.1817 | 1991.8017 | True | cf_h2o/results/resco_official_rl_mid_budget/cologne1+mplight_testing@5_libsumo@True_gui@False_seed@131_32f90119bb3bde47.json |
| True | IDQN | ingolstadt7 | 131 | 39.1509 | 76.0159 | 12.4953 | 1.5929 | 1057.2678 | True | cf_h2o/results/resco_official_rl_mid_budget/ingolstadt7+idqn_testing@5_libsumo@True_gui@False_seed@131_231f9d8af806ab52.json |
| True | MPLight | ingolstadt7 | 131 | 261.4806 | 132.8456 | 74.7673 | 4.5061 | 1589.5534 | True | cf_h2o/results/resco_official_rl_mid_budget/ingolstadt7+mplight_testing@5_libsumo@True_gui@False_seed@131_a52e3d78ac77fd12.json |
| True | IDQN | cologne3 | 131 | 30.0605 | 64.6053 | 13.2531 | 2.1540 | 469.3995 | True | cf_h2o/results/resco_official_rl_mid_budget/cologne3+idqn_testing@5_libsumo@True_gui@False_seed@131_ee67b93ea3a4aeae.json |
| True | IDQN | ingolstadt21 | 131 | 260.6003 | 358.2011 | 181.1352 | 3.6985 | 3061.7333 | True | cf_h2o/results/resco_official_rl_mid_budget/ingolstadt21+idqn_testing@5_libsumo@True_gui@False_seed@131_37f8069560b064ee.json |
| True | IDQN | ingolstadt1 | 131 | 18.9953 | 38.1277 | 5.7383 | 1.5786 | 157.5136 | True | cf_h2o/results/resco_official_rl_mid_budget/ingolstadt1+idqn_testing@5_libsumo@True_gui@False_seed@131_23496144f5f7dcd1.json |
| True | MPLight | ingolstadt1 | 131 | 512.5157 | 91.2463 | 70.9314 | 2.5207 | 1714.8796 | True | cf_h2o/results/resco_official_rl_mid_budget/ingolstadt1+mplight_testing@5_libsumo@True_gui@False_seed@131_1f9367120eb7cc41.json |

## Notes

- `testing` episodes are summarized after official RESCO switches the agent into testing mode.
- Official RESCO `timeLoss` adds SUMO `departDelay` inside its parser.
- Use larger episode budgets before citing these numbers as trained RL baselines.
