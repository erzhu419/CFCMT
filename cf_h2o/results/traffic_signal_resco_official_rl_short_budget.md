# Official RESCO RL Baseline Adapter

Protocol: calls the official RESCO `main.py` with RESCO's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| algorithm | scenario | runs | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | test_max_queues | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDQN | cologne1 | 1 | 27.9349 | 49.6881 | 12.3521 | 9.5915 | 4.2712 | 52.5354 |
| IDQN | cologne3 | 1 | 88.5373 | 110.6854 | 61.9982 | 5.7180 | 2.0136 | 132.9737 |
| IDQN | cologne8 | 1 | 284.1709 | 277.4619 | 200.5296 | 10.2194 | 3.2731 | 266.4326 |
| IDQN | ingolstadt1 | 1 | 18.3405 | 37.8051 | 5.8260 | 2.0687 | 1.2441 | 45.2116 |
| IDQN | ingolstadt21 | 1 | 478.5396 | 551.3159 | 393.0022 | 7.9044 | 3.5228 | 662.3716 |
| IDQN | ingolstadt7 | 1 | 217.9859 | 76.7407 | 21.8844 | 2.3426 | 1.1971 | 258.8174 |
| MPLight | cologne1 | 1 | 566.8647 | 203.1496 | 120.1025 | 79.7295 | 18.8870 | 450.4112 |
| MPLight | cologne3 | 1 | 776.0452 | 415.8230 | 381.3279 | 46.0552 | 17.6012 | 504.0865 |
| MPLight | ingolstadt1 | 1 | 552.0249 | 115.9478 | 97.6693 | 13.9133 | 4.7316 | 410.6902 |
| MPLight | ingolstadt7 | 1 | 226.8502 | 124.2702 | 69.1262 | 3.0829 | 1.8383 | 475.6478 |

## Runs

| ok | algorithm | scenario | seed | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | elapsed_sec | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| False | MPLight | grid4x4 | None | nan | nan | nan | nan | 0.5650 | None |
| False | IDQN | grid4x4 | None | nan | nan | nan | nan | 0.5690 | None |
| False | IDQN | arterial4x4 | None | nan | nan | nan | nan | 0.5720 | None |
| False | MPLight | arterial4x4 | None | nan | nan | nan | nan | 0.5729 | None |
| False | MPLight | cologne8 | None | nan | nan | nan | nan | 6.3762 | None |
| True | IDQN | ingolstadt1 | None | 18.3405 | 37.8051 | 5.8260 | 2.0687 | 45.2116 | cf_h2o/results/resco_official_rl_short_budget/ingolstadt1+idqn_testing@2_libsumo@True_gui@False_4798b5aa1289080f.json |
| True | IDQN | cologne1 | None | 27.9349 | 49.6881 | 12.3521 | 9.5915 | 52.5354 | cf_h2o/results/resco_official_rl_short_budget/cologne1+idqn_testing@2_libsumo@True_gui@False_838e2026ebf1661d.json |
| True | IDQN | cologne3 | None | 88.5373 | 110.6854 | 61.9982 | 5.7180 | 132.9737 | cf_h2o/results/resco_official_rl_short_budget/cologne3+idqn_testing@2_libsumo@True_gui@False_60e1b38bc5df79f5.json |
| False | MPLight | ingolstadt21 | None | nan | nan | nan | nan | 4.3784 | None |
| True | IDQN | ingolstadt7 | None | 217.9859 | 76.7407 | 21.8844 | 2.3426 | 258.8174 | cf_h2o/results/resco_official_rl_short_budget/ingolstadt7+idqn_testing@2_libsumo@True_gui@False_286eb4ccd74ca8b5.json |
| True | IDQN | cologne8 | None | 284.1709 | 277.4619 | 200.5296 | 10.2194 | 266.4326 | cf_h2o/results/resco_official_rl_short_budget/cologne8+idqn_testing@2_libsumo@True_gui@False_b9fe1ce93220d489.json |
| True | MPLight | ingolstadt1 | None | 552.0249 | 115.9478 | 97.6693 | 13.9133 | 410.6902 | cf_h2o/results/resco_official_rl_short_budget/ingolstadt1+mplight_testing@2_libsumo@True_gui@False_a6a18c2363c4b824.json |
| True | MPLight | cologne1 | None | 566.8647 | 203.1496 | 120.1025 | 79.7295 | 450.4112 | cf_h2o/results/resco_official_rl_short_budget/cologne1+mplight_testing@2_libsumo@True_gui@False_7499313609230790.json |
| True | MPLight | cologne3 | None | 776.0452 | 415.8230 | 381.3279 | 46.0552 | 504.0865 | cf_h2o/results/resco_official_rl_short_budget/cologne3+mplight_testing@2_libsumo@True_gui@False_934a85ae686295fd.json |
| True | MPLight | ingolstadt7 | None | 226.8502 | 124.2702 | 69.1262 | 3.0829 | 475.6478 | cf_h2o/results/resco_official_rl_short_budget/ingolstadt7+mplight_testing@2_libsumo@True_gui@False_c8265a019eb92c99.json |
| True | IDQN | ingolstadt21 | None | 478.5396 | 551.3159 | 393.0022 | 7.9044 | 662.3716 | cf_h2o/results/resco_official_rl_short_budget/ingolstadt21+idqn_testing@2_libsumo@True_gui@False_08a12f1aa6270b42.json |

## Notes

- `testing` episodes are summarized after official RESCO switches the agent into testing mode.
- Official RESCO `timeLoss` adds SUMO `departDelay` inside its parser.
- Use larger episode budgets before citing these numbers as trained RL baselines.
