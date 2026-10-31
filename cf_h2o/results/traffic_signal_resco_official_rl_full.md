# Official RESCO RL Baseline Adapter

Protocol: calls the official RESCO `main.py` with RESCO's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| algorithm | scenario | runs | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | test_max_queues | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDQN | arterial4x4 | 3 | 1505.5358 | 501.3054 | 436.9201 | 20.0860 | 9.4094 | 17179.2053 |
| IDQN | cologne1 | 3 | 612.7367 | 164.9049 | 138.9499 | 48.6100 | 14.0242 | 812.4412 |
| IDQN | cologne3 | 3 | 86.6096 | 92.3130 | 43.2678 | 3.7133 | 1.5442 | 2351.5003 |
| IDQN | cologne8 | 3 | 24.7600 | 89.3794 | 8.3538 | 0.8162 | 0.5433 | 5946.6425 |
| IDQN | grid4x4 | 3 | 47.7521 | 158.2155 | 25.2053 | 0.8278 | 0.4205 | 11120.6226 |
| IDQN | ingolstadt1 | 3 | 18.0972 | 37.1910 | 5.4896 | 1.4845 | 0.9028 | 911.3322 |
| IDQN | ingolstadt21 | 3 | 256.5877 | 353.1114 | 176.4054 | 3.7606 | 1.9214 | 13141.2565 |
| IDQN | ingolstadt7 | 3 | 51.8692 | 78.8268 | 17.3567 | 1.9823 | 1.0498 | 5313.6457 |
| MPLight | arterial4x4 | 3 | 1056.7967 | 357.4536 | 238.7453 | 12.3698 | 5.9620 | 13576.0240 |
| MPLight | cologne1 | 3 | 189.3935 | 122.7513 | 68.8432 | 42.3779 | 12.9511 | 6985.7837 |
| MPLight | cologne3 | 3 | 1011.2390 | 543.5108 | 515.3289 | 44.8835 | 15.0655 | 10132.7531 |
| MPLight | grid4x4 | 3 | 44.3930 | 156.1122 | 16.5979 | 0.7476 | 0.5279 | 10392.9849 |
| MPLight | ingolstadt1 | 3 | 401.8947 | 85.9055 | 63.1920 | 2.1634 | 1.4672 | 7888.1431 |
| MPLight | ingolstadt7 | 3 | 284.3911 | 111.5600 | 57.4631 | 4.0143 | 2.0078 | 8755.1389 |

## Runs

| ok | algorithm | scenario | seed | test_timeLoss | test_duration | test_waitingTime | test_queue_lengths | elapsed_sec | cached | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | MPLight | grid4x4 | 337 | 43.6556 | 155.3435 | 16.8068 | 0.7456 | 10367.5524 | False | cf_h2o/results/resco_official_rl_full/routegrid4x4rouxml+mplight_testing@5_libsumo@True_gui@False_seed@337_a900275188dadd9c.json |
| True | MPLight | grid4x4 | 911 | 45.5357 | 157.2420 | 17.0426 | 0.7650 | 10396.9638 | False | cf_h2o/results/resco_official_rl_full/routegrid4x4rouxml+mplight_testing@5_libsumo@True_gui@False_seed@911_67363a57a973f5a8.json |
| True | MPLight | grid4x4 | 131 | 43.9876 | 155.7513 | 15.9443 | 0.7322 | 10414.4387 | False | cf_h2o/results/resco_official_rl_full/routegrid4x4rouxml+mplight_testing@5_libsumo@True_gui@False_seed@131_16ae673963def57c.json |
| True | IDQN | grid4x4 | 911 | 49.8594 | 160.1917 | 27.2759 | 0.8775 | 11107.0985 | False | cf_h2o/results/resco_official_rl_full/routegrid4x4rouxml+idqn_testing@5_libsumo@True_gui@False_seed@911_9337e6f245508ec8.json |
| True | IDQN | grid4x4 | 337 | 30.9823 | 142.5012 | 8.2504 | 0.4137 | 11122.6956 | False | cf_h2o/results/resco_official_rl_full/routegrid4x4rouxml+idqn_testing@5_libsumo@True_gui@False_seed@337_611b06c4a7004c35.json |
| True | IDQN | grid4x4 | 131 | 62.4147 | 171.9536 | 40.0896 | 1.1922 | 11132.0738 | False | cf_h2o/results/resco_official_rl_full/routegrid4x4rouxml+idqn_testing@5_libsumo@True_gui@False_seed@131_c20f6b8197cfe80c.json |
| True | IDQN | cologne1 | 131 | 32.4308 | 53.2782 | 15.1922 | 11.2092 | 752.8621 | False | cf_h2o/results/resco_official_rl_full/cologne1+idqn_testing@5_libsumo@True_gui@False_seed@131_feb2d80d25f976a8.json |
| True | IDQN | cologne1 | 337 | 1776.3271 | 390.2945 | 387.3347 | 124.8061 | 915.2101 | False | cf_h2o/results/resco_official_rl_full/cologne1+idqn_testing@5_libsumo@True_gui@False_seed@337_0360c22914615b7e.json |
| True | IDQN | cologne1 | 911 | 29.4522 | 51.1420 | 14.3230 | 9.8147 | 769.2513 | False | cf_h2o/results/resco_official_rl_full/cologne1+idqn_testing@5_libsumo@True_gui@False_seed@911_ef39d13842f1a9f4.json |
| True | IDQN | arterial4x4 | 337 | 1255.7336 | 469.9592 | 384.7571 | 18.2597 | 17052.4342 | False | cf_h2o/results/resco_official_rl_full/routearterial4x4rouxml+idqn_testing@5_libsumo@True_gui@False_seed@337_1d25b0aaac1c2229.json |
| True | IDQN | arterial4x4 | 131 | 1845.6157 | 538.3510 | 508.1782 | 22.3420 | 17417.3978 | False | cf_h2o/results/resco_official_rl_full/routearterial4x4rouxml+idqn_testing@5_libsumo@True_gui@False_seed@131_9de60e99cce8e8d5.json |
| True | MPLight | cologne1 | 131 | 297.8892 | 125.7937 | 77.9434 | 47.2852 | 6526.1139 | False | cf_h2o/results/resco_official_rl_full/cologne1+mplight_testing@5_libsumo@True_gui@False_seed@131_32f9011924d1c157.json |
| True | MPLight | cologne1 | 337 | 239.3680 | 189.7373 | 112.9036 | 68.4818 | 6659.1725 | False | cf_h2o/results/resco_official_rl_full/cologne1+mplight_testing@5_libsumo@True_gui@False_seed@337_0156ab817270d032.json |
| True | IDQN | cologne3 | 131 | 54.5119 | 84.0457 | 33.5567 | 3.5954 | 2172.0234 | False | cf_h2o/results/resco_official_rl_full/cologne3+idqn_testing@5_libsumo@True_gui@False_seed@131_ee67b93e06c2362d.json |
| True | IDQN | cologne3 | 337 | 178.7896 | 131.1169 | 85.3408 | 5.6580 | 2378.1328 | False | cf_h2o/results/resco_official_rl_full/cologne3+idqn_testing@5_libsumo@True_gui@False_seed@337_389d08875e74deae.json |
| True | IDQN | cologne3 | 911 | 26.5273 | 61.7765 | 10.9060 | 1.8865 | 2504.3447 | False | cf_h2o/results/resco_official_rl_full/cologne3+idqn_testing@5_libsumo@True_gui@False_seed@911_2963caf789cd3800.json |
| True | MPLight | arterial4x4 | 337 | 1067.1800 | 335.8660 | 210.3153 | 11.4879 | 13354.1987 | False | cf_h2o/results/resco_official_rl_full/routearterial4x4rouxml+mplight_testing@5_libsumo@True_gui@False_seed@337_8c115605eecd21bc.json |
| True | MPLight | arterial4x4 | 911 | 1102.8937 | 311.2465 | 215.7209 | 10.6787 | 13114.0033 | False | cf_h2o/results/resco_official_rl_full/routearterial4x4rouxml+mplight_testing@5_libsumo@True_gui@False_seed@911_13ab2bbcca2428ee.json |
| True | MPLight | arterial4x4 | 131 | 1000.3164 | 425.2482 | 290.1998 | 14.9427 | 14259.8700 | False | cf_h2o/results/resco_official_rl_full/routearterial4x4rouxml+mplight_testing@5_libsumo@True_gui@False_seed@131_5706022bd201e52a.json |
| True | MPLight | cologne1 | 911 | 30.9231 | 52.7228 | 15.6825 | 11.3667 | 7772.0648 | False | cf_h2o/results/resco_official_rl_full/cologne1+mplight_testing@5_libsumo@True_gui@False_seed@911_93543d98422139d9.json |
| True | IDQN | ingolstadt1 | 131 | 18.7677 | 37.6781 | 5.8684 | 1.4660 | 883.2351 | False | cf_h2o/results/resco_official_rl_full/ingolstadt1+idqn_testing@5_libsumo@True_gui@False_seed@131_23496144a33ab7db.json |
| True | IDQN | ingolstadt1 | 337 | 17.7639 | 36.9625 | 5.2281 | 1.5104 | 1090.5052 | False | cf_h2o/results/resco_official_rl_full/ingolstadt1+idqn_testing@5_libsumo@True_gui@False_seed@337_7e52669262f5910b.json |
| True | IDQN | arterial4x4 | 911 | 1415.2581 | 495.6060 | 417.8251 | 19.6562 | 17067.7839 | False | cf_h2o/results/resco_official_rl_full/routearterial4x4rouxml+idqn_testing@5_libsumo@True_gui@False_seed@911_e84712e715ec3164.json |
| True | IDQN | ingolstadt1 | 911 | 17.7600 | 36.9324 | 5.3724 | 1.4771 | 760.2563 | False | cf_h2o/results/resco_official_rl_full/ingolstadt1+idqn_testing@5_libsumo@True_gui@False_seed@911_c3f395e7a7c221f5.json |
| True | MPLight | cologne3 | 131 | 909.1939 | 498.5559 | 470.7513 | 42.4420 | 9859.2217 | False | cf_h2o/results/resco_official_rl_full/cologne3+mplight_testing@5_libsumo@True_gui@False_seed@131_e3e2a9e90248db4c.json |
| True | IDQN | cologne8 | 131 | 24.4332 | 89.0807 | 8.0859 | 0.8024 | 6054.6408 | False | cf_h2o/results/resco_official_rl_full/cologne8+idqn_testing@5_libsumo@True_gui@False_seed@131_6cab02c0d5a05ba1.json |
| True | IDQN | cologne8 | 337 | 25.0811 | 89.7025 | 8.6819 | 0.8466 | 5925.9536 | False | cf_h2o/results/resco_official_rl_full/cologne8+idqn_testing@5_libsumo@True_gui@False_seed@337_d5997aa095152c89.json |
| True | IDQN | cologne8 | 911 | 24.7658 | 89.3548 | 8.2935 | 0.7994 | 5859.3332 | False | cf_h2o/results/resco_official_rl_full/cologne8+idqn_testing@5_libsumo@True_gui@False_seed@911_fb3133fc0c98ef40.json |
| True | MPLight | cologne3 | 337 | 918.7961 | 523.7286 | 493.6354 | 40.3289 | 10024.7310 | False | cf_h2o/results/resco_official_rl_full/cologne3+mplight_testing@5_libsumo@True_gui@False_seed@337_bd152ce84ac6a276.json |
| True | MPLight | cologne3 | 911 | 1205.7269 | 608.2478 | 581.6000 | 51.8797 | 10514.3067 | False | cf_h2o/results/resco_official_rl_full/cologne3+mplight_testing@5_libsumo@True_gui@False_seed@911_7e43320a1f58601a.json |
| True | IDQN | ingolstadt7 | 131 | 32.9041 | 73.1439 | 11.4758 | 1.4346 | 5325.9139 | False | cf_h2o/results/resco_official_rl_full/ingolstadt7+idqn_testing@5_libsumo@True_gui@False_seed@131_231f9d8a8e6245b9.json |
| True | MPLight | ingolstadt1 | 131 | 442.5897 | 86.9326 | 65.7132 | 1.7559 | 7994.9034 | False | cf_h2o/results/resco_official_rl_full/ingolstadt1+mplight_testing@5_libsumo@True_gui@False_seed@131_1f93671236418df6.json |
| True | IDQN | ingolstadt7 | 337 | 76.8428 | 88.5877 | 27.7159 | 2.8204 | 5308.7406 | False | cf_h2o/results/resco_official_rl_full/ingolstadt7+idqn_testing@5_libsumo@True_gui@False_seed@337_3ae362298a74c1a9.json |
| True | MPLight | ingolstadt1 | 337 | 570.0591 | 93.4922 | 75.6378 | 2.4655 | 7982.5647 | False | cf_h2o/results/resco_official_rl_full/ingolstadt1+mplight_testing@5_libsumo@True_gui@False_seed@337_5b791e4cbe5e9b73.json |
| True | IDQN | ingolstadt7 | 911 | 45.8607 | 74.7487 | 12.8784 | 1.6919 | 5306.2825 | False | cf_h2o/results/resco_official_rl_full/ingolstadt7+idqn_testing@5_libsumo@True_gui@False_seed@911_1b60afc64cfc082c.json |
| True | MPLight | ingolstadt1 | 911 | 193.0354 | 77.2917 | 48.2249 | 2.2688 | 7686.9611 | False | cf_h2o/results/resco_official_rl_full/ingolstadt1+mplight_testing@5_libsumo@True_gui@False_seed@911_6d988afe0707033b.json |
| True | MPLight | ingolstadt7 | 131 | 201.2471 | 79.2247 | 23.3686 | 2.7992 | 9544.1374 | False | cf_h2o/results/resco_official_rl_full/ingolstadt7+mplight_testing@5_libsumo@True_gui@False_seed@131_a52e3d782804d753.json |
| True | MPLight | ingolstadt7 | 337 | 250.8196 | 123.6744 | 67.9518 | 4.9580 | 9049.0760 | False | cf_h2o/results/resco_official_rl_full/ingolstadt7+mplight_testing@5_libsumo@True_gui@False_seed@337_48899fc79b8d96af.json |
| True | MPLight | ingolstadt7 | 911 | 401.1066 | 131.7808 | 81.0689 | 4.2856 | 7672.2034 | False | cf_h2o/results/resco_official_rl_full/ingolstadt7+mplight_testing@5_libsumo@True_gui@False_seed@911_6f174b54bd5b1441.json |
| True | IDQN | ingolstadt21 | 131 | 200.9675 | 326.9271 | 144.3710 | 3.3259 | 13113.9732 | False | cf_h2o/results/resco_official_rl_full/ingolstadt21+idqn_testing@5_libsumo@True_gui@False_seed@131_37f806955558c7fa.json |
| True | IDQN | ingolstadt21 | 911 | 241.1639 | 330.5350 | 158.1476 | 3.7693 | 13103.7728 | False | cf_h2o/results/resco_official_rl_full/ingolstadt21+idqn_testing@5_libsumo@True_gui@False_seed@911_6696ef681ed0c549.json |
| True | IDQN | ingolstadt21 | 337 | 327.6318 | 401.8722 | 226.6975 | 4.1864 | 13206.0235 | False | cf_h2o/results/resco_official_rl_full/ingolstadt21+idqn_testing@5_libsumo@True_gui@False_seed@337_be3efc58b12b6bbf.json |

## Notes

- `testing` episodes are summarized after official RESCO switches the agent into testing mode.
- Official RESCO `timeLoss` adds SUMO `departDelay` inside its parser.
- Use larger episode budgets before citing these numbers as trained RL baselines.
