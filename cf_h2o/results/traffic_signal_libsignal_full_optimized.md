# LibSignal SUMO Baseline Adapter

Protocol: calls LibSignal `run.py` with LibSignal's own state, reward, action, and logging definitions.

These rows are leaderboard-style RL cross-checks, not same-protocol CFCMT phase-MPC rows.

## Aggregate

| agent | network | runs | travel_time | reward | queue | delay | throughput | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqn | sumo1x1 | 3 | 38.9448 | -3.9320 | 3.2398 | 0.2831 | 1998.6667 | 175.9667 |
| dqn | sumo1x3 | 3 | 97.0904 | -28.8363 | 6.5312 | 0.2443 | 2487.3333 | 425.3838 |
| dqn | sumo4x4 | 3 | 140.6097 | -8.2498 | 0.6027 | 0.0409 | 1465.0000 | 2140.7727 |
| frap | sumo1x1 | 3 | 41.9234 | -2.9908 | 24.4611 | 0.4596 | 1664.6667 | 439.6200 |
| frap | sumo1x3 | 3 | 118.7116 | -5.5377 | 11.6722 | 0.3977 | 2505.3333 | 1583.0533 |
| frap | sumo4x4 | 3 | 140.4773 | -0.6911 | 0.6123 | 0.0422 | 1464.0000 | 7188.1846 |
| mplight | sumo1x1 | 3 | 40.5462 | -0.4764 | 4.3731 | 0.3190 | 1995.0000 | 371.5690 |
| mplight | sumo1x3 | 3 | 66.5017 | -0.7088 | 1.8759 | 0.2431 | 2814.3333 | 647.6640 |
| mplight | sumo4x4 | 3 | 140.4690 | -0.6799 | 0.6074 | 0.0423 | 1464.3333 | 1017.2967 |
| presslight | sumo1x1 | 3 | 39.7481 | -9.5055 | 3.4417 | 0.2850 | 1998.6667 | 175.8352 |
| presslight | sumo1x3 | 3 | 63.1881 | -10.1500 | 1.1636 | 0.1882 | 2816.0000 | 393.4139 |
| presslight | sumo4x4 | 3 | 144.2793 | -8.1274 | 0.7107 | 0.0481 | 1463.3333 | 2080.0427 |

## Runs

| ok | agent | network | seed | travel_time | reward | queue | delay | throughput | elapsed_sec | cached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | presslight | sumo1x1 | 911 | 40.0300 | -9.5575 | 3.5056 | 0.2888 | 1999.0000 | 170.9863 | False |
| True | dqn | sumo1x1 | 131 | 38.5958 | -3.7538 | 3.1222 | 0.2780 | 1999.0000 | 174.9529 | False |
| True | dqn | sumo1x1 | 911 | 38.9029 | -3.6954 | 3.1528 | 0.2840 | 1998.0000 | 175.2312 | False |
| True | presslight | sumo1x1 | 131 | 38.4887 | -9.0997 | 2.9750 | 0.2765 | 1999.0000 | 175.4214 | False |
| True | dqn | sumo1x1 | 337 | 39.3357 | -4.3467 | 3.4444 | 0.2874 | 1999.0000 | 177.7162 | False |
| True | presslight | sumo1x1 | 337 | 40.7257 | -9.8594 | 3.8444 | 0.2897 | 1998.0000 | 181.0981 | False |
| True | frap | sumo1x1 | 131 | 44.3453 | -7.9911 | 64.3000 | 0.7043 | 1005.0000 | 445.0154 | False |
| True | frap | sumo1x1 | 337 | 40.4870 | -0.4749 | 4.4222 | 0.3334 | 1994.0000 | 447.6595 | False |
| True | mplight | sumo1x1 | 337 | 40.4652 | -0.5002 | 4.5722 | 0.3177 | 1995.0000 | 363.2915 | False |
| True | mplight | sumo1x1 | 131 | 39.6752 | -0.4183 | 3.9528 | 0.3118 | 1995.0000 | 370.9340 | False |
| True | mplight | sumo1x1 | 911 | 41.4982 | -0.5106 | 4.5944 | 0.3276 | 1995.0000 | 380.4817 | False |
| True | frap | sumo1x1 | 911 | 40.9378 | -0.5063 | 4.6611 | 0.3410 | 1995.0000 | 426.1852 | False |
| True | dqn | sumo1x3 | 131 | 65.2353 | -4.7481 | 1.2019 | 0.1925 | 2818.0000 | 426.0426 | False |
| True | dqn | sumo1x3 | 337 | 163.5360 | -78.0605 | 17.3389 | 0.3622 | 1821.0000 | 442.5519 | False |
| True | presslight | sumo1x3 | 131 | 63.8224 | -10.2544 | 1.1991 | 0.1917 | 2809.0000 | 396.3421 | False |
| True | dqn | sumo1x3 | 911 | 62.4998 | -3.7004 | 1.0528 | 0.1783 | 2823.0000 | 407.5567 | False |
| True | presslight | sumo1x3 | 337 | 59.3819 | -9.8903 | 1.0963 | 0.1811 | 2823.0000 | 391.8020 | False |
| True | presslight | sumo1x3 | 911 | 66.3601 | -10.3053 | 1.1954 | 0.1917 | 2816.0000 | 392.0975 | False |
| True | mplight | sumo1x3 | 131 | 66.6385 | -0.6862 | 1.8324 | 0.2437 | 2816.0000 | 569.3371 | False |
| True | mplight | sumo1x3 | 337 | 66.5119 | -0.6992 | 1.8639 | 0.2436 | 2813.0000 | 690.1481 | False |
| True | mplight | sumo1x3 | 911 | 66.3547 | -0.7411 | 1.9315 | 0.2421 | 2814.0000 | 683.5069 | False |
| True | frap | sumo1x3 | 911 | 75.4416 | -1.4321 | 3.4944 | 0.3022 | 2808.0000 | 1486.0490 | False |
| True | frap | sumo1x3 | 131 | 82.1048 | -2.6566 | 4.8565 | 0.3254 | 2806.0000 | 1627.5971 | False |
| True | frap | sumo1x3 | 337 | 198.5883 | -12.5244 | 26.6657 | 0.5656 | 1902.0000 | 1635.5139 | False |
| True | dqn | sumo4x4 | 131 | 140.9549 | -8.3614 | 0.6101 | 0.0420 | 1464.0000 | 2128.0391 | False |
| True | dqn | sumo4x4 | 337 | 140.5116 | -8.3175 | 0.6042 | 0.0406 | 1466.0000 | 2162.2640 | False |
| True | dqn | sumo4x4 | 911 | 140.3625 | -8.0706 | 0.5938 | 0.0402 | 1465.0000 | 2132.0151 | False |
| True | presslight | sumo4x4 | 131 | 144.0137 | -8.0653 | 0.7045 | 0.0480 | 1464.0000 | 2099.8808 | False |
| True | presslight | sumo4x4 | 337 | 144.5089 | -8.1236 | 0.7052 | 0.0481 | 1464.0000 | 2101.9488 | False |
| True | mplight | sumo4x4 | 131 | 140.4542 | -0.6816 | 0.6073 | 0.0425 | 1464.0000 | 984.1661 | False |
| True | presslight | sumo4x4 | 911 | 144.3153 | -8.1933 | 0.7224 | 0.0483 | 1462.0000 | 2038.2985 | False |
| True | mplight | sumo4x4 | 337 | 140.7691 | -0.6925 | 0.6214 | 0.0431 | 1464.0000 | 1023.2936 | False |
| True | mplight | sumo4x4 | 911 | 140.1836 | -0.6656 | 0.5934 | 0.0412 | 1465.0000 | 1044.4305 | False |
| True | frap | sumo4x4 | 131 | 140.5010 | -0.6834 | 0.6047 | 0.0424 | 1463.0000 | 7170.0094 | False |
| True | frap | sumo4x4 | 337 | 140.6393 | -0.6994 | 0.6217 | 0.0424 | 1464.0000 | 7249.5308 | False |
| True | frap | sumo4x4 | 911 | 140.2915 | -0.6906 | 0.6106 | 0.0419 | 1465.0000 | 7145.0136 | False |

## Notes

- The adapter parses the final metric line printed by LibSignal.
- Use larger episode/step budgets before citing RL numbers as trained baselines.
- The local LibSignal checkout has SUMO-only compatibility patches for optional CityFlow/CoLight imports.
