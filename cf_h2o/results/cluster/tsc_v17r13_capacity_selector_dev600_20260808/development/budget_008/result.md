# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 8 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_core_advantage_contrast_guard | 4.6221 | 0.8865 | 1.2638 | 0.5267 |
| causal_core_advantage_contrast_mpc | 4.6942 | 0.9022 | 1.3068 | 0.5156 |
| causal_core_advantage_contrast_regularized | 4.7322 | 0.9067 | 1.2755 | 0.5195 |
| causal_target_only_contrast_mpc | 4.7489 | 0.9072 | 1.2737 | 0.5151 |
| dense_balanced_advantage_contrast_mpc | 4.7592 | 0.9275 | 1.3560 | 0.5128 |
| dense_balanced_advantage_contrast_guard | 4.7744 | 0.9023 | 1.2790 | 0.5175 |
| dense_balanced_advantage_contrast_regularized | 4.8130 | 0.9038 | 1.2909 | 0.5146 |
| causal_balanced_advantage_contrast_mpc | 4.8538 | 0.9507 | 1.3514 | 0.5102 |
| causal_rigid_advantage_contrast_mpc | 4.8725 | 0.9297 | 1.3098 | 0.5079 |
| cfcmt_mechanism_contrast_mpc | 4.9271 | 0.9160 | 1.2770 | 0.5063 |
| cfcmt_mechanism_contrast_guard | 4.9493 | 0.8983 | 1.2627 | 0.5131 |
| causal_target_only_contrast_guard | 4.9493 | 0.8983 | 1.2627 | 0.5131 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0798 | 0.9130 | 1.2380 | 0.5040 |
| causal_rigid_advantage_contrast_guard | 5.0806 | 0.9142 | 1.2398 | 0.5034 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_target_adapter_contrast_regularized | 5.0818 | 0.9154 | 1.2441 | 0.5035 |
| cfcmt_mechanism_contrast_regularized | 5.0818 | 0.9155 | 1.2437 | 0.5035 |
| causal_target_only_contrast_regularized | 5.0818 | 0.9155 | 1.2437 | 0.5035 |
| causal_rigid_advantage_contrast_regularized | 5.0820 | 0.9158 | 1.2444 | 0.5032 |
| causal_target_adapter_contrast_guard | 5.0823 | 0.9134 | 1.2378 | 0.5054 |
| max_pressure | 5.0872 | 0.9305 | 1.2768 | 0.5038 |
| dense_contrast_guard | 5.0872 | 0.9207 | 1.2533 | 0.5028 |
| causal_target_adapter_contrast_mpc | 5.0890 | 0.9205 | 1.2730 | 0.5015 |
| phase_pressure | 5.1061 | 0.9101 | 1.2048 | 0.4917 |
| spillback_pressure | 5.1532 | 0.9454 | 1.2956 | 0.4764 |

## Selected Rule Prior

| Target | Prior |
|---|---|
| grid4x4 | gp_d0p45_o0_s0 |
| arterial4x4 | gp_d0p45_o0_s0 |
| cologne1 | gp_d0p45_o0_s0 |
| cologne3 | gp_d0p45_o0_s0 |
| cologne8 | gp_d0p45_o0_s0 |
| ingolstadt1 | max_pressure |
| ingolstadt7 | max_pressure |
| ingolstadt21 | max_pressure |
| atlanta_1x5 | gp_d0p45_o0_s0 |
| hangzhou_bc_tyc | gp_d0p45_o0_s0 |
| hangzhou_kn_hz | gp_d0p45_o0_s0 |
| hangzhou_qc_yn | gp_d0p45_o0_s0 |
| hangzhou_sb_sx | gp_d0p45_o0_s0 |
| hangzhou_4x4 | gp_d0p45_o0_s0 |
| hangzhou_4x4_hetero | gp_d0p45_o0_s0 |
| manhattan_28x7 | gp_d0p45_o0_s0 |

## Hierarchical Bootstrap

| Baseline | Relative delta | 95% CI | Wins |
|---|---:|---:|---:|
| fixed_program | -0.1158 | [-0.2357, 0.0275] | 15/16 |
| max_pressure | 0.0008 | [-0.0041, 0.0094] | 8/16 |
| phase_pressure | -0.0137 | [-0.0463, 0.0094] | 7/16 |
| spillback_pressure | -0.0365 | [-0.0910, -0.0044] | 14/16 |
| selected_source_prior | 0.0002 | [0.0000, 0.0007] | 0/16 |
| simulator_contrast_guard | 0.0006 | [-0.0001, 0.0028] | 1/16 |
| dense_contrast_guard | -0.0013 | [-0.0057, 0.0020] | 2/16 |
| dense_balanced_advantage_contrast_mpc | 0.0089 | [-0.0197, 0.0500] | 11/16 |
| dense_balanced_advantage_contrast_regularized | 0.0140 | [-0.0000, 0.0422] | 0/16 |
| dense_balanced_advantage_contrast_guard | 0.0166 | [-0.0004, 0.0491] | 0/16 |
| causal_core_advantage_contrast_mpc | 0.0126 | [-0.0224, 0.0634] | 11/16 |
| causal_core_advantage_contrast_regularized | 0.0188 | [-0.0000, 0.0571] | 1/16 |
| causal_core_advantage_contrast_guard | 0.0255 | [-0.0001, 0.0767] | 1/16 |
| causal_balanced_advantage_contrast_mpc | 0.0005 | [-0.0267, 0.0369] | 14/16 |
| causal_balanced_advantage_contrast_regularized | 0.0002 | [0.0000, 0.0007] | 0/16 |
| causal_balanced_advantage_contrast_guard | 0.0002 | [0.0000, 0.0007] | 0/16 |
| causal_rigid_advantage_contrast_mpc | 0.0027 | [-0.0186, 0.0327] | 11/16 |
| causal_rigid_advantage_contrast_regularized | -0.0002 | [-0.0008, 0.0000] | 1/16 |
| causal_rigid_advantage_contrast_guard | 0.0004 | [-0.0004, 0.0023] | 0/16 |
| cfcmt_mechanism_contrast_mpc | -0.0062 | [-0.0350, 0.0235] | 10/16 |
| cfcmt_mechanism_contrast_guard | 0.0066 | [-0.0002, 0.0198] | 1/16 |
| causal_target_adapter_contrast_mpc | -0.0021 | [-0.0117, 0.0062] | 10/16 |
| causal_target_adapter_contrast_regularized | 0.0000 | [-0.0001, 0.0001] | 1/16 |
| causal_target_adapter_contrast_guard | -0.0003 | [-0.0033, 0.0016] | 1/16 |
| causal_target_only_contrast_mpc | 0.0120 | [-0.0153, 0.0521] | 5/16 |
| causal_target_only_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_target_only_contrast_guard | 0.0066 | [-0.0002, 0.0198] | 1/16 |
