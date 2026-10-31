# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 60 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| dense_balanced_advantage_contrast_mpc | 4.5567 | 0.8806 | 1.2728 | 0.5300 |
| causal_core_advantage_contrast_guard | 4.5968 | 0.8803 | 1.2631 | 0.5302 |
| causal_core_advantage_contrast_regularized | 4.6023 | 0.8811 | 1.2515 | 0.5288 |
| causal_target_only_contrast_guard | 4.6706 | 0.8868 | 1.2736 | 0.5210 |
| causal_target_adapter_contrast_guard | 4.7042 | 0.8730 | 1.2346 | 0.5244 |
| causal_core_advantage_contrast_mpc | 4.7447 | 0.9174 | 1.3370 | 0.5150 |
| causal_balanced_advantage_contrast_mpc | 4.7781 | 0.9224 | 1.3056 | 0.5110 |
| cfcmt_mechanism_contrast_mpc | 4.8695 | 0.8894 | 1.2312 | 0.5127 |
| causal_rigid_advantage_contrast_mpc | 4.8966 | 0.9233 | 1.2881 | 0.5118 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0748 | 0.9137 | 1.2447 | 0.5027 |
| dense_balanced_advantage_contrast_guard | 5.0809 | 0.9154 | 1.2407 | 0.5029 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_rigid_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| cfcmt_mechanism_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| dense_balanced_advantage_contrast_regularized | 5.0825 | 0.9175 | 1.2433 | 0.5040 |
| dense_contrast_guard | 5.0830 | 0.9189 | 1.2612 | 0.5040 |
| causal_rigid_advantage_contrast_guard | 5.0840 | 0.9184 | 1.2502 | 0.5037 |
| cfcmt_mechanism_contrast_guard | 5.0841 | 0.9188 | 1.2524 | 0.5030 |
| max_pressure | 5.0872 | 0.9305 | 1.2768 | 0.5038 |
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
| fixed_program | -0.1159 | [-0.2361, 0.0275] | 15/16 |
| max_pressure | 0.0006 | [-0.0043, 0.0092] | 9/16 |
| phase_pressure | -0.0139 | [-0.0464, 0.0091] | 7/16 |
| spillback_pressure | -0.0367 | [-0.0911, -0.0048] | 15/16 |
| selected_source_prior | 0.0000 | [0.0000, 0.0000] | 0/16 |
| simulator_contrast_guard | 0.0009 | [-0.0021, 0.0069] | 3/16 |
| dense_contrast_guard | 0.0000 | [-0.0025, 0.0024] | 1/16 |
| dense_balanced_advantage_contrast_mpc | 0.0295 | [-0.0055, 0.0903] | 9/16 |
| dense_balanced_advantage_contrast_regularized | -0.0002 | [-0.0013, 0.0003] | 1/16 |
| dense_balanced_advantage_contrast_guard | -0.0004 | [-0.0025, 0.0022] | 2/16 |
| causal_core_advantage_contrast_mpc | 0.0128 | [-0.0214, 0.0581] | 7/16 |
| causal_core_advantage_contrast_regularized | 0.0273 | [0.0000, 0.0802] | 0/16 |
| causal_core_advantage_contrast_guard | 0.0265 | [-0.0049, 0.0815] | 5/16 |
| causal_balanced_advantage_contrast_mpc | 0.0072 | [-0.0222, 0.0479] | 13/16 |
| causal_balanced_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_balanced_advantage_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_mpc | 0.0018 | [-0.0196, 0.0301] | 12/16 |
| causal_rigid_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_guard | -0.0004 | [-0.0022, 0.0005] | 2/16 |
| cfcmt_mechanism_contrast_mpc | 0.0064 | [-0.0129, 0.0317] | 10/16 |
| cfcmt_mechanism_contrast_guard | -0.0005 | [-0.0024, 0.0002] | 2/16 |
| causal_target_adapter_contrast_guard | 0.0211 | [-0.0000, 0.0621] | 1/16 |
| causal_target_only_contrast_guard | 0.0284 | [0.0012, 0.0679] | 4/16 |
