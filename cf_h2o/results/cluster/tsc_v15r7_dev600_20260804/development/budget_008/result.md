# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 8 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_core_advantage_contrast_mpc | 4.4503 | 0.8877 | 1.3006 | 0.5294 |
| dense_balanced_advantage_contrast_mpc | 4.5645 | 0.8463 | 1.2447 | 0.5289 |
| causal_rigid_advantage_contrast_mpc | 4.6757 | 0.9276 | 1.3677 | 0.5135 |
| causal_balanced_advantage_contrast_mpc | 4.7215 | 0.9091 | 1.3037 | 0.5138 |
| cfcmt_mechanism_contrast_mpc | 4.8203 | 0.9500 | 1.3466 | 0.5098 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| dense_balanced_advantage_contrast_regularized | 5.0767 | 0.9140 | 1.2406 | 0.5039 |
| causal_core_advantage_contrast_regularized | 5.0793 | 0.9155 | 1.2406 | 0.5041 |
| simulator_contrast_guard | 5.0798 | 0.9131 | 1.2389 | 0.5035 |
| causal_rigid_advantage_contrast_guard | 5.0807 | 0.9143 | 1.2404 | 0.5025 |
| cfcmt_mechanism_contrast_guard | 5.0807 | 0.9143 | 1.2404 | 0.5025 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_rigid_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| cfcmt_mechanism_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_target_only_contrast_guard | 5.0816 | 0.9152 | 1.2436 | 0.5034 |
| causal_core_advantage_contrast_guard | 5.0817 | 0.9158 | 1.2452 | 0.5038 |
| causal_target_adapter_contrast_guard | 5.0818 | 0.9153 | 1.2431 | 0.5035 |
| max_pressure | 5.0872 | 0.9305 | 1.2768 | 0.5038 |
| dense_balanced_advantage_contrast_guard | 5.0934 | 0.9223 | 1.2523 | 0.5027 |
| phase_pressure | 5.1061 | 0.9101 | 1.2048 | 0.4917 |
| dense_contrast_guard | 5.1130 | 0.9334 | 1.2661 | 0.5024 |
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
| simulator_contrast_guard | 0.0005 | [-0.0002, 0.0025] | 2/16 |
| dense_contrast_guard | -0.0047 | [-0.0222, 0.0021] | 3/16 |
| dense_balanced_advantage_contrast_mpc | 0.0237 | [-0.0165, 0.0869] | 10/16 |
| dense_balanced_advantage_contrast_regularized | 0.0019 | [0.0000, 0.0086] | 0/16 |
| dense_balanced_advantage_contrast_guard | -0.0010 | [-0.0118, 0.0077] | 3/16 |
| causal_core_advantage_contrast_mpc | 0.0327 | [-0.0134, 0.1092] | 9/16 |
| causal_core_advantage_contrast_regularized | 0.0010 | [-0.0029, 0.0090] | 0/16 |
| causal_core_advantage_contrast_guard | -0.0000 | [-0.0040, 0.0045] | 1/16 |
| causal_balanced_advantage_contrast_mpc | 0.0133 | [-0.0146, 0.0568] | 11/16 |
| causal_balanced_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_balanced_advantage_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_mpc | 0.0108 | [-0.0231, 0.0657] | 14/16 |
| causal_rigid_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_guard | -0.0003 | [-0.0019, 0.0017] | 2/16 |
| cfcmt_mechanism_contrast_mpc | 0.0046 | [-0.0371, 0.0486] | 12/16 |
| cfcmt_mechanism_contrast_guard | -0.0003 | [-0.0019, 0.0017] | 2/16 |
| causal_target_adapter_contrast_guard | -0.0003 | [-0.0010, 0.0000] | 2/16 |
| causal_target_only_contrast_guard | -0.0001 | [-0.0004, 0.0002] | 1/16 |
