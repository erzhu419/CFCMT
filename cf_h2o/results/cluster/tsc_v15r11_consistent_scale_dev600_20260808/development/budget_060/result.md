# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 60 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| dense_balanced_advantage_contrast_mpc | 4.5068 | 0.9118 | 1.3464 | 0.5257 |
| dense_balanced_advantage_contrast_regularized | 4.5698 | 0.8896 | 1.2584 | 0.5283 |
| causal_target_only_contrast_guard | 4.5717 | 0.8752 | 1.2649 | 0.5269 |
| causal_core_advantage_contrast_regularized | 4.6048 | 0.8826 | 1.2464 | 0.5285 |
| causal_core_advantage_contrast_guard | 4.6738 | 0.8865 | 1.2534 | 0.5245 |
| causal_target_adapter_contrast_guard | 4.7134 | 0.8891 | 1.2642 | 0.5217 |
| causal_core_advantage_contrast_mpc | 4.7479 | 0.9241 | 1.3365 | 0.5140 |
| causal_rigid_advantage_contrast_regularized | 4.7629 | 0.8979 | 1.2722 | 0.5191 |
| cfcmt_mechanism_contrast_regularized | 4.7629 | 0.8979 | 1.2722 | 0.5191 |
| causal_balanced_advantage_contrast_mpc | 4.7758 | 0.9233 | 1.2958 | 0.5099 |
| causal_rigid_advantage_contrast_guard | 4.8641 | 0.9039 | 1.2911 | 0.5139 |
| cfcmt_mechanism_contrast_guard | 4.8641 | 0.9039 | 1.2911 | 0.5139 |
| causal_rigid_advantage_contrast_mpc | 4.8864 | 0.9265 | 1.2831 | 0.5098 |
| cfcmt_mechanism_contrast_mpc | 4.9015 | 0.9032 | 1.2607 | 0.5088 |
| dense_balanced_advantage_contrast_guard | 4.9292 | 0.9075 | 1.2628 | 0.5108 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0741 | 0.9139 | 1.2416 | 0.5044 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| dense_contrast_guard | 5.0830 | 0.9189 | 1.2619 | 0.5041 |
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
| fixed_program | -0.1343 | [-0.2359, -0.0231] | 15/16 |
| max_pressure | -0.0146 | [-0.0471, 0.0070] | 9/16 |
| phase_pressure | -0.0291 | [-0.0673, 0.0054] | 7/16 |
| spillback_pressure | -0.0519 | [-0.1051, -0.0127] | 14/16 |
| selected_source_prior | -0.0152 | [-0.0468, 0.0005] | 1/16 |
| simulator_contrast_guard | -0.0139 | [-0.0461, 0.0058] | 2/16 |
| dense_contrast_guard | -0.0152 | [-0.0467, 0.0022] | 2/16 |
| dense_balanced_advantage_contrast_mpc | 0.0135 | [-0.0100, 0.0448] | 8/16 |
| dense_balanced_advantage_contrast_regularized | 0.0108 | [-0.0007, 0.0333] | 1/16 |
| dense_balanced_advantage_contrast_guard | -0.0075 | [-0.0258, 0.0028] | 3/16 |
| causal_core_advantage_contrast_mpc | -0.0020 | [-0.0209, 0.0157] | 9/16 |
| causal_core_advantage_contrast_regularized | 0.0096 | [0.0000, 0.0270] | 0/16 |
| causal_core_advantage_contrast_guard | 0.0067 | [-0.0044, 0.0192] | 4/16 |
| causal_balanced_advantage_contrast_mpc | -0.0061 | [-0.0241, 0.0118] | 9/16 |
| causal_balanced_advantage_contrast_regularized | -0.0152 | [-0.0468, 0.0005] | 1/16 |
| causal_balanced_advantage_contrast_guard | -0.0152 | [-0.0468, 0.0005] | 1/16 |
| causal_rigid_advantage_contrast_mpc | -0.0186 | [-0.0330, -0.0047] | 14/16 |
| causal_rigid_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_guard | -0.0066 | [-0.0166, 0.0000] | 4/16 |
| cfcmt_mechanism_contrast_mpc | -0.0163 | [-0.0336, -0.0005] | 11/16 |
| cfcmt_mechanism_contrast_guard | -0.0066 | [-0.0166, 0.0000] | 4/16 |
| causal_target_adapter_contrast_guard | 0.0029 | [-0.0001, 0.0102] | 1/16 |
| causal_target_only_contrast_guard | 0.0138 | [-0.0009, 0.0335] | 2/16 |
