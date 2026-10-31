# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 60 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| dense_balanced_advantage_contrast_mpc | 4.5068 | 0.9118 | 1.3464 | 0.5257 |
| causal_target_only_contrast_mpc | 4.5103 | 0.8736 | 1.2691 | 0.5290 |
| dense_balanced_advantage_contrast_regularized | 4.5698 | 0.8896 | 1.2584 | 0.5283 |
| causal_target_only_contrast_guard | 4.5717 | 0.8752 | 1.2649 | 0.5269 |
| cfcmt_mechanism_contrast_guard | 4.5748 | 0.8751 | 1.2727 | 0.5262 |
| causal_core_advantage_contrast_regularized | 4.6048 | 0.8826 | 1.2464 | 0.5285 |
| cfcmt_mechanism_contrast_regularized | 4.6439 | 0.8854 | 1.2776 | 0.5237 |
| causal_target_only_contrast_regularized | 4.6543 | 0.8861 | 1.2755 | 0.5233 |
| causal_core_advantage_contrast_guard | 4.6738 | 0.8865 | 1.2534 | 0.5245 |
| causal_target_adapter_contrast_guard | 4.7134 | 0.8891 | 1.2642 | 0.5217 |
| causal_core_advantage_contrast_mpc | 4.7479 | 0.9241 | 1.3365 | 0.5140 |
| causal_rigid_advantage_contrast_regularized | 4.7629 | 0.8979 | 1.2722 | 0.5191 |
| causal_balanced_advantage_contrast_mpc | 4.7758 | 0.9233 | 1.2958 | 0.5099 |
| causal_target_adapter_contrast_regularized | 4.8094 | 0.8868 | 1.2534 | 0.5186 |
| causal_target_adapter_contrast_mpc | 4.8156 | 0.9068 | 1.2915 | 0.5117 |
| causal_rigid_advantage_contrast_guard | 4.8641 | 0.9039 | 1.2911 | 0.5139 |
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
| fixed_program | -0.1412 | [-0.2360, -0.0430] | 15/16 |
| max_pressure | -0.0203 | [-0.0642, 0.0070] | 9/16 |
| phase_pressure | -0.0348 | [-0.0813, 0.0053] | 7/16 |
| spillback_pressure | -0.0576 | [-0.1127, -0.0127] | 14/16 |
| selected_source_prior | -0.0209 | [-0.0637, 0.0005] | 1/16 |
| simulator_contrast_guard | -0.0196 | [-0.0630, 0.0058] | 2/16 |
| dense_contrast_guard | -0.0209 | [-0.0638, 0.0022] | 2/16 |
| dense_balanced_advantage_contrast_mpc | 0.0067 | [-0.0103, 0.0257] | 8/16 |
| dense_balanced_advantage_contrast_regularized | 0.0041 | [-0.0007, 0.0134] | 1/16 |
| dense_balanced_advantage_contrast_guard | -0.0134 | [-0.0435, 0.0028] | 3/16 |
| causal_core_advantage_contrast_mpc | -0.0084 | [-0.0262, 0.0104] | 10/16 |
| causal_core_advantage_contrast_regularized | 0.0030 | [0.0000, 0.0080] | 0/16 |
| causal_core_advantage_contrast_guard | 0.0002 | [-0.0097, 0.0119] | 5/16 |
| causal_balanced_advantage_contrast_mpc | -0.0124 | [-0.0300, 0.0066] | 10/16 |
| causal_balanced_advantage_contrast_regularized | -0.0209 | [-0.0637, 0.0005] | 1/16 |
| causal_balanced_advantage_contrast_guard | -0.0209 | [-0.0637, 0.0005] | 1/16 |
| causal_rigid_advantage_contrast_mpc | -0.0248 | [-0.0431, -0.0072] | 14/16 |
| causal_rigid_advantage_contrast_regularized | -0.0063 | [-0.0192, 0.0002] | 1/16 |
| causal_rigid_advantage_contrast_guard | -0.0127 | [-0.0348, 0.0000] | 4/16 |
| cfcmt_mechanism_contrast_mpc | -0.0224 | [-0.0478, -0.0012] | 11/16 |
| cfcmt_mechanism_contrast_guard | 0.0061 | [-0.0024, 0.0163] | 4/16 |
| causal_target_adapter_contrast_mpc | -0.0087 | [-0.0306, 0.0098] | 6/16 |
| causal_target_adapter_contrast_regularized | -0.0085 | [-0.0266, 0.0005] | 1/16 |
| causal_target_adapter_contrast_guard | -0.0034 | [-0.0122, 0.0008] | 2/16 |
| causal_target_only_contrast_mpc | 0.0022 | [-0.0147, 0.0220] | 8/16 |
| causal_target_only_contrast_regularized | -0.0005 | [-0.0023, 0.0005] | 1/16 |
| causal_target_only_contrast_guard | 0.0072 | [-0.0018, 0.0175] | 2/16 |
