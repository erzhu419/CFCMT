# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 16 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_core_advantage_contrast_mpc | 4.5399 | 0.9022 | 1.3201 | 0.5254 |
| dense_balanced_advantage_contrast_mpc | 4.5828 | 0.9178 | 1.3129 | 0.5168 |
| causal_balanced_advantage_contrast_mpc | 4.5950 | 0.9248 | 1.3377 | 0.5173 |
| dense_balanced_advantage_contrast_regularized | 4.5965 | 0.8974 | 1.2902 | 0.5238 |
| causal_core_advantage_contrast_guard | 4.6099 | 0.9002 | 1.2890 | 0.5224 |
| causal_core_advantage_contrast_regularized | 4.6131 | 0.8814 | 1.2550 | 0.5275 |
| causal_target_only_contrast_guard | 4.6208 | 0.8880 | 1.2947 | 0.5235 |
| causal_target_adapter_contrast_guard | 4.6387 | 0.8781 | 1.2732 | 0.5233 |
| dense_balanced_advantage_contrast_guard | 4.6555 | 0.9068 | 1.2790 | 0.5202 |
| causal_rigid_advantage_contrast_regularized | 4.6752 | 0.8864 | 1.2532 | 0.5245 |
| cfcmt_mechanism_contrast_regularized | 4.6792 | 0.8829 | 1.2403 | 0.5257 |
| cfcmt_mechanism_contrast_guard | 4.7095 | 0.8944 | 1.2702 | 0.5213 |
| causal_rigid_advantage_contrast_guard | 4.7110 | 0.9041 | 1.2841 | 0.5191 |
| causal_rigid_advantage_contrast_mpc | 4.7752 | 0.9105 | 1.3349 | 0.5035 |
| cfcmt_mechanism_contrast_mpc | 4.9806 | 0.9777 | 1.4008 | 0.4991 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0742 | 0.9144 | 1.2437 | 0.5035 |
| dense_contrast_guard | 5.0805 | 0.9153 | 1.2470 | 0.5021 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
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
| fixed_program | -0.1395 | [-0.2371, -0.0370] | 15/16 |
| max_pressure | -0.0191 | [-0.0590, 0.0066] | 9/16 |
| phase_pressure | -0.0336 | [-0.0768, 0.0041] | 7/16 |
| spillback_pressure | -0.0564 | [-0.1103, -0.0139] | 15/16 |
| selected_source_prior | -0.0197 | [-0.0585, 0.0000] | 2/16 |
| simulator_contrast_guard | -0.0184 | [-0.0579, 0.0055] | 4/16 |
| dense_contrast_guard | -0.0230 | [-0.0601, 0.0004] | 5/16 |
| dense_balanced_advantage_contrast_mpc | 0.0022 | [-0.0190, 0.0245] | 10/16 |
| dense_balanced_advantage_contrast_regularized | 0.0068 | [-0.0003, 0.0176] | 0/16 |
| dense_balanced_advantage_contrast_guard | 0.0001 | [-0.0074, 0.0076] | 4/16 |
| causal_core_advantage_contrast_mpc | 0.0011 | [-0.0193, 0.0237] | 12/16 |
| causal_core_advantage_contrast_regularized | 0.0034 | [-0.0033, 0.0125] | 1/16 |
| causal_core_advantage_contrast_guard | 0.0048 | [-0.0024, 0.0141] | 3/16 |
| causal_balanced_advantage_contrast_mpc | -0.0032 | [-0.0174, 0.0151] | 13/16 |
| causal_balanced_advantage_contrast_regularized | -0.0197 | [-0.0585, 0.0000] | 2/16 |
| causal_balanced_advantage_contrast_guard | -0.0197 | [-0.0585, 0.0000] | 2/16 |
| causal_rigid_advantage_contrast_mpc | -0.0218 | [-0.0405, -0.0052] | 11/16 |
| causal_rigid_advantage_contrast_regularized | 0.0002 | [-0.0023, 0.0035] | 1/16 |
| causal_rigid_advantage_contrast_guard | -0.0026 | [-0.0081, 0.0006] | 5/16 |
| cfcmt_mechanism_contrast_mpc | -0.0332 | [-0.0867, -0.0010] | 13/16 |
| cfcmt_mechanism_contrast_guard | -0.0027 | [-0.0081, 0.0006] | 5/16 |
| causal_target_adapter_contrast_guard | 0.0020 | [-0.0026, 0.0095] | 3/16 |
| causal_target_only_contrast_guard | 0.0016 | [-0.0051, 0.0100] | 2/16 |
