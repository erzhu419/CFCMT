# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 32 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_target_only_contrast_mpc | 4.4786 | 0.8721 | 1.2512 | 0.5292 |
| dense_balanced_advantage_contrast_mpc | 4.6107 | 0.9081 | 1.3428 | 0.5237 |
| causal_core_advantage_contrast_regularized | 4.6200 | 0.8859 | 1.2538 | 0.5273 |
| dense_balanced_advantage_contrast_regularized | 4.6256 | 0.8752 | 1.2494 | 0.5272 |
| cfcmt_mechanism_contrast_regularized | 4.6442 | 0.8744 | 1.2702 | 0.5244 |
| causal_target_only_contrast_regularized | 4.6442 | 0.8744 | 1.2702 | 0.5244 |
| cfcmt_mechanism_contrast_guard | 4.6585 | 0.8815 | 1.2616 | 0.5232 |
| causal_target_only_contrast_guard | 4.6585 | 0.8815 | 1.2616 | 0.5232 |
| causal_core_advantage_contrast_guard | 4.6806 | 0.8825 | 1.2553 | 0.5237 |
| causal_core_advantage_contrast_mpc | 4.7117 | 0.8886 | 1.2746 | 0.5146 |
| causal_balanced_advantage_contrast_mpc | 4.7616 | 0.9395 | 1.3197 | 0.5119 |
| causal_rigid_advantage_contrast_regularized | 4.8104 | 0.8918 | 1.2658 | 0.5178 |
| causal_rigid_advantage_contrast_guard | 4.8956 | 0.9077 | 1.2643 | 0.5127 |
| cfcmt_mechanism_contrast_mpc | 4.8998 | 0.9052 | 1.2636 | 0.5115 |
| causal_rigid_advantage_contrast_mpc | 4.9242 | 0.9302 | 1.3165 | 0.5094 |
| dense_balanced_advantage_contrast_guard | 4.9361 | 0.9019 | 1.2613 | 0.5132 |
| causal_target_adapter_contrast_mpc | 5.0012 | 0.9154 | 1.2927 | 0.5047 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0742 | 0.9142 | 1.2431 | 0.5032 |
| dense_contrast_guard | 5.0808 | 0.9181 | 1.2520 | 0.5020 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_target_adapter_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_target_adapter_contrast_guard | 5.0820 | 0.9169 | 1.2477 | 0.5025 |
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
| fixed_program | -0.1413 | [-0.2361, -0.0431] | 15/16 |
| max_pressure | -0.0204 | [-0.0643, 0.0069] | 9/16 |
| phase_pressure | -0.0349 | [-0.0812, 0.0052] | 7/16 |
| spillback_pressure | -0.0577 | [-0.1129, -0.0129] | 15/16 |
| selected_source_prior | -0.0210 | [-0.0633, 0.0000] | 1/16 |
| simulator_contrast_guard | -0.0199 | [-0.0631, 0.0056] | 5/16 |
| dense_contrast_guard | -0.0225 | [-0.0644, 0.0006] | 6/16 |
| dense_balanced_advantage_contrast_mpc | -0.0002 | [-0.0096, 0.0098] | 7/16 |
| dense_balanced_advantage_contrast_regularized | 0.0022 | [-0.0004, 0.0068] | 1/16 |
| dense_balanced_advantage_contrast_guard | -0.0145 | [-0.0439, 0.0037] | 4/16 |
| causal_core_advantage_contrast_mpc | -0.0068 | [-0.0229, 0.0097] | 9/16 |
| causal_core_advantage_contrast_regularized | 0.0012 | [-0.0006, 0.0045] | 1/16 |
| causal_core_advantage_contrast_guard | -0.0013 | [-0.0081, 0.0066] | 5/16 |
| causal_balanced_advantage_contrast_mpc | -0.0139 | [-0.0277, 0.0021] | 14/16 |
| causal_balanced_advantage_contrast_regularized | -0.0210 | [-0.0633, 0.0000] | 1/16 |
| causal_balanced_advantage_contrast_guard | -0.0210 | [-0.0633, 0.0000] | 1/16 |
| causal_rigid_advantage_contrast_mpc | -0.0221 | [-0.0449, -0.0013] | 13/16 |
| causal_rigid_advantage_contrast_regularized | -0.0088 | [-0.0265, 0.0000] | 2/16 |
| causal_rigid_advantage_contrast_guard | -0.0142 | [-0.0386, 0.0000] | 5/16 |
| cfcmt_mechanism_contrast_mpc | -0.0170 | [-0.0431, 0.0040] | 10/16 |
| cfcmt_mechanism_contrast_guard | 0.0001 | [-0.0054, 0.0056] | 3/16 |
| causal_target_adapter_contrast_mpc | -0.0178 | [-0.0536, 0.0054] | 8/16 |
| causal_target_adapter_contrast_regularized | -0.0210 | [-0.0633, 0.0000] | 1/16 |
| causal_target_adapter_contrast_guard | -0.0214 | [-0.0635, 0.0000] | 4/16 |
| causal_target_only_contrast_mpc | 0.0081 | [-0.0111, 0.0294] | 7/16 |
| causal_target_only_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_target_only_contrast_guard | 0.0001 | [-0.0054, 0.0056] | 3/16 |
