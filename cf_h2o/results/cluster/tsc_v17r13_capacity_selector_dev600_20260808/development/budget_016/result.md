# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 16 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| dense_balanced_advantage_contrast_mpc | 4.5966 | 0.9155 | 1.3354 | 0.5226 |
| cfcmt_mechanism_contrast_regularized | 4.6153 | 0.8713 | 1.2349 | 0.5261 |
| causal_target_only_contrast_regularized | 4.6153 | 0.8713 | 1.2349 | 0.5261 |
| causal_target_only_contrast_mpc | 4.6640 | 0.8678 | 1.2661 | 0.5210 |
| causal_core_advantage_contrast_regularized | 4.7129 | 0.8878 | 1.2542 | 0.5234 |
| dense_balanced_advantage_contrast_regularized | 4.7229 | 0.8909 | 1.2744 | 0.5203 |
| causal_core_advantage_contrast_guard | 4.7346 | 0.8921 | 1.2636 | 0.5209 |
| causal_core_advantage_contrast_mpc | 4.7781 | 0.9142 | 1.3030 | 0.5109 |
| causal_balanced_advantage_contrast_mpc | 4.7933 | 0.9287 | 1.3123 | 0.5155 |
| cfcmt_mechanism_contrast_guard | 4.8290 | 0.8999 | 1.2912 | 0.5123 |
| causal_target_only_contrast_guard | 4.8290 | 0.8999 | 1.2912 | 0.5123 |
| causal_rigid_advantage_contrast_mpc | 4.8460 | 0.9273 | 1.3103 | 0.5127 |
| causal_rigid_advantage_contrast_guard | 4.8739 | 0.9026 | 1.2688 | 0.5144 |
| causal_rigid_advantage_contrast_regularized | 4.8797 | 0.9097 | 1.2587 | 0.5111 |
| cfcmt_mechanism_contrast_mpc | 4.9160 | 0.9121 | 1.2783 | 0.5077 |
| dense_balanced_advantage_contrast_guard | 4.9988 | 0.9077 | 1.2394 | 0.5084 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0742 | 0.9143 | 1.2425 | 0.5038 |
| dense_contrast_guard | 5.0801 | 0.9150 | 1.2462 | 0.5035 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_target_adapter_contrast_guard | 5.0817 | 0.9156 | 1.2444 | 0.5030 |
| causal_target_adapter_contrast_regularized | 5.0818 | 0.9156 | 1.2442 | 0.5035 |
| max_pressure | 5.0872 | 0.9305 | 1.2768 | 0.5038 |
| causal_target_adapter_contrast_mpc | 5.0881 | 0.9165 | 1.2671 | 0.5031 |
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
| fixed_program | -0.1427 | [-0.2361, -0.0476] | 15/16 |
| max_pressure | -0.0214 | [-0.0688, 0.0075] | 8/16 |
| phase_pressure | -0.0359 | [-0.0850, 0.0056] | 7/16 |
| spillback_pressure | -0.0587 | [-0.1153, -0.0124] | 14/16 |
| selected_source_prior | -0.0220 | [-0.0685, 0.0017] | 2/16 |
| simulator_contrast_guard | -0.0208 | [-0.0675, 0.0057] | 5/16 |
| dense_contrast_guard | -0.0218 | [-0.0682, 0.0026] | 3/16 |
| dense_balanced_advantage_contrast_mpc | -0.0014 | [-0.0171, 0.0110] | 8/16 |
| dense_balanced_advantage_contrast_regularized | -0.0045 | [-0.0183, 0.0055] | 2/16 |
| dense_balanced_advantage_contrast_guard | -0.0185 | [-0.0579, 0.0024] | 3/16 |
| causal_core_advantage_contrast_mpc | -0.0147 | [-0.0326, 0.0007] | 8/16 |
| causal_core_advantage_contrast_regularized | -0.0052 | [-0.0181, 0.0009] | 3/16 |
| causal_core_advantage_contrast_guard | -0.0063 | [-0.0207, 0.0017] | 4/16 |
| causal_balanced_advantage_contrast_mpc | -0.0155 | [-0.0344, 0.0041] | 13/16 |
| causal_balanced_advantage_contrast_regularized | -0.0220 | [-0.0685, 0.0017] | 2/16 |
| causal_balanced_advantage_contrast_guard | -0.0220 | [-0.0685, 0.0017] | 2/16 |
| causal_rigid_advantage_contrast_mpc | -0.0213 | [-0.0410, -0.0026] | 13/16 |
| causal_rigid_advantage_contrast_regularized | -0.0133 | [-0.0419, 0.0012] | 2/16 |
| causal_rigid_advantage_contrast_guard | -0.0132 | [-0.0409, 0.0014] | 3/16 |
| cfcmt_mechanism_contrast_mpc | -0.0262 | [-0.0542, -0.0018] | 10/16 |
| cfcmt_mechanism_contrast_guard | -0.0111 | [-0.0352, 0.0010] | 3/16 |
| causal_target_adapter_contrast_mpc | -0.0247 | [-0.0711, 0.0053] | 8/16 |
| causal_target_adapter_contrast_regularized | -0.0222 | [-0.0685, 0.0013] | 3/16 |
| causal_target_adapter_contrast_guard | -0.0221 | [-0.0686, 0.0015] | 4/16 |
| causal_target_only_contrast_mpc | -0.0099 | [-0.0200, -0.0006] | 8/16 |
| causal_target_only_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_target_only_contrast_guard | -0.0111 | [-0.0352, 0.0010] | 3/16 |
