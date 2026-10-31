# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 120 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_target_only_contrast_guard | 4.5780 | 0.8713 | 1.2521 | 0.5291 |
| dense_balanced_advantage_contrast_regularized | 4.6356 | 0.8941 | 1.2803 | 0.5255 |
| causal_core_advantage_contrast_regularized | 4.6532 | 0.8806 | 1.2397 | 0.5263 |
| causal_core_advantage_contrast_guard | 4.6644 | 0.8879 | 1.2501 | 0.5246 |
| causal_target_adapter_contrast_guard | 4.6702 | 0.8784 | 1.2699 | 0.5239 |
| dense_balanced_advantage_contrast_mpc | 4.6761 | 0.9170 | 1.3325 | 0.5196 |
| causal_core_advantage_contrast_mpc | 4.6825 | 0.8932 | 1.2806 | 0.5171 |
| causal_balanced_advantage_contrast_mpc | 4.7517 | 0.9248 | 1.3128 | 0.5114 |
| causal_rigid_advantage_contrast_regularized | 4.8386 | 0.9024 | 1.2769 | 0.5151 |
| cfcmt_mechanism_contrast_regularized | 4.8386 | 0.9024 | 1.2769 | 0.5151 |
| causal_rigid_advantage_contrast_mpc | 4.8755 | 0.9250 | 1.2828 | 0.5101 |
| cfcmt_mechanism_contrast_mpc | 4.8780 | 0.8860 | 1.2490 | 0.5108 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0742 | 0.9141 | 1.2440 | 0.5041 |
| dense_contrast_guard | 5.0797 | 0.9145 | 1.2457 | 0.5042 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| dense_balanced_advantage_contrast_guard | 5.0819 | 0.9234 | 1.2523 | 0.5043 |
| cfcmt_mechanism_contrast_guard | 5.0851 | 0.9190 | 1.2508 | 0.5024 |
| causal_rigid_advantage_contrast_guard | 5.0855 | 0.9193 | 1.2521 | 0.5025 |
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
| fixed_program | -0.1300 | [-0.2361, -0.0119] | 15/16 |
| max_pressure | -0.0111 | [-0.0363, 0.0069] | 9/16 |
| phase_pressure | -0.0256 | [-0.0601, 0.0053] | 7/16 |
| spillback_pressure | -0.0483 | [-0.1005, -0.0129] | 15/16 |
| selected_source_prior | -0.0117 | [-0.0352, 0.0000] | 1/16 |
| simulator_contrast_guard | -0.0104 | [-0.0351, 0.0056] | 5/16 |
| dense_contrast_guard | -0.0105 | [-0.0347, 0.0033] | 2/16 |
| dense_balanced_advantage_contrast_mpc | 0.0087 | [-0.0065, 0.0305] | 8/16 |
| dense_balanced_advantage_contrast_regularized | 0.0110 | [-0.0009, 0.0341] | 1/16 |
| dense_balanced_advantage_contrast_guard | -0.0124 | [-0.0358, 0.0020] | 5/16 |
| causal_core_advantage_contrast_mpc | 0.0027 | [-0.0216, 0.0269] | 9/16 |
| causal_core_advantage_contrast_regularized | 0.0108 | [0.0000, 0.0309] | 0/16 |
| causal_core_advantage_contrast_guard | 0.0073 | [-0.0076, 0.0285] | 5/16 |
| causal_balanced_advantage_contrast_mpc | -0.0068 | [-0.0292, 0.0162] | 14/16 |
| causal_balanced_advantage_contrast_regularized | -0.0117 | [-0.0352, 0.0000] | 1/16 |
| causal_balanced_advantage_contrast_guard | -0.0117 | [-0.0352, 0.0000] | 1/16 |
| causal_rigid_advantage_contrast_mpc | -0.0070 | [-0.0191, 0.0058] | 13/16 |
| causal_rigid_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_guard | -0.0132 | [-0.0363, 0.0001] | 5/16 |
| cfcmt_mechanism_contrast_mpc | -0.0085 | [-0.0218, 0.0033] | 11/16 |
| cfcmt_mechanism_contrast_guard | -0.0129 | [-0.0360, 0.0001] | 5/16 |
| causal_target_adapter_contrast_guard | 0.0100 | [-0.0002, 0.0295] | 1/16 |
| causal_target_only_contrast_guard | 0.0174 | [-0.0005, 0.0447] | 1/16 |
