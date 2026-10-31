# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 120 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_core_advantage_contrast_mpc | 4.4435 | 0.8953 | 1.3100 | 0.5309 |
| dense_balanced_advantage_contrast_mpc | 4.5185 | 0.9226 | 1.3460 | 0.5231 |
| causal_target_only_contrast_guard | 4.5484 | 0.8559 | 1.2281 | 0.5280 |
| causal_core_advantage_contrast_regularized | 4.5555 | 0.8868 | 1.2550 | 0.5294 |
| causal_target_adapter_contrast_guard | 4.5642 | 0.8803 | 1.2794 | 0.5251 |
| dense_balanced_advantage_contrast_regularized | 4.5666 | 0.8819 | 1.2477 | 0.5294 |
| causal_rigid_advantage_contrast_regularized | 4.5670 | 0.9057 | 1.3106 | 0.5251 |
| cfcmt_mechanism_contrast_regularized | 4.5826 | 0.9056 | 1.2988 | 0.5242 |
| causal_rigid_advantage_contrast_guard | 4.5883 | 0.8911 | 1.2732 | 0.5248 |
| cfcmt_mechanism_contrast_guard | 4.5883 | 0.8911 | 1.2732 | 0.5248 |
| causal_balanced_advantage_contrast_mpc | 4.6039 | 0.9083 | 1.2886 | 0.5194 |
| causal_rigid_advantage_contrast_mpc | 4.6449 | 0.9150 | 1.3256 | 0.5144 |
| cfcmt_mechanism_contrast_mpc | 4.7189 | 0.9451 | 1.3592 | 0.5129 |
| causal_core_advantage_contrast_guard | 4.8171 | 0.9030 | 1.2581 | 0.5156 |
| dense_balanced_advantage_contrast_guard | 5.0689 | 0.9137 | 1.2398 | 0.5044 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0742 | 0.9142 | 1.2446 | 0.5040 |
| dense_contrast_guard | 5.0797 | 0.9145 | 1.2457 | 0.5042 |
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
| fixed_program | -0.1449 | [-0.2361, -0.0526] | 15/16 |
| max_pressure | -0.0233 | [-0.0731, 0.0069] | 9/16 |
| phase_pressure | -0.0378 | [-0.0890, 0.0052] | 7/16 |
| spillback_pressure | -0.0606 | [-0.1175, -0.0129] | 15/16 |
| selected_source_prior | -0.0239 | [-0.0722, 0.0000] | 1/16 |
| simulator_contrast_guard | -0.0227 | [-0.0719, 0.0055] | 5/16 |
| dense_contrast_guard | -0.0227 | [-0.0717, 0.0034] | 2/16 |
| dense_balanced_advantage_contrast_mpc | -0.0023 | [-0.0167, 0.0134] | 10/16 |
| dense_balanced_advantage_contrast_regularized | 0.0016 | [-0.0012, 0.0058] | 0/16 |
| dense_balanced_advantage_contrast_guard | -0.0204 | [-0.0713, 0.0107] | 3/16 |
| causal_core_advantage_contrast_mpc | 0.0044 | [-0.0106, 0.0233] | 11/16 |
| causal_core_advantage_contrast_regularized | 0.0015 | [0.0000, 0.0059] | 0/16 |
| causal_core_advantage_contrast_guard | -0.0132 | [-0.0380, 0.0018] | 5/16 |
| causal_balanced_advantage_contrast_mpc | -0.0037 | [-0.0159, 0.0108] | 11/16 |
| causal_balanced_advantage_contrast_regularized | -0.0239 | [-0.0722, 0.0000] | 1/16 |
| causal_balanced_advantage_contrast_guard | -0.0239 | [-0.0722, 0.0000] | 1/16 |
| causal_rigid_advantage_contrast_mpc | -0.0110 | [-0.0244, 0.0025] | 11/16 |
| causal_rigid_advantage_contrast_regularized | 0.0009 | [0.0000, 0.0029] | 0/16 |
| causal_rigid_advantage_contrast_guard | 0.0026 | [-0.0040, 0.0132] | 3/16 |
| cfcmt_mechanism_contrast_mpc | -0.0206 | [-0.0504, -0.0007] | 13/16 |
| cfcmt_mechanism_contrast_guard | 0.0026 | [-0.0040, 0.0132] | 3/16 |
| causal_target_adapter_contrast_guard | 0.0019 | [-0.0009, 0.0061] | 0/16 |
| causal_target_only_contrast_guard | 0.0029 | [-0.0018, 0.0104] | 2/16 |
