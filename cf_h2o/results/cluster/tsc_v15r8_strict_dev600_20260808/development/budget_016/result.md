# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 16 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_core_advantage_contrast_guard | 4.5434 | 0.8818 | 1.2526 | 0.5288 |
| causal_core_advantage_contrast_mpc | 4.5494 | 0.8880 | 1.2626 | 0.5257 |
| causal_core_advantage_contrast_regularized | 4.5540 | 0.8788 | 1.2507 | 0.5303 |
| causal_target_only_contrast_guard | 4.5750 | 0.8808 | 1.2636 | 0.5294 |
| causal_target_adapter_contrast_guard | 4.5938 | 0.8947 | 1.2779 | 0.5246 |
| causal_rigid_advantage_contrast_regularized | 4.6552 | 0.8957 | 1.2802 | 0.5236 |
| dense_balanced_advantage_contrast_guard | 4.7068 | 0.9042 | 1.2733 | 0.5178 |
| dense_balanced_advantage_contrast_mpc | 4.7424 | 0.9500 | 1.3714 | 0.5102 |
| causal_rigid_advantage_contrast_guard | 4.7540 | 0.9029 | 1.2980 | 0.5185 |
| dense_balanced_advantage_contrast_regularized | 4.7610 | 0.8928 | 1.2609 | 0.5192 |
| causal_rigid_advantage_contrast_mpc | 4.7957 | 0.9343 | 1.3359 | 0.5068 |
| causal_balanced_advantage_contrast_mpc | 4.8078 | 0.9220 | 1.3485 | 0.5085 |
| cfcmt_mechanism_contrast_guard | 4.8137 | 0.8972 | 1.2862 | 0.5180 |
| cfcmt_mechanism_contrast_regularized | 4.8715 | 0.9011 | 1.2684 | 0.5147 |
| cfcmt_mechanism_contrast_mpc | 4.9258 | 0.9648 | 1.3723 | 0.5022 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0742 | 0.9143 | 1.2423 | 0.5032 |
| dense_contrast_guard | 5.0806 | 0.9153 | 1.2464 | 0.5014 |
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
| fixed_program | -0.1284 | [-0.2371, -0.0064] | 15/16 |
| max_pressure | -0.0099 | [-0.0313, 0.0066] | 9/16 |
| phase_pressure | -0.0244 | [-0.0574, 0.0042] | 7/16 |
| spillback_pressure | -0.0472 | [-0.0984, -0.0138] | 15/16 |
| selected_source_prior | -0.0105 | [-0.0304, 0.0000] | 2/16 |
| simulator_contrast_guard | -0.0093 | [-0.0304, 0.0053] | 5/16 |
| dense_contrast_guard | -0.0144 | [-0.0345, 0.0003] | 5/16 |
| dense_balanced_advantage_contrast_mpc | -0.0030 | [-0.0531, 0.0336] | 11/16 |
| dense_balanced_advantage_contrast_regularized | 0.0053 | [-0.0061, 0.0194] | 2/16 |
| dense_balanced_advantage_contrast_guard | 0.0088 | [-0.0022, 0.0269] | 2/16 |
| causal_core_advantage_contrast_mpc | 0.0153 | [-0.0110, 0.0531] | 10/16 |
| causal_core_advantage_contrast_regularized | 0.0185 | [-0.0041, 0.0548] | 1/16 |
| causal_core_advantage_contrast_guard | 0.0190 | [-0.0045, 0.0563] | 3/16 |
| causal_balanced_advantage_contrast_mpc | -0.0046 | [-0.0196, 0.0110] | 11/16 |
| causal_balanced_advantage_contrast_regularized | -0.0105 | [-0.0304, 0.0000] | 2/16 |
| causal_balanced_advantage_contrast_guard | -0.0105 | [-0.0304, 0.0000] | 2/16 |
| causal_rigid_advantage_contrast_mpc | -0.0076 | [-0.0256, 0.0113] | 11/16 |
| causal_rigid_advantage_contrast_regularized | 0.0116 | [-0.0009, 0.0367] | 1/16 |
| causal_rigid_advantage_contrast_guard | 0.0057 | [-0.0025, 0.0196] | 4/16 |
| cfcmt_mechanism_contrast_mpc | -0.0108 | [-0.0324, 0.0082] | 12/16 |
| cfcmt_mechanism_contrast_guard | 0.0026 | [-0.0020, 0.0097] | 4/16 |
| causal_target_adapter_contrast_guard | 0.0149 | [-0.0026, 0.0479] | 3/16 |
| causal_target_only_contrast_guard | 0.0158 | [-0.0028, 0.0500] | 3/16 |
