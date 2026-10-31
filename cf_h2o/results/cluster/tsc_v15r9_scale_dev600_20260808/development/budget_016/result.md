# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 16 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| dense_balanced_advantage_contrast_guard | 4.5711 | 0.8854 | 1.2577 | 0.5275 |
| dense_balanced_advantage_contrast_mpc | 4.5778 | 0.9398 | 1.3742 | 0.5240 |
| dense_balanced_advantage_contrast_regularized | 4.6421 | 0.8667 | 1.2421 | 0.5237 |
| causal_core_advantage_contrast_regularized | 4.6458 | 0.8823 | 1.2550 | 0.5263 |
| causal_core_advantage_contrast_mpc | 4.7631 | 0.9236 | 1.3324 | 0.5151 |
| causal_core_advantage_contrast_guard | 4.7987 | 0.8923 | 1.2558 | 0.5181 |
| causal_target_only_contrast_guard | 4.7992 | 0.8933 | 1.2902 | 0.5175 |
| causal_balanced_advantage_contrast_mpc | 4.8087 | 0.9271 | 1.3183 | 0.5108 |
| cfcmt_mechanism_contrast_mpc | 4.9130 | 0.9120 | 1.2782 | 0.5065 |
| causal_rigid_advantage_contrast_guard | 4.9927 | 0.9111 | 1.2508 | 0.5075 |
| cfcmt_mechanism_contrast_guard | 4.9927 | 0.9111 | 1.2508 | 0.5075 |
| causal_rigid_advantage_contrast_regularized | 5.0036 | 0.8990 | 1.2296 | 0.5063 |
| cfcmt_mechanism_contrast_regularized | 5.0036 | 0.8990 | 1.2296 | 0.5063 |
| causal_rigid_advantage_contrast_mpc | 5.0070 | 0.9488 | 1.3383 | 0.5030 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0742 | 0.9143 | 1.2423 | 0.5032 |
| dense_contrast_guard | 5.0806 | 0.9153 | 1.2464 | 0.5014 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_target_adapter_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
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
| fixed_program | -0.1194 | [-0.2360, 0.0149] | 15/16 |
| max_pressure | -0.0020 | [-0.0118, 0.0090] | 8/16 |
| phase_pressure | -0.0165 | [-0.0494, 0.0083] | 7/16 |
| spillback_pressure | -0.0393 | [-0.0928, -0.0076] | 14/16 |
| selected_source_prior | -0.0026 | [-0.0110, 0.0031] | 1/16 |
| simulator_contrast_guard | -0.0015 | [-0.0108, 0.0070] | 4/16 |
| dense_contrast_guard | -0.0066 | [-0.0206, 0.0037] | 4/16 |
| dense_balanced_advantage_contrast_mpc | 0.0225 | [-0.0140, 0.0775] | 8/16 |
| dense_balanced_advantage_contrast_regularized | 0.0215 | [0.0000, 0.0620] | 1/16 |
| dense_balanced_advantage_contrast_guard | 0.0260 | [0.0005, 0.0747] | 0/16 |
| causal_core_advantage_contrast_mpc | 0.0086 | [-0.0167, 0.0398] | 8/16 |
| causal_core_advantage_contrast_regularized | 0.0207 | [-0.0001, 0.0612] | 1/16 |
| causal_core_advantage_contrast_guard | 0.0122 | [0.0000, 0.0340] | 2/16 |
| causal_balanced_advantage_contrast_mpc | 0.0003 | [-0.0234, 0.0304] | 12/16 |
| causal_balanced_advantage_contrast_regularized | -0.0026 | [-0.0110, 0.0031] | 1/16 |
| causal_balanced_advantage_contrast_guard | -0.0026 | [-0.0110, 0.0031] | 1/16 |
| causal_rigid_advantage_contrast_mpc | -0.0116 | [-0.0263, 0.0032] | 12/16 |
| causal_rigid_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_guard | 0.0017 | [-0.0005, 0.0044] | 0/16 |
| cfcmt_mechanism_contrast_mpc | -0.0048 | [-0.0277, 0.0164] | 9/16 |
| cfcmt_mechanism_contrast_guard | 0.0017 | [-0.0005, 0.0044] | 0/16 |
| causal_target_adapter_contrast_guard | -0.0026 | [-0.0110, 0.0031] | 1/16 |
| causal_target_only_contrast_guard | 0.0121 | [0.0000, 0.0335] | 0/16 |
