# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 32 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| dense_balanced_advantage_contrast_mpc | 4.5644 | 0.9240 | 1.3478 | 0.5229 |
| causal_core_advantage_contrast_guard | 4.6841 | 0.8821 | 1.2461 | 0.5231 |
| causal_core_advantage_contrast_mpc | 4.6936 | 0.8809 | 1.2745 | 0.5178 |
| causal_target_only_contrast_guard | 4.7808 | 0.8980 | 1.2864 | 0.5155 |
| causal_balanced_advantage_contrast_mpc | 4.8666 | 0.9335 | 1.2957 | 0.5077 |
| cfcmt_mechanism_contrast_mpc | 4.9095 | 0.9061 | 1.2675 | 0.5103 |
| causal_rigid_advantage_contrast_mpc | 4.9726 | 0.9330 | 1.3250 | 0.5082 |
| cfcmt_mechanism_contrast_guard | 4.9773 | 0.9276 | 1.2940 | 0.5083 |
| causal_rigid_advantage_contrast_guard | 4.9777 | 0.9278 | 1.2949 | 0.5076 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0742 | 0.9140 | 1.2418 | 0.5042 |
| causal_target_adapter_contrast_guard | 5.0777 | 0.9147 | 1.2423 | 0.5041 |
| dense_balanced_advantage_contrast_regularized | 5.0778 | 0.9109 | 1.2354 | 0.5047 |
| dense_contrast_guard | 5.0800 | 0.9150 | 1.2468 | 0.5032 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_core_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_rigid_advantage_contrast_regularized | 5.0817 | 0.9154 | 1.2438 | 0.5038 |
| cfcmt_mechanism_contrast_regularized | 5.0817 | 0.9154 | 1.2438 | 0.5038 |
| dense_balanced_advantage_contrast_guard | 5.0865 | 0.9146 | 1.2377 | 0.5061 |
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
| fixed_program | -0.1158 | [-0.2358, 0.0275] | 15/16 |
| max_pressure | 0.0007 | [-0.0042, 0.0093] | 9/16 |
| phase_pressure | -0.0138 | [-0.0464, 0.0093] | 7/16 |
| spillback_pressure | -0.0366 | [-0.0910, -0.0045] | 15/16 |
| selected_source_prior | 0.0001 | [0.0000, 0.0005] | 0/16 |
| simulator_contrast_guard | 0.0013 | [-0.0011, 0.0073] | 2/16 |
| dense_contrast_guard | 0.0004 | [-0.0005, 0.0025] | 2/16 |
| dense_balanced_advantage_contrast_mpc | 0.0296 | [-0.0062, 0.0885] | 7/16 |
| dense_balanced_advantage_contrast_regularized | 0.0014 | [-0.0002, 0.0057] | 1/16 |
| dense_balanced_advantage_contrast_guard | -0.0002 | [-0.0063, 0.0054] | 3/16 |
| causal_core_advantage_contrast_mpc | 0.0188 | [-0.0120, 0.0644] | 10/16 |
| causal_core_advantage_contrast_regularized | 0.0001 | [0.0000, 0.0005] | 0/16 |
| causal_core_advantage_contrast_guard | 0.0227 | [-0.0021, 0.0663] | 1/16 |
| causal_balanced_advantage_contrast_mpc | 0.0039 | [-0.0231, 0.0375] | 12/16 |
| causal_balanced_advantage_contrast_regularized | 0.0001 | [0.0000, 0.0005] | 0/16 |
| causal_balanced_advantage_contrast_guard | 0.0001 | [0.0000, 0.0005] | 0/16 |
| causal_rigid_advantage_contrast_mpc | -0.0011 | [-0.0182, 0.0198] | 11/16 |
| causal_rigid_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_guard | 0.0006 | [-0.0144, 0.0161] | 4/16 |
| cfcmt_mechanism_contrast_mpc | 0.0005 | [-0.0212, 0.0259] | 11/16 |
| cfcmt_mechanism_contrast_guard | 0.0007 | [-0.0141, 0.0159] | 5/16 |
| causal_target_adapter_contrast_guard | 0.0014 | [0.0000, 0.0058] | 0/16 |
| causal_target_only_contrast_guard | 0.0171 | [-0.0005, 0.0480] | 2/16 |
