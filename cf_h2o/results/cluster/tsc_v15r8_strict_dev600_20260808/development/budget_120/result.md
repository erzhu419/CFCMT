# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 120 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_core_advantage_contrast_mpc | 4.4471 | 0.8953 | 1.3178 | 0.5320 |
| causal_target_adapter_contrast_guard | 4.5153 | 0.8796 | 1.2733 | 0.5311 |
| causal_core_advantage_contrast_regularized | 4.5389 | 0.8843 | 1.2736 | 0.5303 |
| causal_rigid_advantage_contrast_guard | 4.5486 | 0.9017 | 1.2951 | 0.5263 |
| cfcmt_mechanism_contrast_guard | 4.5486 | 0.9017 | 1.2951 | 0.5263 |
| cfcmt_mechanism_contrast_regularized | 4.5652 | 0.9042 | 1.2988 | 0.5253 |
| causal_rigid_advantage_contrast_regularized | 4.5696 | 0.9046 | 1.2954 | 0.5245 |
| dense_balanced_advantage_contrast_mpc | 4.5777 | 0.9256 | 1.3388 | 0.5195 |
| causal_target_only_contrast_guard | 4.6061 | 0.8874 | 1.2855 | 0.5290 |
| causal_rigid_advantage_contrast_mpc | 4.6460 | 0.9157 | 1.3198 | 0.5167 |
| dense_balanced_advantage_contrast_regularized | 4.6514 | 0.8963 | 1.2641 | 0.5225 |
| causal_balanced_advantage_contrast_mpc | 4.6609 | 0.9013 | 1.2963 | 0.5184 |
| cfcmt_mechanism_contrast_mpc | 4.7840 | 0.9514 | 1.3530 | 0.5095 |
| causal_core_advantage_contrast_guard | 4.8100 | 0.8988 | 1.2464 | 0.5164 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| dense_balanced_advantage_contrast_guard | 5.0709 | 0.9145 | 1.2396 | 0.5050 |
| simulator_contrast_guard | 5.0742 | 0.9141 | 1.2440 | 0.5041 |
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
| fixed_program | -0.1459 | [-0.2361, -0.0552] | 15/16 |
| max_pressure | -0.0242 | [-0.0756, 0.0069] | 9/16 |
| phase_pressure | -0.0387 | [-0.0912, 0.0052] | 7/16 |
| spillback_pressure | -0.0615 | [-0.1196, -0.0129] | 15/16 |
| selected_source_prior | -0.0248 | [-0.0750, 0.0000] | 1/16 |
| simulator_contrast_guard | -0.0235 | [-0.0744, 0.0056] | 5/16 |
| dense_contrast_guard | -0.0236 | [-0.0743, 0.0033] | 2/16 |
| dense_balanced_advantage_contrast_mpc | -0.0007 | [-0.0130, 0.0175] | 11/16 |
| dense_balanced_advantage_contrast_regularized | -0.0046 | [-0.0147, 0.0013] | 1/16 |
| dense_balanced_advantage_contrast_guard | -0.0219 | [-0.0738, 0.0081] | 2/16 |
| causal_core_advantage_contrast_mpc | 0.0061 | [-0.0066, 0.0223] | 7/16 |
| causal_core_advantage_contrast_regularized | 0.0015 | [-0.0001, 0.0059] | 0/16 |
| causal_core_advantage_contrast_guard | -0.0115 | [-0.0391, 0.0054] | 4/16 |
| causal_balanced_advantage_contrast_mpc | -0.0096 | [-0.0258, 0.0095] | 12/16 |
| causal_balanced_advantage_contrast_regularized | -0.0248 | [-0.0750, 0.0000] | 1/16 |
| causal_balanced_advantage_contrast_guard | -0.0248 | [-0.0750, 0.0000] | 1/16 |
| causal_rigid_advantage_contrast_mpc | -0.0117 | [-0.0248, 0.0014] | 11/16 |
| causal_rigid_advantage_contrast_regularized | -0.0002 | [-0.0015, 0.0007] | 1/16 |
| causal_rigid_advantage_contrast_guard | 0.0021 | [-0.0009, 0.0072] | 2/16 |
| cfcmt_mechanism_contrast_mpc | -0.0245 | [-0.0529, -0.0030] | 13/16 |
| cfcmt_mechanism_contrast_guard | 0.0021 | [-0.0009, 0.0072] | 2/16 |
| causal_target_adapter_contrast_guard | 0.0038 | [0.0000, 0.0109] | 0/16 |
| causal_target_only_contrast_guard | -0.0010 | [-0.0092, 0.0045] | 3/16 |
