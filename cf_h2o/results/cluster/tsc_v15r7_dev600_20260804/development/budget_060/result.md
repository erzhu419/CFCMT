# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 60 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| dense_balanced_advantage_contrast_mpc | 4.4135 | 0.9075 | 1.3619 | 0.5310 |
| causal_core_advantage_contrast_mpc | 4.4639 | 0.8931 | 1.3126 | 0.5296 |
| causal_core_advantage_contrast_guard | 4.4848 | 0.8811 | 1.2673 | 0.5324 |
| causal_core_advantage_contrast_regularized | 4.5078 | 0.8840 | 1.2694 | 0.5308 |
| causal_balanced_advantage_contrast_mpc | 4.5280 | 0.8968 | 1.2766 | 0.5257 |
| dense_balanced_advantage_contrast_regularized | 4.5633 | 0.8839 | 1.2702 | 0.5294 |
| cfcmt_mechanism_contrast_regularized | 4.5754 | 0.9032 | 1.3016 | 0.5238 |
| causal_rigid_advantage_contrast_regularized | 4.5757 | 0.9045 | 1.3032 | 0.5234 |
| causal_rigid_advantage_contrast_mpc | 4.6369 | 0.9106 | 1.3047 | 0.5195 |
| causal_target_only_contrast_guard | 4.6582 | 0.8709 | 1.2519 | 0.5225 |
| causal_rigid_advantage_contrast_guard | 4.6868 | 0.8969 | 1.2705 | 0.5209 |
| cfcmt_mechanism_contrast_guard | 4.6868 | 0.8969 | 1.2705 | 0.5209 |
| causal_target_adapter_contrast_guard | 4.6869 | 0.8765 | 1.2466 | 0.5223 |
| cfcmt_mechanism_contrast_mpc | 4.7870 | 0.9531 | 1.3585 | 0.5099 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0748 | 0.9137 | 1.2440 | 0.5025 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| dense_contrast_guard | 5.0831 | 0.9190 | 1.2613 | 0.5038 |
| max_pressure | 5.0872 | 0.9305 | 1.2768 | 0.5038 |
| dense_balanced_advantage_contrast_guard | 5.0890 | 0.9182 | 1.2516 | 0.5039 |
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
| fixed_program | -0.1457 | [-0.2378, -0.0533] | 15/16 |
| max_pressure | -0.0244 | [-0.0738, 0.0063] | 9/16 |
| phase_pressure | -0.0389 | [-0.0897, 0.0037] | 7/16 |
| spillback_pressure | -0.0616 | [-0.1184, -0.0141] | 15/16 |
| selected_source_prior | -0.0250 | [-0.0730, 0.0000] | 2/16 |
| simulator_contrast_guard | -0.0241 | [-0.0725, 0.0023] | 4/16 |
| dense_contrast_guard | -0.0251 | [-0.0731, 0.0014] | 3/16 |
| dense_balanced_advantage_contrast_mpc | 0.0080 | [-0.0094, 0.0295] | 10/16 |
| dense_balanced_advantage_contrast_regularized | 0.0000 | [-0.0039, 0.0038] | 1/16 |
| dense_balanced_advantage_contrast_guard | -0.0260 | [-0.0736, 0.0008] | 6/16 |
| causal_core_advantage_contrast_mpc | 0.0066 | [-0.0046, 0.0202] | 9/16 |
| causal_core_advantage_contrast_regularized | 0.0046 | [0.0000, 0.0125] | 0/16 |
| causal_core_advantage_contrast_guard | 0.0062 | [-0.0016, 0.0165] | 1/16 |
| causal_balanced_advantage_contrast_mpc | -0.0031 | [-0.0162, 0.0123] | 11/16 |
| causal_balanced_advantage_contrast_regularized | -0.0250 | [-0.0730, 0.0000] | 2/16 |
| causal_balanced_advantage_contrast_guard | -0.0250 | [-0.0730, 0.0000] | 2/16 |
| causal_rigid_advantage_contrast_mpc | -0.0079 | [-0.0235, 0.0090] | 10/16 |
| causal_rigid_advantage_contrast_regularized | -0.0000 | [-0.0001, 0.0000] | 1/16 |
| causal_rigid_advantage_contrast_guard | -0.0052 | [-0.0189, 0.0043] | 3/16 |
| cfcmt_mechanism_contrast_mpc | -0.0237 | [-0.0560, 0.0002] | 13/16 |
| cfcmt_mechanism_contrast_guard | -0.0052 | [-0.0189, 0.0043] | 3/16 |
| causal_target_adapter_contrast_guard | -0.0066 | [-0.0188, 0.0005] | 4/16 |
| causal_target_only_contrast_guard | -0.0024 | [-0.0152, 0.0108] | 5/16 |
