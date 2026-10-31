# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 8 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| cfcmt_mechanism_contrast_mpc | 4.9271 | 0.9160 | 1.2770 | 0.5063 |
| causal_rigid_advantage_contrast_mpc | 5.0324 | 0.9413 | 1.3121 | 0.4998 |
| dense_balanced_advantage_contrast_mpc | 5.0600 | 0.9382 | 1.3089 | 0.5005 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0798 | 0.9131 | 1.2393 | 0.5037 |
| dense_balanced_advantage_contrast_guard | 5.0803 | 0.9137 | 1.2408 | 0.5034 |
| causal_rigid_advantage_contrast_guard | 5.0808 | 0.9149 | 1.2406 | 0.5035 |
| cfcmt_mechanism_contrast_guard | 5.0808 | 0.9149 | 1.2406 | 0.5035 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| dense_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_core_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_target_adapter_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_core_advantage_contrast_guard | 5.0816 | 0.9157 | 1.2442 | 0.5034 |
| causal_rigid_advantage_contrast_regularized | 5.0818 | 0.9157 | 1.2436 | 0.5036 |
| cfcmt_mechanism_contrast_regularized | 5.0818 | 0.9157 | 1.2436 | 0.5036 |
| causal_target_only_contrast_guard | 5.0820 | 0.9165 | 1.2448 | 0.5027 |
| dense_contrast_guard | 5.0869 | 0.9204 | 1.2537 | 0.5045 |
| max_pressure | 5.0872 | 0.9305 | 1.2768 | 0.5038 |
| causal_core_advantage_contrast_mpc | 5.1053 | 0.9260 | 1.2895 | 0.4975 |
| phase_pressure | 5.1061 | 0.9101 | 1.2048 | 0.4917 |
| causal_balanced_advantage_contrast_mpc | 5.1384 | 0.9623 | 1.3218 | 0.4972 |
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
| fixed_program | -0.1158 | [-0.2357, 0.0275] | 15/16 |
| max_pressure | 0.0007 | [-0.0041, 0.0093] | 8/16 |
| phase_pressure | -0.0138 | [-0.0463, 0.0094] | 7/16 |
| spillback_pressure | -0.0366 | [-0.0910, -0.0044] | 14/16 |
| selected_source_prior | 0.0001 | [-0.0002, 0.0007] | 1/16 |
| simulator_contrast_guard | 0.0005 | [-0.0003, 0.0028] | 2/16 |
| dense_contrast_guard | -0.0004 | [-0.0049, 0.0027] | 1/16 |
| dense_balanced_advantage_contrast_mpc | -0.0045 | [-0.0151, 0.0086] | 9/16 |
| dense_balanced_advantage_contrast_regularized | 0.0001 | [-0.0002, 0.0007] | 1/16 |
| dense_balanced_advantage_contrast_guard | 0.0005 | [-0.0003, 0.0026] | 1/16 |
| causal_core_advantage_contrast_mpc | -0.0083 | [-0.0210, 0.0018] | 11/16 |
| causal_core_advantage_contrast_regularized | 0.0001 | [-0.0002, 0.0007] | 1/16 |
| causal_core_advantage_contrast_guard | 0.0001 | [-0.0005, 0.0007] | 1/16 |
| causal_balanced_advantage_contrast_mpc | -0.0176 | [-0.0310, -0.0055] | 15/16 |
| causal_balanced_advantage_contrast_regularized | 0.0001 | [-0.0002, 0.0007] | 1/16 |
| causal_balanced_advantage_contrast_guard | 0.0001 | [-0.0002, 0.0007] | 1/16 |
| causal_rigid_advantage_contrast_mpc | -0.0105 | [-0.0286, 0.0076] | 12/16 |
| causal_rigid_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_guard | -0.0003 | [-0.0023, 0.0021] | 1/16 |
| cfcmt_mechanism_contrast_mpc | -0.0063 | [-0.0350, 0.0234] | 10/16 |
| cfcmt_mechanism_contrast_guard | -0.0003 | [-0.0023, 0.0021] | 1/16 |
| causal_target_adapter_contrast_guard | 0.0001 | [-0.0002, 0.0007] | 1/16 |
| causal_target_only_contrast_guard | -0.0005 | [-0.0015, 0.0000] | 2/16 |
