# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 32 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_core_advantage_contrast_mpc | 4.5218 | 0.9037 | 1.3168 | 0.5259 |
| causal_core_advantage_contrast_guard | 4.5251 | 0.8831 | 1.2710 | 0.5305 |
| causal_balanced_advantage_contrast_mpc | 4.5401 | 0.9061 | 1.3143 | 0.5213 |
| causal_target_adapter_contrast_guard | 4.5768 | 0.8840 | 1.2902 | 0.5229 |
| causal_target_only_contrast_guard | 4.6049 | 0.8909 | 1.2750 | 0.5239 |
| dense_balanced_advantage_contrast_guard | 4.6131 | 0.9028 | 1.2951 | 0.5228 |
| causal_rigid_advantage_contrast_mpc | 4.6443 | 0.9163 | 1.3243 | 0.5144 |
| dense_balanced_advantage_contrast_mpc | 4.6585 | 0.9399 | 1.3660 | 0.5183 |
| causal_rigid_advantage_contrast_guard | 4.7257 | 0.9094 | 1.2925 | 0.5197 |
| cfcmt_mechanism_contrast_guard | 4.7257 | 0.9098 | 1.2925 | 0.5195 |
| cfcmt_mechanism_contrast_mpc | 4.7824 | 0.9504 | 1.3558 | 0.5103 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| cfcmt_mechanism_contrast_regularized | 5.0728 | 0.9103 | 1.2350 | 0.5047 |
| dense_balanced_advantage_contrast_regularized | 5.0739 | 0.9145 | 1.2400 | 0.5046 |
| simulator_contrast_guard | 5.0742 | 0.9140 | 1.2418 | 0.5042 |
| causal_rigid_advantage_contrast_regularized | 5.0774 | 0.9164 | 1.2431 | 0.5043 |
| causal_core_advantage_contrast_regularized | 5.0790 | 0.9145 | 1.2433 | 0.5040 |
| dense_contrast_guard | 5.0797 | 0.9149 | 1.2469 | 0.5037 |
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
| fixed_program | -0.1180 | [-0.2391, 0.0257] | 15/16 |
| max_pressure | -0.0019 | [-0.0109, 0.0085] | 10/16 |
| phase_pressure | -0.0165 | [-0.0480, 0.0046] | 7/16 |
| spillback_pressure | -0.0392 | [-0.0931, -0.0066] | 15/16 |
| selected_source_prior | -0.0024 | [-0.0089, 0.0010] | 2/16 |
| simulator_contrast_guard | -0.0012 | [-0.0083, 0.0059] | 4/16 |
| dense_contrast_guard | -0.0020 | [-0.0082, 0.0012] | 4/16 |
| dense_balanced_advantage_contrast_mpc | 0.0195 | [-0.0129, 0.0687] | 12/16 |
| dense_balanced_advantage_contrast_regularized | 0.0001 | [-0.0047, 0.0049] | 1/16 |
| dense_balanced_advantage_contrast_guard | 0.0225 | [-0.0102, 0.0780] | 4/16 |
| causal_core_advantage_contrast_mpc | 0.0282 | [-0.0101, 0.0932] | 10/16 |
| causal_core_advantage_contrast_regularized | -0.0016 | [-0.0065, 0.0012] | 2/16 |
| causal_core_advantage_contrast_guard | 0.0304 | [-0.0054, 0.0932] | 5/16 |
| causal_balanced_advantage_contrast_mpc | 0.0215 | [-0.0192, 0.0889] | 12/16 |
| causal_balanced_advantage_contrast_regularized | -0.0024 | [-0.0089, 0.0010] | 2/16 |
| causal_balanced_advantage_contrast_guard | -0.0024 | [-0.0089, 0.0010] | 2/16 |
| causal_rigid_advantage_contrast_mpc | 0.0128 | [-0.0253, 0.0674] | 10/16 |
| causal_rigid_advantage_contrast_regularized | -0.0011 | [-0.0048, 0.0000] | 1/16 |
| causal_rigid_advantage_contrast_guard | 0.0150 | [-0.0141, 0.0591] | 4/16 |
| cfcmt_mechanism_contrast_mpc | 0.0013 | [-0.0456, 0.0540] | 12/16 |
| cfcmt_mechanism_contrast_guard | 0.0150 | [-0.0141, 0.0591] | 4/16 |
| causal_target_adapter_contrast_guard | 0.0259 | [-0.0073, 0.0845] | 4/16 |
| causal_target_only_contrast_guard | 0.0234 | [-0.0092, 0.0798] | 5/16 |
