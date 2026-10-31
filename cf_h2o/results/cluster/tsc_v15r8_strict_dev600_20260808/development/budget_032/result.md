# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 32 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_core_advantage_contrast_guard | 4.5483 | 0.9007 | 1.3174 | 0.5290 |
| causal_target_only_contrast_guard | 4.5733 | 0.8711 | 1.2692 | 0.5263 |
| causal_core_advantage_contrast_mpc | 4.5847 | 0.9232 | 1.3402 | 0.5204 |
| causal_target_adapter_contrast_guard | 4.6230 | 0.8896 | 1.2939 | 0.5218 |
| dense_balanced_advantage_contrast_mpc | 4.6344 | 0.9404 | 1.3556 | 0.5142 |
| dense_balanced_advantage_contrast_guard | 4.6424 | 0.9046 | 1.2966 | 0.5225 |
| causal_balanced_advantage_contrast_mpc | 4.6557 | 0.9157 | 1.3327 | 0.5133 |
| causal_rigid_advantage_contrast_mpc | 4.6764 | 0.9104 | 1.3268 | 0.5171 |
| cfcmt_mechanism_contrast_guard | 4.7924 | 0.9048 | 1.2809 | 0.5179 |
| causal_rigid_advantage_contrast_guard | 4.7931 | 0.9051 | 1.2821 | 0.5181 |
| cfcmt_mechanism_contrast_mpc | 4.8235 | 0.9407 | 1.3371 | 0.5088 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0742 | 0.9140 | 1.2418 | 0.5042 |
| dense_balanced_advantage_contrast_regularized | 5.0743 | 0.9155 | 1.2423 | 0.5047 |
| dense_contrast_guard | 5.0800 | 0.9150 | 1.2468 | 0.5032 |
| causal_core_advantage_contrast_regularized | 5.0804 | 0.9147 | 1.2433 | 0.5039 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| cfcmt_mechanism_contrast_regularized | 5.0828 | 0.9129 | 1.2363 | 0.5046 |
| causal_rigid_advantage_contrast_regularized | 5.0864 | 0.9176 | 1.2433 | 0.5039 |
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
| fixed_program | -0.1154 | [-0.2372, 0.0275] | 15/16 |
| max_pressure | 0.0013 | [-0.0066, 0.0128] | 9/16 |
| phase_pressure | -0.0133 | [-0.0462, 0.0108] | 7/16 |
| spillback_pressure | -0.0362 | [-0.0910, -0.0043] | 15/16 |
| selected_source_prior | 0.0007 | [-0.0047, 0.0101] | 1/16 |
| simulator_contrast_guard | 0.0019 | [-0.0046, 0.0137] | 3/16 |
| dense_contrast_guard | 0.0010 | [-0.0043, 0.0106] | 4/16 |
| dense_balanced_advantage_contrast_mpc | 0.0221 | [-0.0127, 0.0737] | 10/16 |
| dense_balanced_advantage_contrast_regularized | 0.0033 | [-0.0038, 0.0192] | 1/16 |
| dense_balanced_advantage_contrast_guard | 0.0241 | [-0.0046, 0.0731] | 4/16 |
| causal_core_advantage_contrast_mpc | 0.0226 | [-0.0154, 0.0830] | 13/16 |
| causal_core_advantage_contrast_regularized | 0.0010 | [-0.0035, 0.0102] | 1/16 |
| causal_core_advantage_contrast_guard | 0.0304 | [-0.0056, 0.0927] | 6/16 |
| causal_balanced_advantage_contrast_mpc | 0.0156 | [-0.0198, 0.0679] | 11/16 |
| causal_balanced_advantage_contrast_regularized | 0.0007 | [-0.0047, 0.0101] | 1/16 |
| causal_balanced_advantage_contrast_guard | 0.0007 | [-0.0047, 0.0101] | 1/16 |
| causal_rigid_advantage_contrast_mpc | 0.0161 | [-0.0161, 0.0648] | 12/16 |
| causal_rigid_advantage_contrast_regularized | -0.0009 | [-0.0038, 0.0000] | 1/16 |
| causal_rigid_advantage_contrast_guard | 0.0144 | [-0.0087, 0.0477] | 3/16 |
| cfcmt_mechanism_contrast_mpc | 0.0035 | [-0.0373, 0.0473] | 11/16 |
| cfcmt_mechanism_contrast_guard | 0.0146 | [-0.0085, 0.0477] | 3/16 |
| causal_target_adapter_contrast_guard | 0.0250 | [-0.0050, 0.0764] | 3/16 |
| causal_target_only_contrast_guard | 0.0304 | [-0.0013, 0.0858] | 1/16 |
