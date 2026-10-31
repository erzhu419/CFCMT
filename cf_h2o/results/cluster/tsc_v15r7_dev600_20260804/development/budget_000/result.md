# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold uses static summaries and online state only; no target transition labels are used.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_balanced_advantage_contrast_mpc | 4.8490 | 0.9355 | 1.3374 | 0.5060 |
| causal_core_advantage_contrast_mpc | 4.8775 | 0.9107 | 1.2925 | 0.5087 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| dense_contrast_guard | 5.0785 | 0.9110 | 1.2349 | 0.5029 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| dense_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_core_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_core_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_rigid_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_rigid_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| cfcmt_mechanism_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| cfcmt_mechanism_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_target_adapter_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_target_only_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| dense_balanced_advantage_contrast_mpc | 5.0865 | 0.9526 | 1.3283 | 0.5040 |
| max_pressure | 5.0872 | 0.9305 | 1.2768 | 0.5038 |
| causal_rigid_advantage_contrast_mpc | 5.0943 | 0.9615 | 1.3080 | 0.4971 |
| simulator_contrast_guard | 5.1017 | 0.9411 | 1.2984 | 0.4995 |
| phase_pressure | 5.1061 | 0.9101 | 1.2048 | 0.4917 |
| dense_balanced_advantage_contrast_guard | 5.1112 | 0.9285 | 1.2598 | 0.5028 |
| cfcmt_mechanism_contrast_mpc | 5.1180 | 0.9782 | 1.3310 | 0.4961 |
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
| fixed_program | -0.1159 | [-0.2361, 0.0275] | 15/16 |
| max_pressure | 0.0006 | [-0.0043, 0.0092] | 9/16 |
| phase_pressure | -0.0139 | [-0.0464, 0.0091] | 7/16 |
| spillback_pressure | -0.0367 | [-0.0911, -0.0048] | 15/16 |
| selected_source_prior | 0.0000 | [0.0000, 0.0000] | 0/16 |
| simulator_contrast_guard | -0.0120 | [-0.0295, 0.0025] | 11/16 |
| dense_contrast_guard | 0.0005 | [-0.0011, 0.0038] | 4/16 |
| dense_balanced_advantage_contrast_mpc | -0.0068 | [-0.0189, 0.0020] | 12/16 |
| dense_balanced_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| dense_balanced_advantage_contrast_guard | -0.0048 | [-0.0210, 0.0013] | 2/16 |
| causal_core_advantage_contrast_mpc | -0.0003 | [-0.0249, 0.0311] | 11/16 |
| causal_core_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_core_advantage_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_balanced_advantage_contrast_mpc | 0.0031 | [-0.0233, 0.0367] | 12/16 |
| causal_balanced_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_balanced_advantage_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_mpc | -0.0121 | [-0.0309, 0.0064] | 12/16 |
| causal_rigid_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| cfcmt_mechanism_contrast_mpc | -0.0117 | [-0.0419, 0.0108] | 11/16 |
| cfcmt_mechanism_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_target_adapter_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_target_only_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
