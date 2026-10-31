# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 60 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| dense_balanced_advantage_contrast_mpc | 4.4257 | 0.9131 | 1.3540 | 0.5310 |
| causal_core_advantage_contrast_guard | 4.4766 | 0.8817 | 1.2660 | 0.5333 |
| causal_core_advantage_contrast_mpc | 4.4839 | 0.9024 | 1.3053 | 0.5287 |
| causal_core_advantage_contrast_regularized | 4.5445 | 0.8761 | 1.2626 | 0.5297 |
| causal_target_only_contrast_guard | 4.5450 | 0.8858 | 1.2942 | 0.5269 |
| causal_rigid_advantage_contrast_regularized | 4.5634 | 0.9061 | 1.3072 | 0.5240 |
| cfcmt_mechanism_contrast_regularized | 4.5648 | 0.9044 | 1.3022 | 0.5245 |
| causal_balanced_advantage_contrast_mpc | 4.5662 | 0.9092 | 1.3167 | 0.5214 |
| causal_target_adapter_contrast_guard | 4.5760 | 0.8966 | 1.3129 | 0.5250 |
| dense_balanced_advantage_contrast_regularized | 4.5793 | 0.8807 | 1.2584 | 0.5299 |
| causal_rigid_advantage_contrast_mpc | 4.6692 | 0.9088 | 1.3003 | 0.5141 |
| causal_rigid_advantage_contrast_guard | 4.6880 | 0.8918 | 1.2547 | 0.5205 |
| cfcmt_mechanism_contrast_guard | 4.6880 | 0.8918 | 1.2547 | 0.5205 |
| cfcmt_mechanism_contrast_mpc | 4.7617 | 0.9483 | 1.3638 | 0.5125 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0748 | 0.9137 | 1.2447 | 0.5027 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| dense_contrast_guard | 5.0830 | 0.9189 | 1.2612 | 0.5040 |
| max_pressure | 5.0872 | 0.9305 | 1.2768 | 0.5038 |
| dense_balanced_advantage_contrast_guard | 5.0967 | 0.9226 | 1.2536 | 0.5042 |
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
| fixed_program | -0.1462 | [-0.2376, -0.0547] | 15/16 |
| max_pressure | -0.0247 | [-0.0754, 0.0065] | 9/16 |
| phase_pressure | -0.0392 | [-0.0914, 0.0040] | 7/16 |
| spillback_pressure | -0.0620 | [-0.1192, -0.0138] | 15/16 |
| selected_source_prior | -0.0253 | [-0.0749, 0.0001] | 2/16 |
| simulator_contrast_guard | -0.0245 | [-0.0744, 0.0023] | 4/16 |
| dense_contrast_guard | -0.0253 | [-0.0748, 0.0016] | 2/16 |
| dense_balanced_advantage_contrast_mpc | 0.0042 | [-0.0142, 0.0250] | 11/16 |
| dense_balanced_advantage_contrast_regularized | -0.0013 | [-0.0063, 0.0024] | 2/16 |
| dense_balanced_advantage_contrast_guard | -0.0275 | [-0.0759, 0.0011] | 4/16 |
| causal_core_advantage_contrast_mpc | 0.0012 | [-0.0113, 0.0154] | 9/16 |
| causal_core_advantage_contrast_regularized | 0.0014 | [-0.0022, 0.0065] | 0/16 |
| causal_core_advantage_contrast_guard | 0.0062 | [-0.0014, 0.0164] | 1/16 |
| causal_balanced_advantage_contrast_mpc | -0.0047 | [-0.0196, 0.0123] | 10/16 |
| causal_balanced_advantage_contrast_regularized | -0.0253 | [-0.0749, 0.0001] | 2/16 |
| causal_balanced_advantage_contrast_guard | -0.0253 | [-0.0749, 0.0001] | 2/16 |
| causal_rigid_advantage_contrast_mpc | -0.0151 | [-0.0306, 0.0020] | 11/16 |
| causal_rigid_advantage_contrast_regularized | 0.0001 | [-0.0000, 0.0004] | 0/16 |
| causal_rigid_advantage_contrast_guard | -0.0041 | [-0.0216, 0.0102] | 1/16 |
| cfcmt_mechanism_contrast_mpc | -0.0215 | [-0.0536, 0.0020] | 11/16 |
| cfcmt_mechanism_contrast_guard | -0.0041 | [-0.0216, 0.0102] | 1/16 |
| causal_target_adapter_contrast_guard | -0.0014 | [-0.0055, 0.0015] | 4/16 |
| causal_target_only_contrast_guard | 0.0043 | [-0.0024, 0.0125] | 3/16 |
