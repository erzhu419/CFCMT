# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 8 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_core_advantage_contrast_mpc | 4.8683 | 0.9120 | 1.3020 | 0.5089 |
| dense_balanced_advantage_contrast_mpc | 5.0040 | 0.9229 | 1.2955 | 0.4989 |
| causal_rigid_advantage_contrast_mpc | 5.0634 | 0.9570 | 1.3564 | 0.4950 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| dense_balanced_advantage_contrast_guard | 5.0726 | 0.9163 | 1.2491 | 0.5040 |
| dense_balanced_advantage_contrast_regularized | 5.0767 | 0.9140 | 1.2406 | 0.5039 |
| causal_core_advantage_contrast_guard | 5.0770 | 0.9139 | 1.2456 | 0.5039 |
| simulator_contrast_guard | 5.0798 | 0.9131 | 1.2393 | 0.5037 |
| causal_rigid_advantage_contrast_guard | 5.0806 | 0.9143 | 1.2410 | 0.5033 |
| cfcmt_mechanism_contrast_guard | 5.0806 | 0.9143 | 1.2410 | 0.5033 |
| causal_target_adapter_contrast_guard | 5.0815 | 0.9149 | 1.2434 | 0.5038 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_rigid_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| cfcmt_mechanism_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_target_only_contrast_guard | 5.0816 | 0.9151 | 1.2439 | 0.5038 |
| causal_core_advantage_contrast_regularized | 5.0833 | 0.9147 | 1.2406 | 0.5037 |
| dense_contrast_guard | 5.0869 | 0.9204 | 1.2537 | 0.5045 |
| max_pressure | 5.0872 | 0.9305 | 1.2768 | 0.5038 |
| phase_pressure | 5.1061 | 0.9101 | 1.2048 | 0.4917 |
| causal_balanced_advantage_contrast_mpc | 5.1062 | 0.9438 | 1.3300 | 0.4940 |
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
| simulator_contrast_guard | 0.0004 | [-0.0002, 0.0024] | 2/16 |
| dense_contrast_guard | -0.0006 | [-0.0050, 0.0024] | 2/16 |
| dense_balanced_advantage_contrast_mpc | -0.0031 | [-0.0205, 0.0128] | 9/16 |
| dense_balanced_advantage_contrast_regularized | 0.0019 | [0.0000, 0.0086] | 0/16 |
| dense_balanced_advantage_contrast_guard | 0.0025 | [-0.0014, 0.0100] | 2/16 |
| causal_core_advantage_contrast_mpc | 0.0042 | [-0.0189, 0.0315] | 9/16 |
| causal_core_advantage_contrast_regularized | -0.0006 | [-0.0043, 0.0017] | 1/16 |
| causal_core_advantage_contrast_guard | 0.0008 | [-0.0032, 0.0058] | 1/16 |
| causal_balanced_advantage_contrast_mpc | -0.0131 | [-0.0240, -0.0041] | 12/16 |
| causal_balanced_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_balanced_advantage_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_mpc | -0.0164 | [-0.0374, 0.0034] | 12/16 |
| causal_rigid_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_guard | -0.0002 | [-0.0014, 0.0018] | 2/16 |
| cfcmt_mechanism_contrast_mpc | -0.0117 | [-0.0419, 0.0108] | 11/16 |
| cfcmt_mechanism_contrast_guard | -0.0002 | [-0.0014, 0.0018] | 2/16 |
| causal_target_adapter_contrast_guard | 0.0004 | [-0.0001, 0.0013] | 1/16 |
| causal_target_only_contrast_guard | -0.0000 | [-0.0003, 0.0003] | 1/16 |
