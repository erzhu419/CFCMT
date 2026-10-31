# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold uses static summaries and online state only; no target transition labels are used.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| causal_rigid_advantage_contrast_mpc | 4.9271 | 0.9160 | 1.2770 | 0.5063 |
| cfcmt_mechanism_contrast_mpc | 4.9271 | 0.9160 | 1.2770 | 0.5063 |
| dense_balanced_advantage_contrast_mpc | 5.0564 | 0.9226 | 1.2696 | 0.5007 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0766 | 0.9133 | 1.2379 | 0.5044 |
| dense_balanced_advantage_contrast_guard | 5.0781 | 0.9164 | 1.2502 | 0.5054 |
| dense_contrast_guard | 5.0784 | 0.9108 | 1.2343 | 0.5030 |
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
| max_pressure | 5.0872 | 0.9305 | 1.2768 | 0.5038 |
| causal_core_advantage_contrast_mpc | 5.0950 | 0.9289 | 1.2856 | 0.4991 |
| phase_pressure | 5.1061 | 0.9101 | 1.2048 | 0.4917 |
| causal_balanced_advantage_contrast_mpc | 5.1102 | 0.9408 | 1.3101 | 0.4981 |
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
| simulator_contrast_guard | 0.0016 | [-0.0004, 0.0055] | 3/16 |
| dense_contrast_guard | 0.0007 | [-0.0009, 0.0039] | 3/16 |
| dense_balanced_advantage_contrast_mpc | -0.0047 | [-0.0206, 0.0094] | 9/16 |
| dense_balanced_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| dense_balanced_advantage_contrast_guard | 0.0014 | [-0.0012, 0.0059] | 0/16 |
| causal_core_advantage_contrast_mpc | -0.0077 | [-0.0166, 0.0010] | 12/16 |
| causal_core_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_core_advantage_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_balanced_advantage_contrast_mpc | -0.0141 | [-0.0254, -0.0030] | 15/16 |
| causal_balanced_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_balanced_advantage_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_mpc | -0.0064 | [-0.0351, 0.0233] | 10/16 |
| causal_rigid_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| cfcmt_mechanism_contrast_mpc | -0.0064 | [-0.0351, 0.0233] | 10/16 |
| cfcmt_mechanism_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_target_adapter_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_target_only_contrast_guard | 0.0000 | [0.0000, 0.0000] | 0/16 |
