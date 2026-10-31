# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 120 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| dense_balanced_advantage_contrast_mpc | 4.5791 | 0.9059 | 1.3085 | 0.5220 |
| causal_target_only_contrast_guard | 4.6036 | 0.8766 | 1.2712 | 0.5283 |
| causal_target_adapter_contrast_guard | 4.6786 | 0.8795 | 1.2772 | 0.5214 |
| causal_core_advantage_contrast_guard | 4.6886 | 0.8979 | 1.2756 | 0.5202 |
| dense_balanced_advantage_contrast_regularized | 4.6988 | 0.8922 | 1.2567 | 0.5220 |
| causal_core_advantage_contrast_mpc | 4.7180 | 0.9111 | 1.3079 | 0.5129 |
| causal_balanced_advantage_contrast_mpc | 4.7511 | 0.9205 | 1.3212 | 0.5134 |
| causal_core_advantage_contrast_regularized | 4.7957 | 0.9005 | 1.2649 | 0.5162 |
| causal_rigid_advantage_contrast_mpc | 4.8348 | 0.9219 | 1.2991 | 0.5094 |
| cfcmt_mechanism_contrast_mpc | 4.8379 | 0.9021 | 1.2726 | 0.5131 |
| causal_rigid_advantage_contrast_regularized | 4.8383 | 0.9004 | 1.2769 | 0.5152 |
| cfcmt_mechanism_contrast_regularized | 4.8383 | 0.9004 | 1.2769 | 0.5152 |
| cfcmt_mechanism_contrast_guard | 4.8810 | 0.9086 | 1.2817 | 0.5107 |
| causal_rigid_advantage_contrast_guard | 4.8819 | 0.9084 | 1.2817 | 0.5103 |
| fixed_program | 5.0706 | 1.3398 | 2.0688 | 0.4697 |
| simulator_contrast_guard | 5.0742 | 0.9140 | 1.2423 | 0.5046 |
| dense_contrast_guard | 5.0798 | 0.9145 | 1.2457 | 0.5040 |
| selected_source_prior | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_regularized | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| causal_balanced_advantage_contrast_guard | 5.0816 | 0.9151 | 1.2433 | 0.5039 |
| dense_balanced_advantage_contrast_guard | 5.0817 | 0.9169 | 1.2450 | 0.5032 |
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
| fixed_program | -0.1301 | [-0.2361, -0.0119] | 15/16 |
| max_pressure | -0.0111 | [-0.0364, 0.0069] | 9/16 |
| phase_pressure | -0.0256 | [-0.0601, 0.0053] | 7/16 |
| spillback_pressure | -0.0484 | [-0.1005, -0.0129] | 15/16 |
| selected_source_prior | -0.0117 | [-0.0352, 0.0000] | 1/16 |
| simulator_contrast_guard | -0.0105 | [-0.0352, 0.0055] | 5/16 |
| dense_contrast_guard | -0.0111 | [-0.0350, 0.0021] | 2/16 |
| dense_balanced_advantage_contrast_mpc | 0.0139 | [-0.0086, 0.0446] | 8/16 |
| dense_balanced_advantage_contrast_regularized | 0.0074 | [-0.0009, 0.0229] | 1/16 |
| dense_balanced_advantage_contrast_guard | -0.0122 | [-0.0357, 0.0004] | 5/16 |
| causal_core_advantage_contrast_mpc | -0.0025 | [-0.0288, 0.0194] | 9/16 |
| causal_core_advantage_contrast_regularized | 0.0029 | [0.0000, 0.0073] | 0/16 |
| causal_core_advantage_contrast_guard | 0.0077 | [-0.0035, 0.0248] | 4/16 |
| causal_balanced_advantage_contrast_mpc | -0.0013 | [-0.0167, 0.0179] | 12/16 |
| causal_balanced_advantage_contrast_regularized | -0.0117 | [-0.0352, 0.0000] | 1/16 |
| causal_balanced_advantage_contrast_guard | -0.0117 | [-0.0352, 0.0000] | 1/16 |
| causal_rigid_advantage_contrast_mpc | -0.0117 | [-0.0286, 0.0026] | 13/16 |
| causal_rigid_advantage_contrast_regularized | 0.0000 | [0.0000, 0.0000] | 0/16 |
| causal_rigid_advantage_contrast_guard | -0.0025 | [-0.0076, 0.0019] | 5/16 |
| cfcmt_mechanism_contrast_mpc | -0.0060 | [-0.0167, 0.0050] | 11/16 |
| cfcmt_mechanism_contrast_guard | -0.0016 | [-0.0071, 0.0036] | 4/16 |
| causal_target_adapter_contrast_guard | 0.0093 | [-0.0009, 0.0279] | 3/16 |
| causal_target_only_contrast_guard | 0.0165 | [0.0011, 0.0411] | 3/16 |
