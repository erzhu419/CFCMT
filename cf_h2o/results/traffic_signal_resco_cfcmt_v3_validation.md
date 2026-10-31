# CFCMT V3 Action-Contrast RESCO Benchmark

Source labels are matched safe one-step counterfactuals. Each target fold additionally uses 60 matched target-simulator action groups for adaptation.

| Policy | System TTS/lane | Queue/lane | P90 queue/lane | Completion |
|---|---:|---:|---:|---:|
| cfcmt_mechanism_contrast_guard | 7.7788 | 1.8633 | 2.7926 | 0.7041 |
| causal_core_advantage_contrast_guard | 7.7966 | 1.8504 | 2.7926 | 0.7123 |
| phase_pressure | 8.1837 | 1.8841 | 2.7330 | 0.6944 |

## Selected Rule Prior

| Target | Prior |
|---|---|
| cologne1 | phase_pressure |
| ingolstadt1 | phase_pressure |
| atlanta_1x5 | phase_pressure |
| hangzhou_bc_tyc | phase_pressure |

## Hierarchical Bootstrap

| Baseline | Relative delta | 95% CI | Wins |
|---|---:|---:|---:|
| phase_pressure | -0.0335 | [-0.0644, -0.0075] | 3/4 |
| causal_core_advantage_contrast_guard | -0.0075 | [-0.0224, 0.0000] | 1/4 |
