# Optimized Full Module Pipeline Summary

Generated from `cf_h2o/results/full_module_pipeline_validation_trust_fallback.json`.

Configuration:
- Full-route city bundles scanned.
- Neural training uses up to 30,000 uniformly sampled rows per source city.
- AutoDAG uses up to 60,000 source rows per split.
- Evaluation uses up to 100,000 target rows per split.
- Auxiliary nonlinear features disabled.
- ID parent filtering disabled for this diagnostic run.
- Trust no longer penalizes residual magnitude: `residual_scale=0.0`, `uncertainty_scale=0.15`, `graph_uncertainty_scale=0.25`.
- New fallback trust variant interpolates between neural residuals and CFCMT ridge residuals instead of falling back to the uncalibrated simulator.

| Method | Mean total MSE | Mean ratio vs H2O+ dense ridge | Wins vs H2O+ |
|---|---:|---:|---:|
| H2O+ dense ridge | 832.542 | 1.000 | 0/6 |
| CFCMT ridge template | 810.881 | 0.813 | 4/6 |
| Neural AutoDAG | 812.077 | 0.860 | 6/6 |
| Neural AutoDAG + latent | 807.466 | 0.841 | 6/6 |
| Neural AutoDAG + latent + trust gate | 803.990 | 0.866 | 6/6 |
| Neural AutoDAG + latent + CFCMT-fallback trust | 804.197 | 0.827 | 6/6 |
| Neural AutoDAG + latent + local + CFCMT-fallback trust | 811.608 | 0.888 | 5/6 |

What improved:
- The previous residual-magnitude trust gate failed on Austin/Halifax/MBTA because it suppressed large but necessary corrections.
- Removing residual-magnitude suppression fixed that failure mode.
- CFCMT-fallback trust is safer than simulator fallback: low trust now backs off to the closed-form CFCMT residual, not to the uncorrected simulator.

Recommended paper-facing interpretation:
- Use `AutoDAG + latent + CFCMT-fallback trust` as the optimized full-framework diagnostic.
- Keep `local graph` as an ablation until real snapshot-derived local graphs are available; the current static-row proxy is not reliable enough.
- Keep CFCMT ridge as the strongest deterministic instantiation and report that the optimized full-framework variant improves robustness: it wins all 6 splits vs H2O+ and slightly improves mean total MSE over ridge CFCMT, though ridge CFCMT still has the best mean ratio because it is very strong on Austin/MBTA.
