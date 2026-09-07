# Full Module Pipeline Validation Summary

Generated from `cf_h2o/results/full_module_pipeline_validation.json`.

Scope:
- Full-route city bundles were scanned: Singapore 1,243,488 rows, Austin 310,272 rows, Halifax 266,448 rows, MBTA 969,360 rows.
- Neural module training used up to 30,000 uniformly sampled rows per source city.
- AutoDAG used up to 60,000 source rows per split.
- Evaluation used up to 100,000 uniformly sampled rows per target split.
- This is a full module-chain diagnostic, not yet the final full-row neural training run.

| Method | Mean total MSE | Mean ratio vs H2O+ dense ridge | Wins vs H2O+ |
|---|---:|---:|---:|
| H2O+ dense ridge | 832.542 | 1.000 | 0/6 |
| CFCMT ridge template | 810.881 | 0.813 | 4/6 |
| Neural template graph | 831.232 | 0.997 | 4/6 |
| Neural AutoDAG | 808.376 | 0.853 | 6/6 |
| Neural AutoDAG + latent factors | 804.196 | 0.846 | 6/6 |
| Neural AutoDAG + latent + local graph | 809.492 | 0.898 | 5/6 |
| Neural AutoDAG + latent + trust gate | 1208.471 | 5.459 | 2/6 |
| Neural AutoDAG + latent + local + trust gate | 1209.912 | 5.518 | 2/6 |
| Uncalibrated simulator | 1557.327 | 8.870 | 2/6 |

Interpretation:
- AutoDAG is a real positive module in this diagnostic: it beats H2O+ on all six configured splits and improves over the neural template graph on the hard non-Singapore targets.
- Latent mechanism factors add a small but consistent average gain over AutoDAG alone.
- Local graph features are not consistently helpful in the current static-row proxy form; they improve some cases but hurt Austin/MBTA.
- The current residual-magnitude trust gate is too conservative when the target needs large residual corrections. It helps Singapore-like near-calibration cases but fails badly for Austin/Halifax/MBTA. It should be treated as an ablation or redesigned before being used as the main method.
- Ridge CFCMT remains the strongest simple baseline and is still best on Austin and MBTA in this run.

Immediate method implication:
- The defensible full-framework claim is currently: graph-posterior and latent-factor modules can be integrated and improve average cross-city robustness over H2O+ dense residuals.
- The trust estimator needs a calibration-aware redesign, for example conditioning trust on target residual validation or separating simulator-trust from residual-amplitude suppression.
