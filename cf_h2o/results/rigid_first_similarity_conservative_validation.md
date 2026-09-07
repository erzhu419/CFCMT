# Rigid-First Similarity-Conservative Validation

Command output: `cf_h2o/results/rigid_first_similarity_conservative_validation.json`

Configuration:
- Full-route city tensors.
- `max_train_rows_per_city=10000`, `max_eval_rows_per_city=30000`.
- Neural full modules skipped with `--skip-neural-full`.
- City tensor cache enabled at `cf_h2o/results/cache/city_tensors`.
- Source leave-one selector uses target/source similarity for ranking, but requires:
  - weighted win rate >= `0.8`
  - unweighted win rate >= `1.0`
  - weighted mean improvement >= `0.5%`

## Summary

| Method | Mean total MSE | Mean vs H2O+ | Wins vs H2O+ |
|---|---:|---:|---:|
| H2O+ dense ridge | 884.853 | 1.000 | 0/6 |
| Rigid CFCMT ridge | 851.893 | 0.765 | 4/6 |
| Rigid-first source-gated delta | 844.811 | 0.762 | 6/6 |
| Rigid-first static-local delta, alpha=0.25 | 848.400 | 0.774 | 4/6 |
| Rigid-first dense+static-local delta, alpha=0.25 | 851.302 | 0.753 | 4/6 |

## Selector Decisions

| Split | Selected candidate | Rigid CFCMT MSE | Source-gated MSE |
|---|---|---:|---:|
| source_open_na_to_singapore | dense_static_local_delta_alpha_0p75 | 2388.763 | 2367.517 |
| source_singapore_austin_halifax_to_mbta | rigid_cfcmt | 116.497 | 116.497 |
| leave_one_city_out_all::singapore_lta_all | dense_static_local_delta_alpha_0p75 | 2388.763 | 2367.517 |
| leave_one_city_out_all::austin_capmetro_all | rigid_cfcmt | 47.640 | 47.640 |
| leave_one_city_out_all::halifax_transit_all | rigid_cfcmt | 53.198 | 53.198 |
| leave_one_city_out_all::mbta_all | rigid_cfcmt | 116.497 | 116.497 |

## Interpretation

Similarity weighting is useful only as a ranking signal after a candidate has already passed strict unweighted source validation. Without the unweighted win-rate guard, the selector over-trusts Austin-like source folds and incorrectly enables dense+static delta for MBTA, raising mean MSE to `862.576`.

The conservative similarity selector improves over the uniform gated result (`846.243`) by choosing a stronger Singapore delta (`alpha=0.75` instead of `alpha=0.50`) while preserving rigid fallback elsewhere.

The current recommended full method is therefore:

`rigid CFCMT + deterministic source-validated delta + target/source similarity ranking + rigid fallback`.

The old neural AutoDAG/latent full route should remain diagnostic until it can be reintroduced as a bounded delta under the same guard.
