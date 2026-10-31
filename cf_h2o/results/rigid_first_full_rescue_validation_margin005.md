# Rigid-First Full Rescue Validation

Command output: `cf_h2o/results/rigid_first_full_rescue_validation_margin005.json`

Configuration:
- Full-route city tensors.
- `max_train_rows_per_city=10000`, `max_eval_rows_per_city=30000`.
- Neural full modules skipped with `--skip-neural-full`.
- Source leave-one selector requires `win_rate=1.0` and `min_improvement=0.005`.

## Summary

| Method | Mean total MSE | Mean vs H2O+ | Wins vs H2O+ |
|---|---:|---:|---:|
| H2O+ dense ridge | 884.853 | 1.000 | 0/6 |
| Rigid CFCMT ridge | 851.893 | 0.765 | 4/6 |
| Rigid-first source-gated delta | 846.243 | 0.763 | 6/6 |
| Rigid-first static-local delta, alpha=0.25 | 848.400 | 0.774 | 4/6 |
| Rigid-first dense+static-local delta, alpha=0.25 | 851.302 | 0.753 | 4/6 |

## Source-Gated Decisions

| Split | Selected candidate | Rigid CFCMT MSE | Source-gated MSE |
|---|---|---:|---:|
| source_open_na_to_singapore | dense_static_local_delta_alpha_0p50 | 2388.763 | 2371.814 |
| source_singapore_austin_halifax_to_mbta | rigid_cfcmt | 116.497 | 116.497 |
| leave_one_city_out_all::singapore_lta_all | dense_static_local_delta_alpha_0p50 | 2388.763 | 2371.814 |
| leave_one_city_out_all::austin_capmetro_all | rigid_cfcmt | 47.640 | 47.640 |
| leave_one_city_out_all::halifax_transit_all | rigid_cfcmt | 53.198 | 53.198 |
| leave_one_city_out_all::mbta_all | rigid_cfcmt | 116.497 | 116.497 |

## Interpretation

The old neural full route is not a strict superset of rigid CFCMT and can degrade performance. The rescued full route is now rigid-first: it learns a deterministic residual delta on top of rigid CFCMT and uses source leave-one validation to set alpha. With a 0.5% stability margin, the selector only enables the delta on Singapore targets and falls back to rigid elsewhere.

This makes the current full method a conservative extension of rigid CFCMT rather than a replacement. The next useful modules are target/source similarity weighting for the selector and cached city tensor loading for faster ablations.
