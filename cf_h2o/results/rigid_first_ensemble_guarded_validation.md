# Rigid-First Similarity Ensemble Guarded Validation

Command:

```bash
python3 -m cf_h2o.eval.full_module_pipeline_validation \
  --out cf_h2o/results/rigid_first_ensemble_guarded_validation.json \
  --skip-neural-full \
  --max-train-rows-per-city 10000 \
  --max-eval-rows-per-city 30000 \
  --run-source-selector \
  --selector-max-train-rows-per-city 3000 \
  --selector-max-eval-rows-per-city 3000 \
  --workers 8 \
  --split-workers 1
```

## Summary

| Method | Mean total MSE | Mean vs H2O+ | Wins vs H2O+ |
| --- | ---: | ---: | ---: |
| H2O+ dense ridge | 884.852541 | 1.000000 | 0/6 |
| Rigid CFCMT ridge | 851.892879 | 0.765115 | 4/6 |
| Rigid-first source-gated delta | 836.271949 | 0.709428 | 6/6 |
| Per-mechanism source-gated delta | 864.592468 | 0.848708 | 6/6 |
| Uncalibrated simulator | 1595.305472 | 8.187813 | 0/6 |

The guarded similarity ensemble is the best selected method in this run. It improves the previous conservative source-gated mean MSE from 844.810779 to 836.271949, while preserving 6/6 wins against H2O+.

## Split Decisions

| Split | Target | Selected rigid-first candidate | Max source weight | H2O+ MSE | Rigid CFCMT MSE | Source-gated MSE | Per-mechanism MSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| source_open_na_to_singapore | singapore_lta_all | dense_static_local_delta_alpha_0p75 | 0.464 | 2376.143 | 2388.763 | 2367.517 | 2372.863 |
| source_singapore_austin_halifax_to_mbta | mbta_all | ensemble_dense_static_local_delta_alpha_0p50 | 0.583 | 209.694 | 116.497 | 94.183 | 170.500 |
| leave_one_city_out_all::singapore_lta_all | singapore_lta_all | dense_static_local_delta_alpha_0p75 | 0.464 | 2376.143 | 2388.763 | 2367.517 | 2372.863 |
| leave_one_city_out_all::austin_capmetro_all | austin_capmetro_all | ensemble_dense_static_local_delta_alpha_0p25 | 0.580 | 63.873 | 47.640 | 41.034 | 47.635 |
| leave_one_city_out_all::halifax_transit_all | halifax_transit_all | rigid_cfcmt | 0.375 | 73.569 | 53.198 | 53.198 | 53.194 |
| leave_one_city_out_all::mbta_all | mbta_all | ensemble_dense_static_local_delta_alpha_0p50 | 0.583 | 209.694 | 116.497 | 94.183 | 170.500 |

## Interpretation

The per-mechanism hard splice remains a diagnostic only: it can win against H2O+ but is weaker than the guarded global selector because source-fold mechanism gains do not reliably transfer to the target.

The similarity ensemble is useful when target covariates identify a dominant nearby source city. The guard requires max source weight >= 0.5 before ensemble candidates can override the conservative rigid/global selector. This keeps MBTA and Austin gains, while making Halifax fall back to rigid CFCMT instead of taking the unguarded ensemble regression.
