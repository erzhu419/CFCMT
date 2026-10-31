| split                                       | selected_feature_set | source_fold_score | selected_candidate                           | h2oplus_mse | guarded_mse | guarded_ratio_vs_h2oplus |
| ------------------------------------------- | -------------------- | ----------------- | -------------------------------------------- | ----------- | ----------- | ------------------------ |
| source_open_na_to_singapore                 | obs_only             | 0.8941            | dense_static_local_delta_alpha_0p75          | 2376.1427   | 2367.5165   | 0.9964                   |
| source_singapore_austin_halifax_to_mbta     | sim_only             | 0.6418            | ensemble_dense_static_local_delta_alpha_0p50 | 209.6939    | 93.8321     | 0.4475                   |
| leave_one_city_out_all::singapore_lta_all   | obs_only             | 0.8941            | dense_static_local_delta_alpha_0p75          | 2376.1427   | 2367.5165   | 0.9964                   |
| leave_one_city_out_all::austin_capmetro_all | sim_only             | 0.8808            | ensemble_dense_static_local_delta_alpha_0p25 | 63.8731     | 40.7936     | 0.6387                   |
| leave_one_city_out_all::halifax_transit_all | sim_only             | 1.0               | rigid_cfcmt                                  | 73.5689     | 53.1982     | 0.7231                   |
| leave_one_city_out_all::mbta_all            | sim_only             | 0.6418            | ensemble_dense_static_local_delta_alpha_0p50 | 209.6939    | 93.8321     | 0.4475                   |
| mean                                        |                      | 0.8254            |                                              | 884.8525    | 836.1148    | 0.7082                   |
