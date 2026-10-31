| method                              | uses_unlabeled_target_summary | uses_target_transition_labels_for_selection | mean_total_mse | mean_ratio_vs_h2oplus | wins_vs_h2oplus |
| ----------------------------------- | ----------------------------- | ------------------------------------------- | -------------- | --------------------- | --------------- |
| H2O+ pooled dense residual          | False                         | False                                       | 884.8525       | 1.0                   | 0               |
| H2O+ source-weighted dense residual | True                          | False                                       | 846.4656       | 0.796                 | 5               |
| Rigid CFCMT                         | False                         | False                                       | 851.8929       | 0.7651                | 4               |
| Guarded CFCMT source ensemble       | True                          | False                                       | 836.1148       | 0.7082                | 6               |
