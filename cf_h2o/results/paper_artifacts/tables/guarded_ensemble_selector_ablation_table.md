| method                                                    | result_source             | splits | mean_total_mse | mean_ratio_vs_h2oplus | wins_vs_h2oplus |
| --------------------------------------------------------- | ------------------------- | ------ | -------------- | --------------------- | --------------- |
| H2O+ dense ridge                                          | guarded run               | 6      | 884.8525       | 1.0                   | 0               |
| H2O+ dense source ensemble, same unlabeled target summary | guarded run               | 6      | 846.4656       | 0.796                 | 5               |
| Rigid CFCMT                                               | guarded run               | 6      | 851.8929       | 0.7651                | 4               |
| Rigid-first pooled delta selector                         | non-ensemble selector run | 6      | 844.8108       | 0.7621                | 6               |
| Source-city ensemble selector, unguarded                  | unguarded ensemble run    | 6      | 838.6957       | 0.7446                | 6               |
| Source-city ensemble selector, guarded                    | guarded ensemble run      | 6      | 836.1148       | 0.7082                | 6               |
| Per-mechanism splice selector                             | guarded ensemble run      | 6      | 864.5925       | 0.8487                | 6               |
