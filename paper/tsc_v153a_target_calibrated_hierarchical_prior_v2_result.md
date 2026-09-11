# V153A Target-Calibrated Hierarchical Prior: Nested-CV Correction v2 Result

## Status After Feature-Binding Diagnosis

The results below are retained historical outputs of the v2 implementation.
The subsequent Cologne diagnostic
`cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/rigid_score_plateau_analysis.json`
confirmed that the rigid target ranker read stored training column positions
from a longer prediction schema: nine of its 29 inputs had different feature
names. V153 v2 uses this path for every inner fit, outer fit and reserve score;
source, placebo and zero-centred candidates all add their corrections to that
rigid score. Consequently, the `1/7` admission count, selected candidates,
intervention counts, comparator effects and confidence intervals below await
replacement by the separately versioned v3 evaluation. They cannot currently
establish source value in Ingolstadt or lack of source value in the other six
cities. The recorded `REJECT` and all v1/v2 numerical artifacts remain preserved;
closed-loop development remains unadmitted.

## Decision

`REJECT` at the frozen seven-city development gate. Ingolstadt passed the
corrected B100 selector and improved rigid CFCMT and both source controls on the
evaluation reserve. The other six cities returned exact rigid fallback. The
only failed aggregate condition was the requirement for at least two admitted
cities: v2 admitted `1/7`, compared with `0/7` in v1.

All seven correction jobs completed successfully. This result replaces the
incomplete v1 nesting evidence, while the original v1 numerical artifact and
`REJECT` decision remain retained.

## Corrected Selection Evidence

Each of the five outer folds withheld 20 target groups from fitting and
selection. Its other 80 groups supplied four inner fits of 60 training groups
and 20 validation groups. The inner predictions selected a candidate, which was
then evaluated using the separate 80-to-20 outer fit. Final candidate selection
continued to use the full B100 OOF table.

Positive gains below mean lower normalized cost for the selected source
candidate. Admission required mean gains of at least `0.0005` against all three
comparators, four non-degrading folds per comparison, at least five nested
interventions, and the unchanged full-OOF checks.

| Target | Nested gain vs rigid | vs placebo | vs zero-centred | Interventions | Admitted |
|---|---:|---:|---:|---:|---|
| Atlanta | `+0.051600` | `+0.001600` | `0.000000` | 7 | No |
| Cologne | `+0.006805` | `+0.004778` | `0.000000` | 5 | No |
| Hangzhou | `-0.008666` | `0.000000` | `0.000000` | 5 | No |
| Ingolstadt | `+0.030447` | `+0.006912` | `+0.007213` | 7 | Yes |
| New York | `+0.022206` | `0.000000` | `0.000000` | 4 | No |
| RESCO synthetic | `-0.005422` | `+0.005700` | `+0.005700` | 16 | No |
| Salt Lake City | `+0.012851` | `-0.012875` | `-0.009992` | 6 | No |

Ingolstadt retained the final `mobility|lambda=0.1` candidate. Its corrected
selector made seven nested interventions, versus four under the v1 audit, and
was non-degrading on 5/5 folds against rigid, 4/5 against placebo and 5/5 against
the zero-centred control. Atlanta and Cologne tied the zero-centred control;
Hangzhou and RESCO regressed rigid under nested selection; New York lacked
source-specific gains and sufficient interventions; Salt Lake City lost to the
source controls.

## Evaluation Reserve

Negative effects below favor source. Ingolstadt's selected source prior improved
rigid by `-0.00684446`, matched placebo by `-0.00498904`, and the zero-centred
prior by `-0.00188740`. All six rejected cities recovered rigid exactly, and no
city regressed. All source-null arms also recovered rigid exactly.

The seven-city macro selected effects were:

| Comparator | Macro effect | City-bootstrap 95% interval |
|---|---:|---|
| Rigid target-only | `-0.00097778` | `[-0.00293334, 0.00000000]` |
| Matched placebo | `-0.00071272` | `[-0.00213816, 0.00000000]` |
| Zero-centred prior | `-0.00026963` | `[-0.00080889, 0.00000000]` |

The final candidate identities and full-OOF diagnostics are exactly unchanged
from v1 in every city. The complete forced-effect aggregate is also unchanged:
forced source has macro effect `-0.00591862` versus rigid, with city-bootstrap
interval `[-0.01691186, +0.00154271]`. This isolates the correction to the
complete-selector audit and its resulting admission decision.

## Interpretation

The correction establishes one development city where the frozen B100
procedure identifies a useful source prior that also beats both controls on
the evaluation reserve. It does not support the earlier claim that every
target-calibrated source choice must fall back. Cross-city repeatability still
fails the unchanged two-city requirement, so the source-aware closed-loop
development matrix remains unadmitted.

## Limitations

This is a correction on the same development bank, including the evaluation
reserve already reported in v1. It is not fresh-city confirmation or closed-loop
control evidence. The mechanism prior still predicts the normalized 450-second
cost contrast; this experiment does not test transfer of independently measured
short-horizon physical response parameters. Zero-centred priors retain
source-derived precision, so their comparison isolates the additional value of
the prior centre.

## Runtime And Artifacts

The frozen snapshot was `2300f33fe093706cbc81`. Tasks `t90435`--`t90441` all
finished with scheduler status `done`; per-city reported computation time ranged
from `90.56` to `183.40` seconds. Retrieval was limited to small `result.json`
files; no CSV, prediction array or checkpoint was downloaded.

- Aggregate: `cf_h2o/results/paper_artifacts/tsc_v153a_target_calibrated_hierarchical_prior_v2.json`
- Target results: `cf_h2o/results/cluster/tsc_v153_target_calibrated_hierarchical_prior_20260909/nested_cv_v2/`
- Launch record: `cf_h2o/results/cluster/tsc_v153_target_calibrated_hierarchical_prior_20260909/nested_cv_launch_v2.json`
- Protocol: `paper/tsc_v153a_target_calibrated_hierarchical_prior_protocol.md`
- Retained v1 result: `cf_h2o/results/paper_artifacts/tsc_v153a_target_calibrated_hierarchical_prior.json`
