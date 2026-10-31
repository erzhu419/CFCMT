# V153A Target-Calibrated Hierarchical Prior: Feature-Binding Correction v3 Result

## Decision

`REJECT`: none of the seven targets admitted a source prior under the frozen
B100 selector. Every city returned the corrected rigid target-only policy
exactly. All selected effects against rigid, matched placebo and zero-centred
prior were consequently zero, including their bootstrap intervals. The
prespecified requirement for at least two admitted cities and the three
strictly beneficial macro-effect checks failed. Source-aware closed-loop
development remains unadmitted.

All seven tasks completed successfully. These results use the corrected rigid
baseline throughout; the v1/v2 numerical artifacts and recorded `REJECT`
decisions remain retained historical outputs. In particular, the v2 Ingolstadt
admission is not evidence of source value after the feature-binding diagnosis.

## Correction And Frozen Design

The Cologne diagnostic reproduced the original rigid scores exactly and found
that nine of the anchor's 29 inputs read different feature names at prediction:
training used `CONTRAST_FEATURES_V3`, while prediction used the expanded
mechanism contrast schema. The anchor selected saved integer positions instead
of resolving its saved feature names. V153 used the affected path for every
inner, outer and final reserve prediction.

V3 resolves each anchor input by its stored feature name in the current
dataset. The actual antisymmetric correction model already resolved names.
The feature values, their scales and the fitted-model design were not changed.
Each target retains the same 100 labelled action groups, six source cities,
six mechanism blocks, five strengths and both source controls. The complete
nested design remains five outer 80-to-20 fits, each with four inner 60-to-20
fits; the full B100 OOF table selects the final candidate before the B100 fit
is evaluated on the same reserve. No reserve labels enter fitting or selection.

Admission still requires the final OOF and nested mean gains to meet
`0.0005` against all three comparators, at least four non-degrading nested
folds per comparison and at least five nested interventions. No threshold,
candidate or information-budget changes were made for this correction.

## Corrected Selection Evidence

Positive gains mean lower normalized cost for source. All seven cities were
rejected.

| Target | Nested gain vs rigid | vs placebo | vs zero-centred | Interventions | Main failed condition |
|---|---:|---:|---:|---:|---|
| Atlanta | `+0.071600` | `+0.001600` | `0.000000` | 10 | Ties zero-centred in both full OOF and nested selection |
| Cologne | `+0.001069` | `0.000000` | `+0.001782` | 9 | Nested selection ties placebo |
| Hangzhou | `+0.007640` | `-0.002454` | `-0.002454` | 7 | Loses to both controls; only 3/5 non-degrading folds vs rigid |
| Ingolstadt | `0.000000` | `0.000000` | `0.000000` | 0 | Zero nested gain and insufficient interventions |
| New York | `+0.007853` | `0.000000` | `0.000000` | 1 | No full-OOF gain; nested control ties and insufficient interventions |
| RESCO synthetic | `-0.019433` | `-0.009689` | `-0.012821` | 14 | Loses to all comparators; non-degrading-fold counts also fail |
| Salt Lake City | `+0.008702` | `0.000000` | `0.000000` | 2 | Nested control ties and insufficient interventions |

Ingolstadt's final candidate changed to `mobility|lambda=0.05`. Its full-OOF
mean gains were positive against all comparators (`+0.00638996`), but the
complete nested selector produced no interventions or gains. The earlier v2
local positive therefore does not survive correction. This change is a result
of correcting the existing procedure, not an independent replication test.

## Reserve Diagnostics Under Forced Source Use

The selected policy uses exact fallback in every city. The following separate
diagnostic forces each full-OOF-selected candidate onto the fixed reserve,
despite its rejected admission. Negative effects favor source. These are
normalized 450-second cost contrasts, not percentages or closed-loop waiting
time changes.

| Target | Forced source minus corrected rigid | minus matched placebo | minus zero-centred |
|---|---:|---:|---:|
| Atlanta | `-0.09034046` | `0.00000000` | `0.00000000` |
| Cologne | `+0.00443824` | `+0.00055594` | `+0.00015672` |
| Hangzhou | `+0.00389389` | `-0.00032838` | `+0.00471584` |
| Ingolstadt | `-0.00193641` | `+0.00002408` | `-0.00078893` |
| New York | `0.00000000` | `0.00000000` | `0.00000000` |
| RESCO synthetic | `-0.00400119` | `-0.00095621` | `-0.00102421` |
| Salt Lake City | `+0.00303573` | `+0.00081656` | `-0.00012548` |

Forced source has macro effect `-0.01213003` versus corrected rigid, with
city-bootstrap 95% interval `[-0.03879283, +0.00266025]`. Atlanta dominates
that mean, but its source, matched-placebo and zero-centred reserve means are
identical (`-0.56763185`). That gain cannot identify added value from the
source prior centre. RESCO has a favorable forced reserve result against all
three comparators, but its B100 nested selector regresses all three. This
shows a mismatch between the observed reserve benefit and what the frozen
selector can reliably identify with the target budget.

The corrected results therefore retain a narrower negative conclusion: this
specific B100 prior-selection procedure does not establish a repeatable,
source-specific benefit across the seven development cities. They do not
establish that every possible source prior or conditional-dynamics model is
ineffective.

## Limitations

The reserve was already reported during v1/v2, so v3 is a correction on an
existing development bank, not fresh confirmation. The mechanism coefficients
still predict normalized 450-second cost contrasts under the existing rollout
policy; this experiment does not test transfer of independently measured
short-horizon physical responses. The zero-centred control retains
source-derived precision and isolates the added value of the prior centre.
No closed-loop efficacy or safety conclusion follows from this offline result.

## Runtime And Artifacts

Snapshot `589fd20266b7265b6f2f` includes the feature-binding fix and the frozen
v3 protocol. Tasks `t90757`--`t90763` all reached scheduler status `done`.
Per-city computation time was `92.81`--`204.84` seconds. Retrieval contained
only seven small `result.json` files; no CSV or checkpoint was downloaded.

- Aggregate: `cf_h2o/results/paper_artifacts/tsc_v153a_target_calibrated_hierarchical_prior_v3.json`
- Target results: `cf_h2o/results/cluster/tsc_v153_target_calibrated_hierarchical_prior_20260909/feature_binding_v3/`
- Launch record: `cf_h2o/results/cluster/tsc_v153_target_calibrated_hierarchical_prior_20260909/feature_binding_launch_v3.json`
- Snapshot manifest: `cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/snapshot_feature_alignment_v1.json`
- Protocol: `paper/tsc_v153a_target_calibrated_hierarchical_prior_protocol.md`
- Exact score diagnosis: `cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/rigid_score_plateau_analysis.json`
