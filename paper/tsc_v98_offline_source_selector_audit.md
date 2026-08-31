# Target-Offline Causal Source Selection Audit

## Question

The v93 experiments reported a positive cross-city source-transfer effect. This
audit separates three claims that were previously conflated:

1. whether the positive closed-loop effect is reproducible on disjoint seeds;
2. whether a source city can be selected without target closed-loop rollouts;
3. whether the effect survives a genuinely source-free target-only comparator.

## Legacy Result

The frozen v93 confirmation used 56 disjoint seeds. For Jinan, its
closed-loop-developed selector chose Atlanta with source mass 1.0. Relative to
the legacy target-only anchor, mean waiting time fell by 4.60% (95% confidence
interval, -4.87% to -4.32%), with improvement on 56 of 56 seeds. The numerical
effect is valid, but source identity, source mass and guard strength were chosen
using eight target closed-loop development seeds. It therefore cannot support a
target-offline selection claim.

## Offline Selector

The v98 selector uses only the balanced B100 target counterfactual pool. Five
scenario-and-seed-stratified folds evaluate each source city and source mass in
`{0, 0.25, 0.5, 0.75, 1.0}` by normalized action regret. A source is admitted
only when its mean improvement exceeds 0.5%, its one-sided 95% upper confidence
bound is below zero, and no fold regresses by more than 1%. Failure of any gate
returns the exact target-only model.

The frozen selector returned:

| Target city | Selected source | Source mass | Offline fold result |
|---|---:|---:|---:|
| Los Angeles | none | 0.0 | exact target-only fallback |
| Jinan | Hangzhou | 1.0 | all 5 folds improved; mean regret delta -10.19% |

No target closed-loop rollout was read by this selection stage.

## Strict Comparator Correction

The legacy `causal_target_only` estimator instantiated a source-prior model and
assigned the target head a numerically near-one weight. Source predictions could
therefore break action ties. The corrected `causal_target_only_v2` estimator
fits only target rows and reports `source_rows_consumed = 0`.

The deployable strict Jinan model is identified by SHA-256
`3424307a5399c4fbed63bc3158001adaf53eaa8ef04d53daf56c5eef744fdd13`.
The remote provenance audit found this exact anchor in all 336 closed-loop result
files.

## Fresh Confirmation

The strict comparison uses 56 newly generated seeds that do not overlap v93 or
the first v98 confirmation. It contains three Jinan demand scenarios and two
paired methods, for 336 rollouts. Source selection, source mass, guard and all
statistical gates were fixed before these rollouts.

| Estimand | Relative waiting-time delta | 95% bootstrap CI | Improved seeds |
|---|---:|---:|---:|
| Equal-scenario seed mean | -4.53% | [-4.79%, -4.28%] | 56 / 56 |
| Jinan nominal demand | -4.29% | [-4.66%, -3.89%] | 56 / 56 |
| Jinan demand 2000 | -4.44% | [-4.89%, -3.98%] | 56 / 56 |
| Jinan demand 2500 | -4.84% | [-5.29%, -4.39%] | 56 / 56 |

The paired Wilcoxon two-sided p-value is
`7.55e-11`. The offline-selected method recorded three collision incidents
versus four for strict target-only, and both methods recorded zero teleports.
All five pre-specified efficacy and safety gates passed.

## Interpretation

The old positive effect is real. More importantly, the corrected experiment
shows that target-offline causal source selection retains the effect when the
comparison model consumes no source rows. The defensible claim is therefore
that source-city information can improve a target-adapted causal controller
after source identity is selected from passive target counterfactual data.

## Remaining Limitation

The Jinan pressure guard is shared by both compared methods, so it cannot explain
their paired difference. However, its risk multiplier was inherited from v93
target closed-loop development. The current experiment proves the source
selection effect, not a fully target-closed-loop-free deployment protocol. A
source-only or analytic guard must be frozen before the unseen-city experiment.

## Canonical Evidence

- Offline selections: `cf_h2o/results/cluster/tsc_v98_target_offline_source_selection_20260831/offline_selection_v3/`
- Strict model audits: `cf_h2o/results/cluster/tsc_v98_target_offline_source_selection_20260831/strict_target_only_v2/`
- Frozen confirmation protocol: `cf_h2o/config/traffic_signal_tsc_v98_strict_target_fresh_confirmation_v2.json`
- Corrected audit: `cf_h2o/results/cluster/tsc_v98_target_offline_source_selection_20260831/strict_fresh_confirmation_v2/audit_v2.json`
- The earlier `audit.json` is retained as a superseded numerical audit; its
  hard-coded claim sentence incorrectly named the v93 anchor.
