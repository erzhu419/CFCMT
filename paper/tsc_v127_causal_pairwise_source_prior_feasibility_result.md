# V127 Causal Pairwise Source-Prior Feasibility Result

## Status

V127 completed under immutable snapshot `08997193db8cfaff6a53` in
5,824.03 s. The corrected JSON-safe task was scheduler task `t88306` with
signature `CFCMT/v127/causal-pairwise-source-prior-b500-v2-json-safe`.
The local and remote `result.json` SHA-256 is
`126526c8adc900aa63a219a47288c9c1c37712fcfd67ea9d22784de8b9cde782`.

The result is an abundant-target development diagnostic on 22 Jinan selector
seeds, 19,851 matched action groups and 158,808 action rows. Per outer fold,
16 complete target selector seeds fitted the ranker, five disjoint seeds
calibrated intervention retention and one seed remained held out. It is not a
few-shot or untouched-city result.

## Result

The frozen development gate rejected the method. Positive normalized deltas
below are worse than exact PhasePressure.

| Arm | Mean normalized delta | 95% CI | Enabled folds | Interventions |
|---|---:|---:|---:|---:|
| Target-only | +0.0000998 | [0, +0.0002995] | 1/22 | 11 |
| Source-aligned | +0.0000998 | [0, +0.0002995] | 1/22 | 11 |
| Source-placebo | +0.0000998 | [0, +0.0002995] | 1/22 | 11 |

The paired source-aligned minus target-only effect was exactly zero, as was
the paired source-aligned minus source-placebo effect. The source channel
therefore supplied no deployable incremental value under this construction.

All 22 folds had calibration candidates with at least 20 interventions, but
only one fold in each arm produced any candidate whose one-sided calibration
upper bound was below zero. The single enabled fold held out seed `74374`,
selected the 20% retention profile and made 11 interventions. Its held-out
normalized delta was `+0.00219646`, and 7 of the 11 selected interventions
were harmful. Every other fold fell back exactly to PhasePressure.

## Interpretation

V127 rejects both the absolute policy and the proposed source representation.
The nonlinear pairwise ranker did not recover a stable improvement even with
abundant target counterfactual labels. Exposing
`0.75 * (Atlanta score - target score)` as one more action feature also did not
make source value identifiable: small training-fit differences occurred in
some folds, but the calibrated held-out policies were identical to target-only
and placebo.

This result does not reverse V98 or V123. Those experiments established that
source information improved an architecture-matched target estimator under
their frozen relative comparisons. V127 asks the stronger question of whether
that information can be converted into a stable absolute intervention policy
over PhasePressure; it cannot.

The current pairwise-source-feature family is closed. The next method must
separate the roles explicitly: a source-pretrained pairwise prior, a
target-group out-of-fold residual or specialist, and a source-null option that
can set deployed source contribution to zero. Source selection, residual
fitting and safety calibration must all share one declared union of target
groups under the total information-budget contract. A new abundant feasibility
gate is required before any few-shot curve or untouched-city confirmation.
