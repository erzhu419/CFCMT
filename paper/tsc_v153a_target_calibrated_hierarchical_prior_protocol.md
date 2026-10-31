# V153A Target-Calibrated Hierarchical Prior Protocol: Feature-Binding Correction v3

## Version Record

The original v1 result remains `REJECT` and is retained at
`cf_h2o/results/paper_artifacts/tsc_v153a_target_calibrated_hierarchical_prior.json`.
Its selector reused the other four cached B100 OOF predictions to select a
candidate for each held-out fold. Those predictions had been fitted using the
outer held-out labels, so that audit did not isolate the complete selection
procedure. The evaluation reserve outside B100 remained excluded from selection.

V2 corrected that nesting error and completed all seven targets. Its retained
aggregate is
`cf_h2o/results/paper_artifacts/tsc_v153a_target_calibrated_hierarchical_prior_v2.json`.
The recorded gate was `REJECT`, with one admitted city. A subsequent Cologne
diagnostic established a separate prediction error: the rigid target ranker
was fitted using `CONTRAST_FEATURES_V3`, but prediction received the longer
mechanism contrast schema and selected stored column positions. Nine of its
29 model inputs therefore read different feature names. V153's inner fits,
outer fits and final reserve evaluation all use this affected prediction path.
The v1/v2 numerical artifacts remain historical records; their source-effect
and admission claims await corrected evaluation.

V3 restores prediction input binding by each fitted ranker's stored feature
names. The explicit contract is `feature_binding=stored_feature_names` in the
configuration, target result, aggregate and launch record. Its target, method,
selector, aggregate and launch protocols end in `-v3`, and scheduler signatures
use `CFCMT/v153a/target-calibrated-hierarchical-prior-v3/<city>`. The launcher
requires the v3 configuration and binding, and the aggregate requires v3 target
results with the same binding. Candidate blocks, strengths, controls, B100
budget, data split, feature values and scales, admission thresholds and the
complete v2 inner/outer fitting procedure remain unchanged.

## Question

V153A asks whether the fixed B100 target history can identify when a robust
cross-city mechanism prior is useful. This follows the V152A rejection of a
target-label-free source-city selector. It changes the information budget, not
the observed V152A thresholds or candidate grid.

## Information Budget

For each target city, six other cities define the equal-city median mechanism
prior and its MAD-adaptive precision. The target contributes exactly 100
labelled action groups. These B100 groups are used both for target model fitting
and prior selection; the same labels are available to the target-only baseline.
All remaining target groups are held out from fitting, selection, normalization
and threshold decisions, and supply the fixed development evaluation reserve.

V153A is few-shot target offline adaptation. It is not zero-shot transfer.

## Selector

The source prior, six mechanism blocks, five strengths, matched placebo,
zero-centred control and residual integration are identical to V152A. B100 is
partitioned into five deterministic 20-group folds. Each outer audit withholds
one fold from all target fitting and candidate selection. Within the other 80
groups, four inner fits train on 60 groups and predict the remaining 20. The
candidate is selected using only these four inner predictions. A separate
80-group fit evaluates that selected candidate on the outer 20 groups.

The five outer 80-to-20 predictions also form the complete B100 OOF table used
to choose the final candidate, after which the target model is fitted on all
100 groups and evaluated on the unchanged reserve. Each city therefore requires
20 inner fits, five outer fits and one final fit; the number of distinct labelled
target groups remains 100.

Admission requires both the final five-fold candidate and the nested selector
procedure to improve rigid, matched placebo and zero-centred prior by at least
`0.0005`; the nested selector must be non-degrading in at least four of five
folds for every comparison and make at least five interventions. Failure
returns rigid CFCMT exactly.

## Development Gate

Across seven development targets, at least two cities must admit a source prior.
No city may regress, each admitted city must improve rigid and both controls on
the untouched evaluation groups, and each rejected city must recover rigid
exactly. Passing authorizes a closed-loop development run only. It cannot
establish untouched-city or real-world efficacy.

## Correction Run

The seven-city correction writes to the separate local directory
`cf_h2o/results/cluster/tsc_v153_target_calibrated_hierarchical_prior_20260909/feature_binding_v3/`
and matching server directory under `CFCMT_RESULTS`. Its aggregate is
`cf_h2o/results/paper_artifacts/tsc_v153a_target_calibrated_hierarchical_prior_v3.json`.
The v1/v2 result files remain separate from the v3 aggregate. Every city requires
new inner, outer and final predictions, candidate selection and admission
adjudication under the corrected binding.

This is a correction on the existing development cities and evaluation reserve.
The reserve was already reported during v1/v2, so v3 is not fresh confirmation.
Its result must be reported separately before any follow-on experiment.
