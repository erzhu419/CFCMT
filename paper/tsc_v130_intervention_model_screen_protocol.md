# TSC V130 Intervention Model Screen Protocol

## Purpose

Determine whether the action-identification bottleneck can be solved before
source transfer is reintroduced. V130 contains no source prediction, source
weight or source selector.

## Frozen model families

1. **Causal improvement classifier.** A histogram gradient-boosting classifier
   predicts whether each non-reference action improves on PhasePressure after a
   group-relative tie margin.
2. **Group-normalized advantage regressor.** A candidate-only histogram
   gradient-boosting regressor uses action-group target normalization and
   balances beneficial versus non-beneficial candidate weights.

Both use only the causal state/action parent set. City context and route IDs are
excluded. Candidate actions must not reduce instantaneous service pressure.

## Evaluation

For each of 22 selector seeds, 11 disjoint seeds fit the model, 10 seeds select
and authorize an intervention-retention profile, and one seed is held out. The
held-out seed is never used for fit or calibration. Calibration and final
absolute gates are identical to V129.

## Decision

A model passes only if its held-out mean normalized waiting delta is at most
`-0.0005` and its upper 95% bound is below zero. A pass authorizes a separate
source-veto/placebo experiment using only the selected model family. Failure of
both arms closes this feature/model family and requires a new intervention
representation rather than further source-weight tuning.
