# TSC V143 Multicity Uniform Source Ensemble Protocol

## Question

V140 and V142 rejected target-label-fitted source weights, yet their fixed
uniform-source descriptive arms improved Jinan target-only across two seed
partitions and all six tested B budgets. V143 tests whether this symmetric
ensemble effect generalizes across complete city groups or is another Jinan
special case.

This protocol is frozen after V142 and before any V143 target result is read.
All seven cities have appeared in earlier development studies, so V143 is
development, not confirmation.

## Complete-city holdout

Targets are Atlanta, Cologne, Hangzhou, Ingolstadt, New York, RESCO synthetic
and Salt Lake City. For each target, every scenario from that city is removed
from all source fits. The source cache contains 17 admitted pure-waiting
scenarios; unsafe Ingolstadt21 remains excluded by the unchanged V114 audit.

Exactly B25 target action groups are selected with deterministic balanced
coverage over every available target scenario and simulator seed. Each stratum
must retain evaluation groups. All remaining target-city groups form the
evaluation set and are never used for fitting or selection.

## Information-matched arms

- `target_only_causal`: causal anchored model trained on the target B25 only;
- `pooled_h2oplus`: dense H2O+-style model trained on all six source cities and
  the same target B25, without source-domain separation;
- `uniform_cfcmt`: six one-source causal models, each trained on one non-target
  city plus the same target B25, averaged with fixed mass 1/6;
- `matched_source_placebo`: the same uniform weights applied after
  deterministic whole-action-group permutation of source residual blocks;
- exact PhasePressure as the absolute reference.

No target label chooses source identity, source mass or city-specific fallback.
The uniform rule is permutation-invariant and fixed globally.

## Development gate

Results are first averaged equally over scenario-seed units within a city, then
equally over the seven city groups. Uniform CFCMT must improve each of
target-only causal, pooled H2O+ and matched placebo by at least 0.0005 on
average, with a negative one-sided 95% city-level upper confidence bound. It
must improve at least five of seven cities against each real comparator and
must not regress any city by more than 0.01 against target-only or H2O+.

Absolute PhasePressure performance is reported separately and is not implied
by a positive transfer-layer result.

## Successor boundary

A pass authorizes a new, separately frozen untouched-city protocol. It does not
revive V141 or turn Boston operational admission into efficacy evidence. A fail
closes the fixed uniform ensemble and motivates source-only mechanism
compatibility weighting rather than target-label gate tuning.
