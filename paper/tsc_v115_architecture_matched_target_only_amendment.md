# V115--V117 architecture-matched target-only amendment

## Timing

This amendment was made after the V114/V116 counterfactual cache collection had
started, but before any V115 model fit, V116 source-selector outcome, or V117
closed-loop rollout existed. The V114/V116 cache collector, cache identities,
450-second pure halted-queue estimand, simulator inputs, adaptation seeds,
selector-development seeds, and 64 V117 confirmation seeds are unchanged.

## Problem found before fitting

The planned `target_only_b100` comparator consumed no source rows, but it used a
single low-capacity `TargetOnlyActionAdvantageRegressor`. The CFCMT B100 arm used
a group-normalized rigid anchor plus an antisymmetric all-action correction.
That comparison enforced the target-label budget but did not hold model
architecture fixed, so it could not cleanly identify the contribution of source
rows.

## Amendment

The primary `target_only_b100` comparator now:

1. uses the same group-normalized rigid anchor as the source-augmented model;
2. uses the same antisymmetric pairwise correction;
3. uses the same blend candidate selected only by seven-source-city LOCO;
4. uses the same hyperparameters, PhasePressure reference, pure 450-second
   halted-queue target, and 100 target action-group identifiers; and
5. consumes zero source transition rows.

The earlier low-capacity strict target-only estimator is still serialized as a
secondary diagnostic artifact. It is not used for the primary source
contribution gate.

## Relation to the positive V98 result

This amendment does not invalidate the V98 finding. V98 used 56 disjoint
closed-loop seeds and found that the offline-selected source-augmented model
reduced mean waiting time by 4.53% relative to a target-only estimator that
consumed zero source rows. That remains valid evidence that source information
helped under the frozen V98 protocol. The narrower issue is attribution: the
V98 target-only estimator had lower capacity than the source-augmented model,
so the observed difference did not isolate source rows from architecture. The
V115--V117 primary comparison repeats the source-contribution question with
architecture, hyperparameters, blend rule and B100 target groups held fixed.
It is a stricter follow-up, not a correction of V98's numerical result.

## Version changes

- V115 fit protocol: `tsc-v115-pure-waiting-mechanism-fit-v3`.
- V116 selector result: `tsc-v116-pure-waiting-causal-multisource-selector-v2`.
- V117 confirmation protocol and downstream freeze, rollout, shard, launch and
  audit artifacts: version 2.

The efficacy thresholds, safety thresholds, primary outcome, pairing unit,
bootstrap settings, and fresh-seed list were not changed. The amendment makes
the comparator stricter and the source-contribution interpretation narrower;
it does not use an observed selector or closed-loop result.
