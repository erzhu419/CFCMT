# V124 Conservative Source-Intervention Gate Result

## Status

V124 completed under immutable snapshot `c56896b8bfd0eecbcbb1` in 153.27 s.
The local and remote result SHA-256 is
`314992433c0d6619e0602d730dcbddefb016e5b74beb7618016ca29dce4cc418`.
It is a cross-fitted Jinan development experiment, not fresh-city
confirmation.

## Primary result

The preregistered development gate failed. The selected B25 policy and all
other budgets fell back to exact PhasePressure in every held-out fold. The
reported mean differences versus PhasePressure and versus the matched target
guard are therefore exactly zero, not estimated improvements.

At B50, a source guard was admissible on 13 of 22 training folds and the target
guard was admissible on all 22, but the source arm never passed the paired
source-over-target authorization test. The deployed target guard had a small
positive held-out mean delta of `+0.00002597`, so even this sparse intervention
did not improve PhasePressure out of sample.

At B500, a source guard was admissible on 11 folds while the target guard was
never admissible. Its training means were approximately `-0.00025` to
`-0.00030` versus PhasePressure with negative one-sided upper bounds, but the
effect did not meet the frozen minimum source-contribution magnitude of
`0.0005`. It was therefore not authorized. This is a diagnostic lead, not a
positive efficacy result.

## Interpretation

V123 and V124 answer different questions:

1. V123 supports relative source value: source information improved the
   architecture-matched target model at B25, B50, B100, B500 and B1000.
2. V124 rejects absolute deployability for the current action selector: no
   source arm was authorized to improve exact PhasePressure under the frozen
   conservative gate.

This does not invalidate the V98 numerical result. V98 compared a frozen
source-selected controller with a lower-capacity strict target-only estimator
under a shared historical pressure guard and found a 4.53% paired improvement.
V123 independently confirms that source rows can add predictive value after
architecture matching. The remaining failure is converting that value into a
stable absolute control gain over the strong rule baseline.

## Decision

No fresh-city efficacy confirmation is opened from V124. PhasePressure remains
the exact fallback. The next development step may inspect oracle action
headroom and the held-out performance of the admissible B500 source guard, but
it may not retroactively relax V124's thresholds or report its zero fallback as
a positive transfer result.
