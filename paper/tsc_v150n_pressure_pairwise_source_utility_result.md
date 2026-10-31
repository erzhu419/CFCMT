# V150N Pressure-Pairwise Source Utility Result

## Decision

V150N is rejected before closed-loop rollout. No target city passed the frozen
nested utility gate, so all seven selected arms are exact rigid CFCMT fallback.
The minimum-record and multicity admission thresholds are not relaxed after
observing this result.

## What Changed From V150M

V150N restricted every source candidate to the binary action pair formed by
the rigid CFCMT action and the PhasePressure reference. A source mechanism was
therefore allowed to express a pairwise PhasePressure preference even when it
did not rank that phase first among every feasible action.

This change produced many raw proposals on the final evaluation reserve:
Atlanta 3, Cologne 101, Hangzhou 1,682, Ingolstadt 84, New York 110, RESCO
synthetic 227, and Salt Lake City 386. The candidate transformation was thus
nondegenerate.

## Why It Failed

The nested B100 utility records remained sparse because V150N created a label
only when a cross-fitted source candidate proposed PhasePressure. Final source
record counts were 0, 20, 10, 4, 0, 1, and 3 in the same city order. Every
count was below the predeclared minimum of 48, and no city produced a
cross-fitted intervention. Consequently, the aggregate effects versus rigid
and matched placebo are exactly zero by fallback, not evidence of equal
closed-loop performance.

## Next Bounded Hypothesis

V150O preserves the same binary action pair but creates one utility label for
every eligible state where rigid and PhasePressure disagree. Source mechanisms
provide deployment-observable evidence features rather than deciding whether a
record exists. A separately nested matched placebo and a same-capacity
source-blind gate test whether any gain comes from source information rather
than target-side state labels alone.

Canonical aggregate:
`cf_h2o/results/paper_artifacts/tsc_v150n_pressure_pairwise_source_utility.json`.
