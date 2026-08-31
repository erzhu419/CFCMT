# TSC v110 target-offline source and guard protocol

## Estimand and information budget

This protocol evaluates target offline adaptation, not zero-shot transfer. For
each target city, 100 complete counterfactual action groups are selected from
the frozen passive/offline pool and partitioned into five seed-blocked folds.
No target closed-loop policy outcome is available during fitting, source
selection, source weighting, or guard selection.

## Cross-fitted decisions

For every outer fold and every source city, the anchored causal model and the
strict target-only model are fit without the held-out target action groups.
Held-out normalized action regret is recorded for source masses 0, 0.25, 0.5,
0.75, and 1.0. The deployment guard is evaluated on the same held-out groups
by comparing its selected action with the frozen generalized-pressure action.

The family-wise type-I error budget is split before evaluation:

- alpha 0.025 for the 28 nonzero source-city/weight candidates;
- alpha 0.025 for the 40 risk-multiplier/pressure-gap guard profiles;
- total family-wise alpha at most 0.05 by the union bound.

Each stage uses a one-sided Bonferroni normal critical value. A source must
clear a 0.005 mean-regret improvement margin and a 0.01 worst-fold regression
bound. A guard must clear a 0.002 mean normalized improvement over the pressure
rule, a 0.01 worst-fold regression bound, and execute at least one held-out
override. The fixed minimum context trust is 0.1.

## Fallback semantics

Failure to certify a nonzero source returns the exact strict target-only
model. Failure to certify a guard returns the exact pressure prior; a disabled
guard never means raw learned execution. Only the selected 200-row guard fold
matrix is retained in the result JSON, rather than all intermediate model
predictions.

## Development and confirmation boundary

LA and Jinan are used first to verify that the historical Jinan positive effect
can be recovered without the target-closed-loop-developed v93 guard. Chicago
is the prospective city: protocol thresholds must be frozen before any Chicago
closed-loop control result is observed.
