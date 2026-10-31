# V157A Feature-Aligned V123 Source-Value Replay

## Question

Does V123's architecture-matched Jinan source contribution survive the V154
feature-binding correction?

## Frozen replay

V157A reruns the complete V123 target-budget curve from its original immutable
snapshot. It retains the source and target caches, Jinan B25--B1000 nested
groups, 22 selector seeds, seven source groups, blend grid, model settings,
10,000 paired bootstrap replicates and all original gates. The only code change
resolves `PairwiseActionAdvantageRegressor` inputs from the fitted model's
stored feature names instead of reusing column positions from its training
dataset.

The full curve is replayed because the affected rigid anchor participates in
both the architecture-matched target-only arm and every source-augmented arm.
All cached 450-second counterfactual outcomes are reused; V157A runs no SUMO
simulation and keeps its large prediction artifact on the server.

## Decision

The original V123 absolute deployment gate is reported unchanged. Separately,
B100 may authorize writing and freezing a same-state 450-second Jinan branch
protocol only if corrected source-minus-target mean is at most `-0.0005` and
its paired 95% upper bound is below zero. This does not authorize a full
closed-loop controller or an external city.

If B100 fails, the branch experiment is not run because its source-value premise
has disappeared. If it passes, a separate protocol must freeze the deployable
B100 source candidate and exact checkpoints before any run. Its four branch
arms are the corrected V123 B100 architecture-matched target-only action, the
frozen corrected V123 B100 source-augmented action, a matched source-label
placebo action and PhasePressure, each followed by the same PhasePressure
continuation for 450 seconds.

## Boundary

V157A is a correction-only development replay. It cannot establish fresh-seed,
unseen-city, closed-loop or PhasePressure-superiority claims.
