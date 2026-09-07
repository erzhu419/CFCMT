# TSC V141 Boston Total-Budget Confirmation Protocol

## Status and purpose

This protocol fixes the next target city and analysis before any V140 outcome
is available. V141 asks whether the V140 causal source-weighting rule retains
relative value in Boston under the same total B25 target-label budget.

Boston is new to CFCMT model, source-weight and controller outcome development.
Its topology has been inspected and locally repaired solely to satisfy SUMO
safety admission. That engineering work is not efficacy evidence. Los Angeles
is not used because it was held out only from one selector stage and has been
observed repeatedly elsewhere in development.

## Non-substitution rule

Boston is the sole V141 target. V141 runs only if both conditions hold:

1. V140 B25 passes its frozen aligned-versus-target and
   aligned-versus-placebo gates.
2. Boston package v9 passes complete route, trigger-window and complete-day
   microscopic admission.

If either condition fails, V141 stops. Another city cannot be selected after
seeing those outcomes.

## Target information budget

The complete published Boston city, day and 3,806,510 routes are simulated with
libsumo. Before any counterfactual waiting outcome is generated, an unlabeled
snapshot inventory is partitioned by fixed time blocks and static TLS topology.
The inventory freezes:

- five adaptation partitions with five action groups each, exactly 25 groups
  in total;
- 17 disjoint evaluation partitions;
- group identities and candidate action sets using only time, topology, lane
  counts and candidate count.

No outcome, model prediction or source score may enter group selection. All
target-model fitting, source weighting and source authorization consume only
the same 25-group union. Evaluation labels remain inaccessible until every
model and weight is frozen.

## Frozen method and comparisons

V141 inherits V140 without Boston-specific tuning:

- the seven source domains and their order;
- target and causal source model architectures;
- nonnegative source weights summing to at most one, with explicit source-null;
- ridge penalty 0.05;
- whole-action-group source placebo;
- B25 and all efficacy thresholds.

Primary relative comparisons are source-weighted versus architecture-matched
target-only and source-weighted versus matched source placebo. Target-only and
source-weighted performance versus PhasePressure are reported separately and
cannot be inferred from a relative transfer gain.

The paired unit is the predeclared Boston spatiotemporal evaluation partition.
Confirmation requires mean gains of at least 0.0005 against both comparators,
negative one-sided 95% upper bounds, improvement in at least 80% of evaluation
partitions, and no collision or teleport regression.

## Claim boundary

V141 can confirm relative source value in one new target city under a strict
few-shot budget. It is not by itself a population-level multi-city estimate,
and it does not establish absolute superiority over PhasePressure unless that
separate absolute comparison also passes.
