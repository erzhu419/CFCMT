# V123 Target-Budget Source-Value Curve Protocol

## Purpose

V98 reported a positive source-selected effect relative to its strict target-
only comparator, whereas V119--V122 did not recover a safe architecture-matched
B100 source effect. V123 tests whether this disagreement is caused by target-
data budget: source information may help when target evidence is scarce and
become redundant or harmful after the target model is sufficiently informed.

## Frozen design

The target budgets are B0, B25, B50, B100, B250, B500 and B1000. Positive
budgets use one deterministic scenario- and adaptation-seed-balanced coverage-
first order. The sets are strictly nested, and B100 must exactly equal the V115
group identities. Target-only and each one-source-at-a-time CFCMT component use
the same anchored model families and exactly the same target groups.

All models are evaluated on the unchanged 22-seed Jinan selector cache used by
V116 and V121. No selector label is used for fitting a world model. Candidate
source city and target/source blend weight are selected by nested leave-one-
selector-seed-out evaluation, so the held-out seed cannot select its own source.

## Decision statistic

For each positive budget, the primary source statistic is the per-seed paired
difference between nested source selection and the architecture-matched target-
only model. V123 reports a deterministic 10,000-replicate paired bootstrap. A
budget exposes source signal only if the mean contribution is at most -0.0005,
the 95% upper bound is below zero, and the selected source policy also beats
PhasePressure on the offline selector estimand.

V123 is development evidence. Passing authorizes a budget-specific guarded
successor and then one frozen fresh-city confirmation; it is not itself a
cross-city confirmation.
