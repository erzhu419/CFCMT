# TSC V142 Total-Budget Policy-Aligned Source Selection Protocol

## Status and purpose

This protocol is frozen after V140 and before running V142. V140 used five
adaptation seeds to estimate continuous non-negative source weights by row-level
squared error. The raw weights were non-zero, but their held-out policy value
was harmful and the gate correctly returned source-null. The frozen V140
evaluation also showed that a non-selected uniform-source descriptive arm could
outperform target-only. V142 therefore tests one specific explanation: the
source fitting objective was misaligned with the downstream argmin policy.

V142 is post-hoc Jinan method development. Its evaluation seeds have appeared
in V140 arm summaries, so V142 cannot be confirmation evidence regardless of
its result.

## Fixed target-information budget

The total target-label budget is B25: exactly 25 unique target action groups.
World-model fitting, source selection and source authorization share this same
set. The 22 frozen selector seeds are sorted numerically and assigned by
alternating position: 11 adaptation seeds at even zero-based positions and 11
evaluation-only seeds at odd positions. B25 groups are allocated as evenly as
possible across the 11 adaptation seeds using the existing deterministic
coverage-first order. Evaluation labels are excluded from every fit, candidate
choice and authorization decision.

## One-module change

Target-only and seven one-source CFCMT models retain the V140 causal families,
parent sets, source caches and source-null semantics. Only source selection
changes. The frozen candidate family contains:

- exact source-null;
- each of seven single-source models at mass 0.25, 0.50, 0.75 and 1.00;
- the uniform seven-source ensemble at the same four total source masses.

This yields 33 candidates. For every held-out adaptation seed, the candidate
with the lowest mean downstream policy value on the other ten seeds is chosen.
Ties prefer source-null. The selected candidate is then evaluated on the held-
out seed. A matched placebo applies the same selected weights to deterministic
whole-action-group-permuted source residuals; it does not receive an independent
candidate search.

## Frozen development gate

The all-adaptation refit must select a non-null candidate. Across the 11 nested
held-out folds, aligned source selection must:

1. improve target-only by at least 0.0005 on average;
2. have a negative one-sided 95% upper confidence bound;
3. improve at least 80% of adaptation seeds;
4. improve the matched placebo by at least 0.0005 with a negative one-sided
   95% upper confidence bound.

Failure returns exact target-only before evaluation. The 11 evaluation seeds
then report source-minus-target and source-minus-placebo paired effects without
changing the gate.

## Decision boundary

A pass authorizes a separately preregistered multicity development matrix that
tests the same discrete selector across complete-city holdouts. It does not
authorize V141, Boston efficacy evaluation, or an untouched-city claim. A fail
closes this finite policy-aligned candidate family under B25.
