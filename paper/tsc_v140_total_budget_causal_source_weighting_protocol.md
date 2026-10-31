# TSC V140 Total-Budget Causal Source Weighting Protocol

## Question

Does source-city causal mechanism information improve an architecture-matched
target model when every use of target labels, including source weighting, is
charged to one fixed target-information budget?

V123 established a real relative source effect, strongest at B25, but its
nested source choice used the 21 non-held-out selector seeds. B25 therefore
described only the world-model fitting budget, not the total target-label
budget. V140 removes that ambiguity.

## Frozen information budget

The development budgets remain B25, B50, B100, B250, B500 and B1000. The old
target-adaptation cache contains only two simulator seeds, which is insufficient
for a seed-held-out source-weight gate. V140 therefore partitions the already
frozen 22-seed Jinan bank before fitting: the five numerically smallest seeds
are assigned to adaptation and the remaining 17 seeds are evaluation-only.
This label-independent rule selects adaptation seeds 22208, 27178, 31657,
34310 and 46535. Within them, groups follow one strictly nested,
seed-balanced coverage-first order. Exactly B unique action groups form the
entire target-label budget.

For each budget, the five target adaptation seeds form an inner
leave-one-seed-out protocol. Target-only and every one-source CFCMT model are
fitted on four adaptation seeds and predict the fifth. These predictions create
one complete OOF table over exactly B target groups. The final selector models
are then refitted on all B groups after weighting is frozen. Target-only and
source-weighted arms therefore receive the same total target groups.

## Causal source weighting

For each action row, the source channels are the seven causal source-model
scores minus the architecture-matched target-only score. A non-negative ridge
fit estimates source weights under the constraint that their sum is at most
one; the unassigned mass is the explicit target-only source null. Rows receive
equal action-group and equal adaptation-seed weight.

Weight evaluation is nested again: for each held-out adaptation seed, weights
are fitted only on the other four seeds' already-OOF predictions. The same
procedure is applied to a deterministic whole-action-group source permutation
that preserves scenario, adaptation seed, action count and source marginals.
The aligned source channel is authorized only when its five held-out seed
effects:

1. improve on target-only by at least `0.0005` on average;
2. have a negative one-sided 95% upper bound;
3. improve at least four of five adaptation seeds;
4. improve on the matched source placebo by at least `0.0005` with a negative
   one-sided upper bound.

Otherwise all source weights are set to zero before selector evaluation.

## Selector evaluation

The 17 seeds not assigned to target adaptation are evaluation-only. Every
budget reports:

- target-only minus PhasePressure;
- source-weighted minus PhasePressure;
- source-weighted minus target-only;
- source-weighted minus matched placebo;
- uniform-source and best-single-source descriptive baselines;
- learned source weights and the fraction assigned to source-null.

B25 is the predeclared primary relative-transfer test because it was the
strongest V123 budget. The other budgets form a descriptive adaptation curve
with a joint max-t interval. Absolute PhasePressure performance remains a
separate estimand and cannot be inferred from a negative source-minus-target
effect.

## Successor and confirmation

Only a B25 source-weighted arm that passes both aligned-versus-target and
aligned-versus-placebo held-out selector tests may advance. Its weighting
protocol and all thresholds are then frozen for one untouched target city.
The confirmation city cannot select a new source family, ridge penalty, budget
or intervention rule.
