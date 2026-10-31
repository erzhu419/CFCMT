# V148 Multicity Mechanism-Compatibility Gate Protocol

## Scientific question

V148 tests whether labeled target-offline next-state transitions can identify
which source city's rigid causal residual is useful. It does not revive the
rejected V134--V135 branch in which a one-step mechanism model selected the
traffic-signal action. The downstream action scores remain exactly the V143
target-only and one-source rigid causal rankers.

## Information budget

Each of the seven development cities retains the frozen B25 adaptation budget
and evaluation split from V143. The 25 adaptation action groups form five
deterministic folds. Each fold fits on B20 and scores the disjoint B5. V148 uses
the five B5 next-state labels, so it is **target-offline few-shot adaptation**,
not target-label-free or zero-shot transfer. Evaluation groups are absent from
all fitting, mechanism scoring, source selection and source-mass estimation.

## Mechanism probe

The probe contains five physical one-step mechanisms:

1. queue propagation;
2. served movement;
3. red accumulation;
4. spillback; and
5. mobility.

Every mechanism uses its previously declared `physical` parent set. A fixed
rank-0, domain-balanced robust residual estimator is used for both target-only
and source-plus-target fits. There is no cross-city hyperparameter selection,
latent factor, direct context parent or dense residual in this probe.

For mechanism `m` and source `s`, the five held-out folds yield group-equal
mean-squared losses `L_target,m` and `L_source,m`. The frozen trust is

`clip((L_target,m - L_source,m) / max(L_target,m, 1e-12), 0, 1)`.

The source mass is the equal mean of the five mechanism trusts. Mechanism
predictions never enter the action score. They only shrink the already fitted
rigid source residual:

`score_s = score_target + mass_s * (score_source - score_target)`.

The matched placebo uses the identical mass applied to the frozen
whole-action-group source-residual permutation.

## Leave-one-city source selection

The seven target results are aggregated with the evaluated target city absent
from every source-history label, including rows where that city would act as a
source. For candidate source `s`, history also excludes the row where `s` is
the target. Thus each history contains five other target cities.

A source is admitted only when, against both target-only and its matched
placebo, the history maximum regression is at most `0.005` and the history 75th
percentile effect is at most `-0.0005`. If no source is admitted, the decision
is exact source-null. Among admitted sources, select the largest current-target
mechanism mass, then the better historical target-only mean, then lexical source
name. No minimum mass or fallback source is tuned after outcomes.

## Development gate

The selected V148 arm must pass all three comparisons against target-only,
pooled H2O+ and the selected matched placebo. Each comparison requires:

- mean seven-city effect at most `-0.0005`;
- one-sided 95% upper bound below zero;
- improvement in at least five of seven cities; and
- maximum city regression at most `0.01`.

Failure closes this mechanism-compatibility family without threshold retuning.
A pass authorizes only the separately registered V149 Boston protocol.

## Claim boundary

V148 can establish that labeled target-offline mechanism risk improves source
selection for a rigid causal transfer layer. It cannot establish a full MC-WM
controller, zero-shot transfer, real-world counterfactual efficacy or universal
superiority to PhasePressure.
