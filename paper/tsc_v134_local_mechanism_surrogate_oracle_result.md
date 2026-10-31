# TSC V134 Local-Mechanism Surrogate Oracle Result

## Decision

`REJECT` under the preregistered development gate. No one-step surrogate
profile satisfied all four requirements: at least 40 interventions, mean
normalized waiting effect at most `-0.0005`, a negative familywise simultaneous
95% upper bound, and improvement in at least 80% of held-out seeds.

This is an oracle proxy-validity screen, not a deployable controller result.
The action selector used the candidate action's realized one-control-interval
mechanism outcomes from the same simulator branch. It did not use the
450-second waiting label to select actions, but these realized mechanism
outcomes are unavailable before deployment.

## Evidence

- Scheduler task: `t88961` on `node005`
- Runtime: 163.77 s
- Immutable snapshot: `35c367707a86ee60774e`
- Result:
  `cf_h2o/results/cluster/tsc_v134_local_mechanism_surrogate_oracle_20260901/development_v1/result.json`
- Result size: 431,416 bytes
- Result SHA-256:
  `5613ccdf2a6727769109b348cfd14bef1b766dc68bea79ddd0fa76a8cccfda0e`
- Familywise critical value: `2.773459`
- Eligible profiles: 0/48

The strongest preregistered surrogate was `balanced_physical`, a normalized
combination of queue propagation, served movement, red accumulation,
spillback and mobility:

| Retained actions | Interventions | Mean effect | Simultaneous upper 95% | Improving seeds | Gate |
|---:|---:|---:|---:|---:|:---|
| 5% | 173 | -0.0007069 | -0.0000349 | 77.3% | reject |
| 10% | 334 | -0.0013154 | -0.0001973 | 68.2% | reject |
| 20% | 658 | -0.0009507 | +0.0006137 | 59.1% | reject |

The 5% and 10% profiles cleared the intervention-count, mean-effect and
familywise-bound requirements but failed the frozen 80% seed-consistency
requirement. The gate must not be relaxed after observing this result.
Single-component proxies were materially weaker: served-movement, spillback
and mobility objectives were harmful on average at their best retained
fractions; queue propagation and queue-plus-red objectives had negative point
estimates only at wider, non-significant familywise bounds.

## Interpretation

The realized one-step mechanisms contain some long-horizon control signal, but
their benefit is heterogeneous across stochastic realizations. This rules out
training a global one-step surrogate ranker and presenting it as a stable
replacement for PhasePressure. It does not justify declaring the mechanism
family empty: the two balanced profiles give a predeclared, familywise-negative
aggregate signal whose failures must be explained by pre-action context before
any learned model is attempted.

## Consequence

V135 may test one new hypothesis only: whether a selector frozen from outer
development seeds can identify, from pre-action target observables, the
contexts in which the 5% or 10% balanced-mechanism intervention is stable.
The held-out seed's 450-second labels and realized one-step outcomes must not
enter that selector. If a target-only outer-fold selector cannot pass the same
absolute and familywise gate, close this mechanism-control branch and retain
CFCMT as a calibration-light transfer layer rather than a claim of universal
superiority over pressure control.
