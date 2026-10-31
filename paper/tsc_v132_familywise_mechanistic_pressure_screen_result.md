# TSC V132 Familywise Mechanistic Pressure Screen Result

## Decision

`REJECT`. The statistically efficient successor also authorized zero outer
folds. The generalized-pressure intervention family is closed and may not be
continued by relaxing its gate or adding source weights.

## Evidence

- Scheduler task: `t88956` on `node006`
- Runtime: 177.47 s
- Immutable snapshot: `72ae8a3c6cad3b50c805`
- Result:
  `cf_h2o/results/cluster/tsc_v132_familywise_mechanistic_pressure_screen_20260901/development_v1/result.json`
- Result SHA-256:
  `f4f57f4aac0f31a7705652cf4c7f20b7b61f49fd1fde4697595d400d162d5007`

The correlation-aware max-t critical value was approximately 2.91, so the
failure was not caused by a Bonferroni-scale multiplicity penalty. The best
non-empty profiles usually proposed only 2--7 interventions across all 21
development seeds. Their mean effects averaged approximately `-0.000090`,
well below the frozen material-effect requirement of `-0.000500`, and no best
profile improved more than 29% of development seeds. No profile came close to
the minimum 80 interventions or 80% seed consistency.

## Consequence

V131 and V132 jointly reject both explanations that could have rescued this
family: neither disjoint calibration nor full-development familywise selection
finds a stable generalized-pressure override. The next experiment must test
whether the 450-second action effect is predictable across simulator
realizations at all. V133 therefore uses exact schedule/TLS/action identity
across 21 historical target seeds to predict a held-out seed. It is an
abundant-target identifiability screen, not a transfer or deployable-policy
result.
