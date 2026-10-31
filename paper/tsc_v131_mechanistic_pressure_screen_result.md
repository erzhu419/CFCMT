# TSC V131 Mechanistic Pressure Screen Result

## Decision

`REJECT`. The low-dimensional generalized-pressure family did not pass the
frozen absolute held-out gate, so no source prior or source-placebo successor
is authorized from V131.

## Evidence

- Scheduler task: `t88712` on `node005`
- Runtime: 101.27 s
- Immutable snapshot: `01377cf6ec2bc9f853ef`
- Result:
  `cf_h2o/results/cluster/tsc_v131_mechanistic_pressure_screen_20260901/development_v1/result.json`
- Result SHA-256:
  `f379ade41bad5caba0d532808b826630cfb8b0574f9eacab3f9541dd2a680b42`

The selector chose a rule in every outer fold, most often
`spillback_pressure` (9 folds) or `gp_d1_o0p02_s0` (6 folds). None of the 22
profiles passed the disjoint ten-seed calibration gate, so every held-out fold
fell back exactly to PhasePressure and the aggregate effect was zero.

Several training profiles appeared beneficial but reversed sign on calibration
seeds. For example, selected `gp_d0p65_o0_s0` profiles reached training means
near `-0.0012` yet calibration means were positive in affected folds. This is
evidence of seed instability, while the 11/10 split also leaves open whether
the development sample was used inefficiently.

## Consequence

V132 keeps the same 31 rules, six retention fractions, pressure-safety
constraint and outer held-out seeds. It replaces the 11/10 split with all 21
development seeds and a one-sided max-t simultaneous confidence bound over all
186 correlated profiles. The V132 protocol is frozen before its result is
observed. Failure closes this generalized-pressure intervention family rather
than authorizing source-weight tuning.
