# TSC V150C Mechanism-Parameter Prior Protocol

## Question

V150A found repeatable source headroom, whereas V150B rejected a city-level
source-utility representation. V150C tests the narrower hypothesis that source
information becomes useful when it is transferred as an action-conditioned
coefficient prior for one causal mechanism rather than as a whole-city residual.

## Frozen design

- Development cities: Atlanta, Cologne, Hangzhou, Ingolstadt, New York,
  RESCO synthetic, and Salt Lake City.
- Target information: the existing 25 action groups, split into five equal OOF
  folds for source/mechanism selection.
- Evaluation: every remaining target action group; no evaluation outcome enters
  fitting, scaling, or selection.
- Candidate family: six non-target source cities crossed with six predeclared
  mechanism blocks: queue service, spillback, mobility, execution, network
  propagation, and right of way.
- Representation: deterministic action contrasts, including protected versus
  permissive movement service, shared receiving lanes, uncontrolled-major merge
  exposure, and matched neighboring-signal execution state.
- Estimator: group-balanced ridge with fixed `l2=0.05`; only the selected block
  is shrunk toward one source coefficient vector with fixed strength `0.10`.
- Placebo: complete source action groups are permuted and re-centered so the
  PhasePressure reference remains exactly zero.
- Fallback: source prior strength zero must produce bitwise-identical
  coefficients, scores, and summaries to the same-architecture target-only arm.

The OOF selector admits a source only when its mean B25 gain is at least
`0.0005`; otherwise it returns target-only exactly. This is target offline
adaptation/few-shot evaluation, not zero-shot transfer.

## Development gate

Promotion to rigid-CFCMT integration requires all of the following:

1. Negative mean selected-source effect relative to target-only.
2. A paired bootstrap 95% upper bound below zero.
3. Improvement in at least five of seven cities.
4. Negative mean effect relative to the matched-placebo selector.
5. Worst city regression no larger than `0.01`.
6. At least five cities admitting a non-null source prior.

Failure closes this mechanism-prior route. Passing only authorizes integration
and closed-loop development; it is not fresh-city confirmation.
