# TSC V150J Conditional Mechanism-Prior Result

## Decision

`REJECT`. The deterministic local right-of-way and neighboring-execution
interactions did not pass the frozen cross-fitted source-identification gate.
All seven cities therefore used the bitwise-exact rigid CFCMT fallback.

## Evidence

- Aggregate artifact:
  `cf_h2o/results/paper_artifacts/tsc_v150j_conditional_mechanism_prior.json`
- SHA-256:
  `a2498680f233285dd28b1c11b51a654f7c0a9d8b8da0e25ac5c8c5cb69a366db`
- Target information budget: 100 labelled action groups per city.
- Source-prior strength: `0.025`, inherited from V150I.
- Source admissions: `0/7` cities.
- No city regressed because every rejected city recovered rigid CFCMT exactly.

Every city had a candidate that passed the full-data mean filters, but none
survived the complete five-fold selector audit. In particular, Hangzhou,
New York, RESCO synthetic and Salt Lake City had negative held-out mean gain
versus rigid for their selected candidates, but their fold stability or matched
placebo comparison failed. Atlanta, Cologne and Ingolstadt did not improve the
rigid baseline in held-out mean.

## Interpretation

The result rejects the bounded hypothesis that adding deterministic local
right-of-way and neighboring execution interactions to the mechanism prior is
sufficient to make a city-level source choice deployable. It does not negate
the rigid/action-contrast architecture result. The next admissible development
step must change the supervision unit: estimate source-mechanism utility at the
decision-state level and cross-fit the complete proposal-plus-gate pipeline.
Further city-level representation or selector variants are not authorized by
this result.
