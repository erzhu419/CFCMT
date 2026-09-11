# V150N Pressure-Pairwise Source Utility Protocol

## Question

Can source-mechanism priors identify when the PhasePressure action is preferable
to the target-only rigid action, under the same B100 labels and nested utility
gate used by V150K?

## Candidate Construction

For each state and mechanism candidate, only two actions are considered:
`a_R`, selected by rigid CFCMT, and `a_P`, selected by PhasePressure. The source
model contributes the pairwise score difference
`score_source(a_R) - score_source(a_P)`.

If the difference is positive, the candidate proposes `a_P` with that margin.
Otherwise it returns the exact rigid score vector. Every non-reference third
action is therefore unreachable. The matched-placebo arm receives the same
transformation before record construction and gate fitting.

## Frozen Evaluation

The target adaptation budget is B100, with five outer folds and four inner
folds. The utility model, uncertainty margin, minimum 48 records, minimum five
interventions, source-null contract, and city-level admission thresholds remain
unchanged. Evaluation uses the same untouched reserve as V150K/V150M.

V150N is rejected unless at least two cities pass all nested source-versus-rigid
and source-versus-placebo checks. Only a pass authorizes closed-loop smoke on a
new development seed.

## Boundary

This is a sequential seven-city development experiment motivated by the frozen
V150M support failure. It is not an independent confirmation.
