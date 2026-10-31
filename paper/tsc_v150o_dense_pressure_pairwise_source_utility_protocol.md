# V150O Dense Pressure-Pairwise Source Utility Protocol

## Question

Does source-mechanism evidence improve the decision between the target-only
rigid action and PhasePressure when the utility learner receives a fixed,
dense B100 counterfactual label for that action pair?

## Change From V150N

V150N created utility records only when a source candidate already proposed
PhasePressure. Its nested folds contained at most 20 such records per city and
therefore could not fit the frozen gate. V150O does not lower the 48-record
minimum. For every adaptation state where rigid and PhasePressure differ, every
source-mechanism candidate receives the same target action-pair label,
`cost(PhasePressure) - cost(rigid)`. Source scores contribute only observable
pairwise evidence and confidence features.

The candidate action remains binary: exact rigid fallback or the fixed
PhasePressure reference. No third action is executable.

## Controls

Three separately nested comparisons use the same B100 folds and utility model:

- target-only rigid CFCMT;
- a matched placebo whose source mechanism labels are group-permuted;
- a source-blind utility gate with the same target labels, state features,
  mechanism indicators and model capacity, but all source corrections removed.

The source-blind arm isolates the benefit of the new target-side utility
representation from the benefit of cross-city information.

## Frozen Gate

Each city must improve all three controls by at least `0.0005` across the five
outer folds, be non-degrading on at least four folds against each control, and
make at least five cross-fitted interventions. At least two cities must pass;
all rejected cities recover rigid CFCMT exactly. Only a full pass authorizes a
closed-loop smoke on a new development seed.

## Boundary

This is a sequential seven-city development experiment after the frozen V150N
rejection. It is target-offline adaptation, not zero-shot, closed-loop evidence
or fresh-city confirmation.
