# TSC V132 Familywise Mechanistic Pressure Screen Protocol

## Purpose

V131 split each outer development set into 11 rule-selection seeds and 10
authorization seeds. Several selected rules changed sign between those halves,
and no fold was authorized. V132 determines whether this was only a loss of
statistical power or a genuine lack of stable absolute improvement.

## Frozen selection protocol

The action family is unchanged: 31 previously declared generalized-pressure
rules, six previously declared retention fractions, and the constraint that an
eligible action cannot reduce instantaneous service pressure relative to exact
PhasePressure. No source artifact is read.

Each of 22 outer folds reserves one selector seed for final scoring. All 21
remaining seeds evaluate the 186 fixed profiles. A Gaussian-multiplier
one-sided max-t bound at familywise alpha 0.05 accounts for selecting among the
correlated profiles. A profile is authorized only when it has at least 80
development interventions, mean normalized waiting improvement of at least
0.0005, a simultaneous upper bound below zero, and improvement on at least 80%
of development seeds. The outer seed is never read during selection or
authorization.

## Decision

V132 passes only if the 22 outer held-out effects have mean normalized waiting
delta at most `-0.0005` and upper 95% bound below zero. Passing authorizes one
source-prior successor with a group-permuted source placebo. Failure closes the
generalized-pressure intervention family; the source mechanism cannot rescue a
target-only controller that fails this absolute gate.
