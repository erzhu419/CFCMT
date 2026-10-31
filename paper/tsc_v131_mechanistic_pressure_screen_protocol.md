# TSC V131 Mechanistic Pressure Screen Protocol

## Purpose

V130 showed that 117-feature black-box intervention models do not generalize
reliably across selector seeds. V131 tests whether the same target labels can
identify a low-dimensional, network-invariant traffic mechanism before source
transfer is reintroduced.

## Frozen mechanism family

The family is the existing 31-member generalized-pressure grid. Its only
parameters are downstream queue weight, downstream occupancy weight and switch
penalty, plus one spillback-pressure form. Candidate actions must not reduce
instantaneous service pressure relative to exact PhasePressure.

For each outer fold, 11 target selector seeds choose one rule and one of the
predeclared retention fractions. Selection minimizes the one-sided training
seed upper bound among profiles with at least 40 interventions, negative mean
effect and improvement in at least 60% of training seeds. Ten disjoint seeds
then apply the frozen absolute calibration gate. The final seed is read only
for held-out scoring.

## Decision

V131 passes only if the 22 held-out seed effects have mean normalized waiting
delta at most `-0.0005` and upper 95% bound below zero. Passing authorizes a
separate experiment in which source-city evidence acts only as a prior over the
same three mechanism coefficients and is compared with a group-permuted source
placebo. Failure closes this mechanism family without source-weight tuning.
