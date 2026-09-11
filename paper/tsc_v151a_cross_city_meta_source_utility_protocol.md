# V151A Cross-City Meta Source Utility Protocol

## Question

V151A asks whether source mechanism evidence can predict a useful
PhasePressure-versus-rigid deviation when the utility relation has never used
the evaluated target city. It follows V150O, which repaired record density but
showed that a target-local utility gate could confuse source value with a
target-only state predictor.

## Data Roles

For each target city, the other six cities act as pseudo-target utility
domains. Each pseudo-target fits its rigid and mechanism-prior candidates on a
fixed B100 adaptation subset and supplies fixed-pair utility labels from the
remaining groups. The evaluated target city is excluded from all pseudo-target
utility labels, source candidates, feature scaling statistics and selector
calibration.

The actual target contributes B100 labels only to fit the rigid target model
and mechanism coefficients. Its utility label count is zero. Thus V151A is
few-shot target mechanism adaptation with target-label-free cross-city utility
transfer, not full zero-shot control.

## Model And Controls

The utility gate is the frozen 24-feature ridge upper-cost-bound model from
V150O. Records are weighted first equally across pseudo-target cities and then
equally across action groups. Source, matched-placebo and source-blind gates are
fitted independently with identical target-side labels and capacity.

The source gate is admitted for a target only if leave-one-pseudo-target-city
evaluation over the six source cities has mean gain of at least `0.0005`
against rigid, placebo and source-blind; is non-degrading in at least five of
six cities for every comparison; and makes at least six interventions. A
rejected gate returns the rigid score exactly.

## Development Gate

The seven-target aggregate requires at least two source-only meta admissions,
no target-city regression, every admitted target to improve rigid and beat both
source controls, and exact fallback elsewhere. Thresholds are fixed before
V151A target results. Passing authorizes closed-loop development only; it does
not establish untouched-city efficacy.
