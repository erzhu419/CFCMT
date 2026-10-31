# TSC V135 Context-Gated Mechanism Oracle Protocol

## Purpose

V134 found familywise-negative aggregate effects for the balanced one-step
physical proxy at 5% and 10% retention, but those profiles improved only 77.3%
and 68.2% of seeds and therefore failed the frozen 80% consistency gate. V135
tests one explanation before any new world model is fitted: whether the harmful
realizations can be recognized from state and action information available
before intervention.

## Frozen nested protocol

The V134 `balanced_physical` oracle remains the proposal generator, including
the pressure-nondegradation constraint and seed-local 5% or 10% retention. For
each outer held-out seed, five disjoint seeds are reserved for authorization
and the remaining 16 development seeds fit a small, strongly regularized
context-effect regressor. Its inputs are only the existing causal-core local
state, deterministic graph and candidate-action parents. Realized one-step
outcomes, the oracle advantage and 450-second outcomes are not numeric context
features.

Within the 16 fitting seeds, leave-one-seed-out predictions estimate a frozen
90th-percentile absolute error. The regressor's predicted 450-second effect
plus 0, 0.5 or 1.0 times that OOF error defines six predeclared context gates
across the two base retention rates. The five authorization seeds choose a
profile only if it makes at least 20 interventions, has mean effect at most
`-0.0005`, has a negative one-sided 95% upper bound and improves at least four
of five seeds. The held-out seed is used once for scoring.

## Decision

The single nested algorithm advances only if its 22 held-out effects include
at least 40 interventions, have mean at most `-0.0005`, bootstrap upper 95%
below zero and improve at least 80% of seeds. Passing licenses V136 to replace
the oracle proposal with a learned balanced-mechanism predictor. Failure closes
the one-step mechanism-control branch.

## Claim boundary

V135 is still an oracle: the evaluated seed's realized one-step outcomes decide
which action is proposed and which proposal falls in its seed-local retention
set. The context gate does not consume those values numerically and does not
use held-out 450-second labels, but the complete procedure is not deployable,
zero-shot or transfer evidence.
