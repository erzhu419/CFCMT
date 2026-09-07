# TSC V134 Local Mechanism Surrogate Oracle Protocol

## Purpose

V133 shows that the direct 450-second action effect is not reproducible across
simulator realizations at exact schedule/TLS/action identity. V134 moves below
that noisy label and asks which one-control-interval physical outcome, if known
perfectly, would select actions that improve 450-second waiting.

## Frozen surrogate family

The eight predeclared proxy costs are one-step total queue, served-movement
queue, red queue, downstream occupancy, negative mean speed, a queue/red blend,
the existing causal-physical blend, and an equalized physical blend. Every
component is normalized within its matched action group before combination.
Candidate actions cannot reduce instantaneous service pressure relative to
PhasePressure.

For each proxy and each of six predeclared retention fractions, the same-seed
one-step counterfactual labels choose an action and the independent 450-second
halted-waiting target scores it. This intentional oracle use is disclosed and
cannot support a policy claim. Across the resulting 48 profiles, a Gaussian
multiplier one-sided max-t bound controls familywise selection over 22 seeds.

## Decision

A proxy advances only with at least 40 interventions, mean normalized waiting
delta at most `-0.0005`, a simultaneous upper 95% bound below zero, and
improvement in at least 80% of seeds. Passing licenses a successor that predicts
only the selected local mechanism. Failure means none of the available
one-step mechanisms is a valid action surrogate for the 450-second objective;
training another model on them is then unjustified.
