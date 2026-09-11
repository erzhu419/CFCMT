# TSC V150A Source-Benefit Repeatability Diagnostic

## Purpose

V144 showed that every development target had at least one source with a
favorable pooled effect, while V145, V146 and V148 failed to identify a
deployable source safely. V150A tests the intermediate question: does the
source benefit selected on some random realizations repeat on a held-out
realization?

## Inputs

- The seven immutable V144 target result JSON files.
- Three evaluation seeds per target.
- Six ordered source candidates per target.
- Architecture-matched target-only and matched source-placebo arms.
- The original B25 target adaptation budget.

No model is refit and no SUMO trajectory is regenerated in V150A.

## Analysis

For every target and source, V150A records the per-seed effect relative to
target-only, source-rank correlation between each seed pair, and benefit-sign
agreement. For each held-out seed, the source with the lowest mean effect on
the other two seeds is fixed and evaluated on the held-out seed. A second
version selects source-null whenever the training-seed mean of the best source
is nonnegative.

The matched-placebo comparison follows the same fixed source identity.

## Information Boundary

This is a development diagnostic, not a deployment gate. Source selection uses
policy outcomes from the other evaluation seeds. Those outcomes are not
available to a B25 target at deployment. The held-out seed outcome is excluded
from its own choice, but the fitted target models may use the already declared
B25 adaptation groups from all three scenario seeds.

V150A can justify collecting fresh development seeds and constructing a
deployment-observable utility target. It cannot establish source transfer,
authorize Boston evaluation, reopen V149, or serve as fresh-city confirmation.
