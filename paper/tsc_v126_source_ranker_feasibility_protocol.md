# V126 Source-Conditioned Ranker Feasibility Protocol

## Motivation

V125 found substantial pressure-nondegrading oracle headroom but showed that
the V124 B500 guard selected harmful held-out actions. V126 tests whether the
problem is learnable from abundant target counterfactual labels before any
effort is spent on a few-shot version or runtime integration.

## Design

The fixed B500 target and source predictions from V123 are combined with the
declared action-contrast parent set. All arms use the same standardized feature
dimension and ridge family:

- `target_only` receives zero-valued source channels;
- `source_aligned` receives seven source mechanism predictions and fixed
  consensus statistics; and
- `source_placebo` receives the same source channels after deterministic
  within-seed whole-group permutation.

Only actions whose instantaneous service pressure is no lower than
PhasePressure are eligible. The outer split leaves one of 22 selector seeds out.
Within each outer training set, three seed-blocked folds select the ridge
penalty by policy value, not row MSE. The held-out seed contributes neither a
label nor a hyperparameter choice. Full unlabeled selector features are used
only to define deterministic feature scaling.

## Gate

The source-aligned arm must beat PhasePressure, target-only and source-placebo
by at least 0.0005 mean normalized cost, and every paired 95% bootstrap upper
bound must be below zero.

V126 deliberately uses 21 target counterfactual seed banks per outer fold. It
is an abundant-information upper-feasibility experiment, not few-shot target
adaptation and not confirmation. Passing only justifies a later target-budget
curve and conservative runtime gate.
