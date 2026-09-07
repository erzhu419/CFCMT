# V127 Target-Information Budget Audit

Date: 2026-09-01

## Purpose

This note fixes the information-budget interpretation before any V127 outcome
is used to design a target-budget curve. It is an audit, not a preregistered
successor protocol.

## What V127 consumes

V127 has three disjoint target-label roles in each outer fold:

1. The B500 target prior was fitted on 500 matched target action groups from
   adaptation seeds 5057 and 6067.
2. The causal pairwise ranker is fitted on 16 V116 selector seeds. Depending
   on the held-out fold, this is 14,195 to 14,654 matched target action groups
   (mean 14,437.09).
3. The intervention-retention profile is calibrated on five different V116
   selector seeds. Depending on the fold, this is 4,368 to 4,744 matched
   target action groups (mean 4,511.59).

The held-out selector seed contributes 728 to 1,017 action groups and is used
only for that fold's final evaluation. Across all 22 V116 seeds, the selector
bank contains 19,851 groups and 158,808 action rows. These counts are derived
from the admitted V116 selector audit with SHA-256
`ab8b85009c8f80e48ce2f8a4ca0e415c6c4a8742dfab7a9ca83ab76388565d42`.

V127 therefore tests abundant-target learnability. Its `B500` label describes
only the upstream target-prior model and must not be reported as the total
target information budget. V127 is neither few-shot adaptation nor a fresh
city result.

## Required budget contract for a successor curve

For a curve with declared budget `B`, `B` must equal the number of unique
matched target action groups whose outcome labels are available anywhere in
the complete method. This union includes groups used by:

- the target mechanism or prior model;
- the pairwise action ranker or target residual;
- source selection or source weighting;
- uncertainty, retention-threshold, and deployment-gate calibration.

Reusing one labelled group in several cross-fitted modules counts once, but no
additional target-labelled group may be hidden behind a pretrained artifact.
Source-city groups are reported separately and do not count toward `B`.

The target groups must be nested across budgets and selected before looking at
the V116 selector outcomes. Group-disjoint cross-fitting inside the same `B`
groups may produce out-of-fold predictions for source weighting and safety
calibration. The final model may then refit on all `B` groups. V116 selector
labels remain evaluation-only for the development curve, and a later city
must remain completely untouched until the method and one budget are frozen.

The zero-target point has `B=0`: no target transition or counterfactual action
outcome may enter any model or gate. Static network information and online
state remain allowed and must be listed separately.

## Outcome-contingent next step

- If the V127 source-aligned arm passes its frozen absolute, target-only, and
  placebo gates, freeze the ranker family and implement the total-budget
  contract above.
- If target-only passes but source-aligned fails, do not run a cosmetic
  prior-budget curve. Replace the source-as-an-extra-feature construction with
  a source-pretrained pairwise prior plus a target residual or source-null
  weighting gate, then repeat an abundant-information feasibility test.
- If all arms fail, reject this pairwise ranker family and inspect its
  held-out calibration profiles before specifying another model.

Only the first case authorizes a target-budget curve under the current V127
family. No case authorizes untouched-city confirmation until a successor
development gate passes.
