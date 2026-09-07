# V119 Target-Calibrated Causal Source Gate

## Why V98 remains positive but insufficient

V98 established a real and reproducible comparison: source-selected CFCMT
improved the strict target-only model on all 56 evaluated pairs, with a mean
waiting-time change of -4.5274%. That result supports the claim that source-city
information can improve a target-offline model. It does not establish safe
improvement over the exact PhasePressure controller because the compared
target-only model did not hold the later architecture fixed and PhasePressure
was not the primary reference in that analysis.

V116 therefore tested a stronger question. Source predictions improved the
architecture-matched target-only score, but the old trust/weight gate admitted
too many harmful overrides and failed the absolute PhasePressure gate. V119
changes the admission mechanism while preserving the V98 evidence and the V116
estimand.

## Frozen estimand and information budget

- Estimand: `one-control-interval-local-mechanisms-plus-pure-halted-queue-rollout-value-v2`.
- Reference action: exact PhasePressure action in the same action group.
- Label: candidate-minus-PhasePressure 450-second pure halted-queue cost,
  normalized by the action-group cost range.
- Target calibration: exactly the 100 Jinan action groups frozen by V115,
  drawn from seeds 5057 and 6067.
- Source predictor: V115 B0 causal source components, which consume no target
  transition labels.
- Development selector: the 22 disclosed V116 seeds. Their outcomes are not
  used to fit the gate model.

## Architecture-matched comparison

All three arms use the same local causal features, standardized ridge family,
five group-disjoint cross-fit folds, selected ridge penalty, B100 labels and
feature dimension.

1. **Target-only:** source slots are fixed to their neutral value.
2. **Source-aligned:** source slots contain city-permutation-invariant score,
   dispersion, support and upper-bound aggregates from all source components.
3. **Source placebo:** the same source blocks and marginal distributions are
   retained, but blocks are reassigned across action groups before fitting and
   evaluation.

The placebo distinguishes useful source-target alignment from a gain caused
only by extra nonzero regressors or model capacity. Source city identifiers are
never input features.

## Admission rule

The model predicts normalized action regret relative to PhasePressure. A
non-reference action is eligible only when its one-sided group-level conformal
upper bound is below zero by the frozen minimum-gain threshold. The aligned
source arm additionally requires at least half of the source components to
prefer the candidate to the reference.

Profiles are selected by nested leave-one-seed evaluation. A source profile
must simultaneously satisfy the existing V116 bounds on mean improvement,
worst-seed regression, harmful overrides and intervention frequency, and must
show at least 0.0005 mean normalized improvement over both its matched
target-only profile and its matched source-placebo profile. Family-wise upper
confidence bounds for all three comparisons must be non-positive.

## Claim boundary

V119 is a development experiment. Passing permits one frozen fresh-city
confirmation; it is not itself confirmatory evidence. Failure retains the
PhasePressure fallback and shows that the current source information is not
deployable under this admission protocol. Failure does not invalidate V98's
narrower source-versus-strict-target-only result.
