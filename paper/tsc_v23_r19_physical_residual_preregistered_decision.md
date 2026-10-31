# TSC v23/r19 Physical-Residual Preregistered Decision

Date: 2026-08-08

## Motivation and fixed change

The v22/r18 audit failed. A code-to-paper estimand audit also established that
the legacy mechanism branch replaced every simulator prior with zero before
fitting, so it was a direct normalized outcome model rather than the claimed
analytic-simulator residual model.

v23/r19 makes one primary methodological change: local state, analytic prior,
and observed next state are expressed in target-label-free portable physical
units, and each mechanism fits `observed - analytic prior`. Queue quantities
are per controlled lane, occupancy remains a `[0,1]` lane fraction, and speed
is relative to 13.89 m/s. The first candidate is rank 0 only. Latent mechanism
adaptation is not eligible until rank 0 passes the breadth gate below.

## Pilot used to define this gate

One zero-shot pilot held out the complete RESCO-synthetic city group and
evaluated 648 cached grid4x4 action groups. It reduced mean normalized action
regret from 0.3453 for rigid CFCMT to 0.2444 and raised optimal-action rate from
56.3% to 67.7%. This pilot is development evidence only and cannot satisfy the
six-city breadth gate.

## Stage A: six-city zero-shot breadth screen

Use one preregistered representative scenario from every city group:

- RESCO synthetic: `grid4x4`
- Cologne: `cologne1`
- Ingolstadt: `ingolstadt1`
- Atlanta: `atlanta_1x5`
- Hangzhou: `hangzhou_4x4`
- New York: `manhattan_28x7`

For each target, hold out every scenario in its city group from fitting and
source policy selection. Use zero target adaptation groups. Evaluate all
matched target action groups from the frozen complete cache. Compare:

1. `causal_rigid_advantage`
2. `cfcmt_physical_mechanism`
3. `cfcmt_physical_fused` (must be identical to the mechanism model at budget 0)

The screening fitter retains the model's nested source-city selection but
omits deployment guard/regularizer source-LOO, because neither quantity is
used by offline action regret. This is a speed optimization, not a different
predictor. On the pilot, optimized and unoptimized predictions, mechanism
selection, and evaluation metrics were exactly equal while wall time fell from
498.8 s to 248.2 s.

Rank-0 physical residual passes Stage A only if all conditions hold:

- mean regret across the six targets improves by at least 10% relative to rigid;
- at least four of six target cities improve by more than numerical tolerance;
- no target's absolute normalized-regret increase exceeds 0.05;
- every target audit confirms zero target adaptation groups and full-city holdout;
- all outputs are finite and every family evaluates the same action-group set.

## Stage B: all-network and budget progression

If Stage A passes, evaluate all 16 scenarios at target budgets
`0, 16, 60, 120`. The target adaptation/calibration split remains disjoint,
and evaluation excludes every selected target group. Report city-group macro
means in addition to scenario-weighted means.

Rank-1/2 latent variants are eligible only after the rank-0 all-network result
is frozen. A latent variant may replace rank 0 only if source-only selection
improves held-out-city regret in at least four city groups, improves the macro
mean, and does not worsen the worst city by more than 0.02.

## Stage C: closed-loop development and confirmation

Only a Stage-B-qualified model enters 600 s closed-loop development. It must be
paired with a separately calibrated safety guard and compared against the
selected source prior, MaxPressure/PhasePressure, rigid CFCMT, target-only,
dense residual, and simulator-only policies under identical signal execution.
Promotion to a disjoint 3600 s confirmation requires:

- paired mean improvement over the selected source prior;
- non-positive city-group median regression;
- improvement in at least four of six city groups;
- top-gain city share below 70%;
- collision-incident noninferiority and zero teleports;
- no use of confirmation seeds during model, guard, or policy selection.

Failure at any stage is retained as a negative ablation. Gates will not be
relaxed after observing the corresponding result.
