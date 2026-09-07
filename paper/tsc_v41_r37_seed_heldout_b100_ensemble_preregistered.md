# TSC v41/r37 Seed-Heldout B100 Ensemble Preregistration

## Status and purpose

This document freezes v41 before any v41 model is fit or any v41 held-out-seed
outcome is computed. The stage is development-only because seed 4047 labels
were touched by earlier v39/v40 development. LA, Nanchang, and fresh simulator
seeds 5057/6067 remain sealed for later confirmation.

v40 passed every integrity check but failed its preregistered efficacy gate
(nested LOCO `-1.1501%`). Its failure localized a protocol mismatch: OOF models
were trained on 48 groups, whereas the evaluated deployment model was refit on
all 60 groups, and adaptation and evaluation mixed the same simulator seeds.
v41 changes this contract rather than relaxing the efficacy threshold.

## Frozen evidence

- 18-network cache SHA-256: `14c6dfad87999e55fa86fa942ac0ca2334129f4d1662b3c31a6e4fc7aa74c242`
- expanded source-rule baseline SHA-256: `f9223a7001078d9a6df0e1a2ad4bfe1d2bc59fc2547621f092e8be702596dbfa`
- v40 audit SHA-256: `760ec21f9a63b6ffaa1bb75b53ea7a00c616bde46118100177363ff6c4488a89`
- executable protocol: `cf_h2o/config/traffic_signal_tsc_v24_seed_heldout_b100_ensemble.json`

## Frozen data partition

Twelve networks have at least 100 safe complete action groups in seeds
2027/3037 and retain all seven development cities. The four low-capacity
Hangzhou networks, Cologne1, and Ingolstadt1 remain source-only.

- adaptation pool: exactly 100 coverage-first groups from seeds 2027/3037;
- OOF construction: five seed-balanced folds of 20 groups;
- each base model: exactly 80 labeled target groups;
- evaluation: every safe complete group from seed 4047 only;
- no seed-4047 label or row enters base-model fitting, OOF selection, target
  context estimation, or candidate choice.

## Frozen model and selection

Each target receives five pairs of rigid-anchor and pairwise-correction models.
Deployment averages each family's five relative action scores. Its uncertainty
is the root mean of within-model uncertainty squared plus between-model score
deviation squared. The anchored candidate grid remains the frozen 31-candidate
v39/v40 grid.

Candidate-selection hyperparameters remain the frozen 128-member v40 grid.
Target decisions use only the 100 OOF groups. A nested leave-one-city-out layer
selects one common target selector using the other six development cities.
There is no seed-4047 target-level tuning.

## Frozen efficacy gate

Both nested-LOCO and all-city development summaries must satisfy all criteria:

1. at least `2%` macro relative regret improvement over the rigid anchor;
2. improvement in at least `5/7` cities;
3. maximum absolute city regression no greater than `0.01`;
4. the all-city selector must be non-anchor.

Failure rejects v41. It does not authorize threshold changes, post-hoc network
exclusions, or access to LA/Nanchang and fresh-seed confirmation outcomes.

## Claim boundary

Passing v41 would establish a development result for simulator-supported target
offline adaptation with independent simulator-seed evaluation. It would not by
itself establish real-world counterfactual validity, zero-shot transfer, or
external-city confirmation.
