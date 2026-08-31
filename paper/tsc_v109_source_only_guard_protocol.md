# TSC v109 source-only deployment-guard protocol

## Objective

Replace the target-closed-loop-developed guard used by the historical v93/v98
pipeline with a guard frozen entirely from source-city counterfactual action
groups. The historical v98 confirmation remains valid evidence for that frozen
historical method, but it is not used to choose the prospective unseen-city
guard.

## Frozen inputs

- immutable source bank: 18 scenarios, 864 cache shards, seeds 2027, 3037,
  and 4047;
- source city groups: Atlanta, Cologne, Hangzhou, Ingolstadt, New York,
  RESCO synthetic, and Salt Lake City;
- pressure prior: `gp_d0_o0p02_s0`;
- online action model: the frozen `constant_alpha_0p9` blend of
  `causal_group_normalized_rigid_advantage` and
  `causal_antisymmetric_pairwise_advantage`;
- target observations, target transition labels, and target closed-loop
  rollouts: none.

## Selection rule

Each source city is held out in turn. Both anchor and correction models are
refit without that city, then combined with the same 0.9 correction weight used
online. These source-LOO action outcomes calibrate both predictive uncertainty
and the deployment guard. The frozen grid contains
eight uncertainty multipliers and five maximum pressure-rule gaps, for 40
candidate guards in total. A one-sided Bonferroni critical value at family-wise
alpha 0.05 is applied across the complete grid. A candidate must also satisfy
the existing mean-UCB, harm-fraction, improving-fraction, and override-count
requirements.

Among admissible candidates, the guard with the lowest mean held-out source
delta is selected. If none is admissible, `enabled=false` means an exact
pressure-prior fallback at deployment; it does not mean unguarded model
execution.

## Runtime contract

The source-LOO uncertainty scale is frozen with the guard and multiplies the
uncertainty emitted by the source/target causal mixture before guard
application. Artifact identity, family, pressure prior, source domains, and
the two `target_*_consumed=false` fields are checked before a rollout can use
the guard.

## Prospective boundary

Snapshot `1f5f4d14debe49f19df9` and family-wise alpha 0.05 were fixed before any
Chicago closed-loop control rollout. Chicago safety admission and subsequent
zero-shot/B100 evaluation may consume this artifact, but they may not modify
its thresholds or uncertainty scale.

The earlier `freeze_v2` anchor-only diagnostic is not deployable: the rigid
anchor agreed with the pressure prior in all 5,828 source-LOO decisions and
therefore did not represent the actual anchored online model. It is retained
only as an implementation-audit record.

## Frozen outcome

The corrected `freeze_v3` composite audit also selected the pressure-prior
fallback. Across 5,828 held-out source action groups, none of the 40 profiles
produced a model override, so there was no source-only effect to certify. The
frozen uncertainty multiplier is 1.9804174692265288 and the result artifact
SHA-256 is
`54b9c6e96cb56d6c4ebd627306a7eae0a9e1203a5f55becc9c1aa5ff50fe5185`.

This is the zero-shot result, not evidence against target-offline adaptation.
The B100 protocol may use target offline action labels in cross-fitted folds to
select a source, source mass, and guard, while target closed-loop outcomes
remain unavailable until final evaluation.
