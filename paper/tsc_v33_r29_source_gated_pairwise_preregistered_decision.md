# TSC v33/r29 Preregistered Decision: Source-Gated Pairwise Transfer

Date frozen: 2026-08-09

## Motivation

v32 demonstrated useful all-action pairwise signal in four development cities
but unsafe negative transfer in two. v33 adds one module only: a source-city
leave-one-domain-out lower-confidence-bound gate between the original rigid
expert and the frozen v32 pairwise expert. The target city does not participate
in expert fitting, gate fitting, gate selection, or error-margin calibration.

## Frozen candidate

- family: `cfcmt_source_gated_pairwise_advantage`
- experts: unchanged `causal_rigid_advantage` and frozen v32
  `causal_antisymmetric_pairwise_advantage`
- source expert predictions: one complete leave-one-source-city-out fold per
  source domain
- gate training rows: source action groups where the two experts choose
  different actions
- gate target: source-only matched-group normalized cost of the pairwise choice
  minus the rigid choice
- gate features: the eight frozen score-geometry features in
  `SOURCE_EXPERT_GATE_FEATURE_NAMES`
- gate estimator: Ridge with alpha 10
- conservative admission: predicted cost delta plus the 0.75 quantile of
  one-sided source-city OOF error must be below zero
- minimum support: 24 disagreement groups from at least three source domains
- source promotion: robust regret gain at least 0.002, at least one third of
  source domains improve, and worst source regret increases by at most 0.01
- target data budget: zero groups and zero labels

If the source gate fails its own checks for a target fold, the deployed family
is exactly the original rigid expert for that fold.

## Data and integrity

- six development targets: grid4x4, Cologne1, Ingolstadt1, Atlanta 1x5,
  Hangzhou 4x4, and Manhattan 28x7
- immutable cache aggregate SHA-256:
  `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`
- immutable source-rule baseline SHA-256:
  `25e6291ba9ff7348642baf09498c9c7bea36ef3fd77a7ee6680b289a03201acb`
- Salt Lake remains sealed and may be opened only after this stage passes

## Frozen promotion gate

Promote the candidate only if all conditions hold:

1. city-macro normalized action regret improves by at least 10% versus rigid;
2. at least four of six target cities improve;
3. maximum absolute city regression is at most 0.05;
4. all snapshot, cache, complete-city holdout, zero-target-budget, source-OOF,
   causal-parent, and gate-support checks pass.

No result-dependent threshold, quantile, parent, or expert change is permitted.
