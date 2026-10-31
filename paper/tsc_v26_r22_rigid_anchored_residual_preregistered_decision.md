# TSC v26/r22 Rigid-Anchored Residual Preregistered Decision

Date frozen: 2026-08-09

## Admissible evidence

This is a sequential method-development screen on the same six opened city
groups used by v25/r21. The complete v25/r21 result is admissible development
evidence. No controller result from either Salt Lake network has been opened.

The v25/r21 temporal repair separated 10 s local mechanism labels from the
60 s rollout value. It reduced the full-family Atlanta regression from about
0.298 to 0.074 normalized action regret, but no one-step family passed the
frozen efficacy gate. Inspection then identified a second estimand mismatch:
the mechanism meta-model predicted the complete normalized action cost and,
at stack weight one, replaced the rigid action model. It did not implement the
claimed rigid-anchored residual world model.

## Frozen method change

The v26/r22 stack keeps the v19 counterfactual cache, action set, source-rule
reference, city-group holdout, and one-step mechanism labels unchanged. Only
the source stack estimand changes:

`predicted action cost = rigid action cost + w * predicted mechanism residual`.

For each outer held-out source city, every inner meta-training city uses both a
mechanism model and a rigid model fitted after excluding the outer and inner
cities. The meta target is observed normalized action cost minus this
pair-excluded rigid prediction. The final residual head is fitted to source
out-of-fold residuals. This prevents the outer city from entering residual
construction through a rigid prediction.

The legacy complete-cost replacement protocol remains in code as an exact
ablation. v26/r22 uses
`rigid_anchored_mechanism_residual_v1` exclusively. Target action labels,
target residual fitting, and target family selection remain forbidden at
budget zero.

## Frozen six-city screen

Targets remain, in fixed order: grid4x4, Cologne1, Ingolstadt1, Atlanta 1x5,
Hangzhou 4x4, and Manhattan 28x7. Each fit excludes the target's complete city
group. The fixed reference is `causal_rigid_advantage`.

Candidate order is frozen in
`cf_h2o.eval.traffic_signal_rigid_residual_selection.CANDIDATES`:

1. rigid plus one-step queue residual;
2. rigid plus one-step mobility residual;
3. rigid plus one-step mobility and queue residuals;
4. rigid plus all five one-step physical residuals.

No red, spillback, or served-only family is reopened because v25/r21 already
showed that each failed as an isolated complete-cost mechanism. Their joint
contribution remains represented in the full five-mechanism candidate.

A candidate passes only if all conditions hold:

- six-city macro normalized action regret improves by at least 10% versus
  rigid;
- at least four of six cities improve;
- maximum city-level absolute regression is at most 0.05.

Passing candidates within 0.005 absolute macro regret of the best passing
candidate form a tie set. Selection then prefers fewer mechanisms and the
fixed order above. A failed screen promotes no family.

## Confirmation boundary

The immutable algorithm snapshot and launch manifests will be generated after
this decision file and selector tests are frozen. Their hashes are experiment
provenance, not tunable inputs. Salt Lake remains unopened unless this screen
passes. If it passes, the selected family and every hyperparameter are frozen
before the two-network Salt Lake confirmation and the subsequent 18-network,
adaptation-budget, and closed-loop matrices.
