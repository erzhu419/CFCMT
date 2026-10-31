# TSC v28/r24 Policy-Conditioned Rollout-Value Residual: Preregistered Decision

Frozen before any v28 leave-one-city-out target result was produced. Salt Lake
remains sealed.

## Corrected Omission

The v26 "full" physical residual was not a full policy-value model. By class
construction it excluded both `interval_cost` and `terminal_system_load`, and
therefore contained only the five local physical mechanisms. This was a valid
local-mechanism ablation but not the intended causal residual world model.

An estimand information audit over the six-city development cache found that
the observed 60-second terminal system load is the strongest stored mechanism
proxy for the 60-second mean control cost: pooled centered action correlation
0.488, best-action agreement 0.683, and normalized action regret 0.2167. The
best one-step local proxy, total queue, reached correlation 0.297 and regret
0.2403. These values are development diagnostics, not held-out evidence.

## Frozen Candidate

Evaluate exactly one new family:

`cfcmt_rollout_value_rigid_residual`

The mechanism head predicts only `terminal_system_load`, defined as global
active-plus-pending vehicles per controlled network lane after the focal action
for one 10-second interval followed by the fixed phase-pressure continuation
policy to 60 seconds. It uses the existing causal parent set for terminal
clearance, portable physical normalization, analytic uncalibrated prior, and
whole-source-city nested cross-fitting.

The stack target is the normalized 60-second interval cost minus the rigid
action score. Deployment is therefore

`rigid score + source-selected rollout-value residual`.

No target transition/action label, target residual, target gate, route ID,
network ID, or target-city hyperparameter is permitted. There is no selective
expert gate in v28. The source stack protocol must be
`rigid_anchored_mechanism_residual_v1` and all five source cities plus ten
pair-excluded rigid/mechanism fits are required.

## Frozen Development Gate

Use the immutable v26 rigid result as reference on Grid4x4, Cologne1,
Ingolstadt1, Atlanta1x5, Hangzhou4x4, and Manhattan28x7. The candidate is
promoted only if all conditions hold:

- macro mean normalized action-regret improvement is at least 10%;
- at least four of six cities improve;
- maximum city regression is at most 0.05;
- all six complete-city zero-target-label results pass snapshot, cache,
  baseline, holdout, source-domain, pair-excluded-fit, and result-hash audit.

If the family fails, it is not combined with local mechanisms in the same
round. A later multiscale combination requires a new frozen protocol. If it
passes, Salt Lake may be opened once for confirmation before closed-loop tests.

## Claim Boundary

`terminal_system_load` is a policy-conditioned source counterfactual outcome,
not a real-world target label and not an action-independent causal mechanism.
Any manuscript claim must describe it as a rollout-value mechanism under the
fixed continuation policy. Its usefulness does not by itself validate a
general long-horizon world model.
