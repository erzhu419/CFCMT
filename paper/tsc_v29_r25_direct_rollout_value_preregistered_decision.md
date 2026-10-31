# TSC v29/r25 Direct Invariant Rollout Value: Preregistered Decision

Frozen before any v29 target result. Salt Lake remains sealed.

## v28 Diagnosis

The v28 terminal-value estimand was informative in observed development labels,
but its source mechanism selector chose `prior_only` for every held-out target.
Four targets therefore fell back exactly to rigid; Grid4x4 and Manhattan used
small weights on the inaccurate analytic long-horizon prior and regressed. This
rejects the v28 prior-based estimator, not yet the terminal-value estimand.

## Frozen Candidate

Evaluate exactly one family:

`cfcmt_direct_rollout_value_rigid_residual`

It retains only the `terminal_clearance -> terminal_system_load` causal
mechanism and the same fixed 10-second focal action plus phase-pressure
continuation estimand. Unlike v28, the terminal mechanism is fitted to a
domain-normalized source outcome with a zero simulator prior. Parent variant
and mechanism inclusion are selected by source-city OOF action ranking. The
final action score remains a rigid-anchored residual:

`rigid score + source-selected direct terminal-value residual`.

The model may use only the registered terminal-clearance parents and invariant
mechanism context factorization. It may not use target labels, target residuals,
an expert gate, `interval_cost` as a mechanism target, city/network ID, or a
dense all-feature head. Five source domains and ten pair-excluded rigid and
mechanism fits are mandatory.

## Frozen Gate

On the same six development targets and immutable v26 rigid reference, promote
only with at least 10% macro regret improvement, at least four cities improved,
maximum city regression at most 0.05, and complete fail-closed provenance and
zero-target-label audit. If it fails, no direct/physical terminal blend is
introduced in this round; the next candidate must address model capacity under
a separately frozen protocol.

## Claim Boundary

Passing would show that a source-trained invariant policy-value mechanism is
useful; it would not support a claim that an uncalibrated long-horizon simulator
prior is useful. Failing would motivate a nonlinear but still parent-restricted
terminal model, not post-hoc target calibration.
