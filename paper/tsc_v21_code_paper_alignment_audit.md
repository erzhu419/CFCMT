# TSC v21 Code--Paper Alignment Audit

Recorded while the frozen v21/r17 exact-ablation matrix was still running on
2026-08-08 (Asia/Shanghai). No v21 result had been inspected.

## Scope

This note distinguishes the estimator implemented by the frozen v21 snapshot
from the residual-world-model language in the current TSC manuscript. It does
not alter the v21 snapshot, cache, policies, seeds, or decision rules.

## Verified implementation path

1. `collect_counterfactual_transitions_v3` computes analytic, uncalibrated
   mechanism priors with `uncalibrated_mechanism_priors_v3` and stores them in
   each absolute `MechanismDataset`.
2. `build_action_contrast_dataset` correctly converts both observed outcomes
   and analytic priors into candidate-minus-reference action contrasts.
3. The primary `cfcmt_fused` family is implemented by
   `FusedCausalMechanismAdvantageModel`.
4. Before its auxiliary mechanism model is fitted,
   `_normalized_mechanism_dataset` replaces every selected mechanism prior by
   zero. At prediction, `_zero_prior_dataset` again supplies zero priors to the
   auxiliary mechanism model.

Therefore, the frozen v21 primary is not an analytic-simulator-prior residual
model along its auxiliary mechanism path. It is a source-selected direct
predictor of normalized mechanism action contrasts, fused with a rigid causal
action core and, when target groups are available, a target specialist. The
analytic prior remains present in other benchmark families, but it is not the
baseline corrected by the primary fused mechanism stack.

## Claim consequence

The current code can support claims about action-contrast cancellation,
declared mechanism parent sets, nested source-city selection, exact rigid
ablation, target out-of-fold fusion, and guarded deployment. It cannot support
the stronger statement that the primary fused path learns a residual on top of
an uncalibrated mechanism simulator.

The v21 experiment remains necessary and interpretable: fused versus
fused-rigid still isolates the selected auxiliary mechanism path under an
otherwise exact controller match. Its outcome must not be relabelled as a
simulator-residual contribution.

## Required remediation before confirmation

The confirmatory method must take one explicit, tested route:

- **Residual route:** normalize the analytic prior with a quantity available
  without target outcome labels, preserve that normalized prior in fit and
  prediction, and learn only the residual. Add tests proving that changing the
  prior changes the residual prediction, that reference actions remain exactly
  zero, and that zero-shot target inference does not use target outcomes.
- **Direct-contrast route:** retain the current estimator and remove
  simulator-residual language from the title, abstract, theory, diagrams, and
  contribution claims.

The residual route is the preferred development hypothesis because it restores
the intended causal-plus-residual world-model design. It must first pass a new
development matrix and an exact prior-removal ablation; it cannot be inserted
directly into a confirmatory run.

## Additional information-accounting clarification

When target groups are available, source-LOO guard fitting currently operates
on the merged source-plus-target-adaptation dataset, while disjoint target
calibration groups perform the subsequent policy calibration. This is legal
few-shot information, not evaluation leakage, but it must be reported as
target-label-informed guard fitting rather than source-only guard fitting. A
future refactor should expose this accounting directly in diagnostics and the
budget auditor.

## Analytic-prior horizon mismatch

The counterfactual label has the explicit estimand
`first_action_then_phase_pressure_rollout_value`: the candidate action is
executed for one control interval and phase pressure controls the remaining
intervals. In contrast, `collect_counterfactual_transitions_v3` currently calls
`uncalibrated_mechanism_priors_v3` with
`control_interval_sec * counterfactual_horizon_intervals`. The analytic prior
therefore treats the full 60-second horizon as one candidate-action interval,
including its service-capacity and clearance terms. It does not implement the
label's first-action-then-prior rollout semantics.

This mismatch does not change the frozen fused versus fused-rigid comparison,
because both primary paths zero these priors. It does make simulator-only and
prior-residual comparisons unsuitable as submission evidence in their current
form. Before confirmation, the prior protocol must either implement the same
rollout estimand or be renamed as a one-step physical proxy, recomputed from
cached state features without rerunning SUMO, and evaluated with an exact
prior-removal ablation.

## Manuscript evidence ledger failures

The current manuscript is a development draft rather than a result-locked
submission. The following statements and artifacts must be replaced from the
final audited matrix rather than edited numerically in place:

1. The abstract and Results report an eight-scenario experiment. The frozen
   development protocol contains 16 target networks grouped into six city or
   benchmark domains. Network-weighted and city-first estimands must both be
   reported so that the six Hangzhou networks do not dominate the conclusion.
2. The abstract, Results, Discussion, and Figure 1 describe the primary as a
   mechanism residual on an uncalibrated simulator. That wording conflicts
   with the verified direct normalized-contrast implementation above.
3. The manuscript's 600-second and 3,600-second numbers come from older
   protocol versions. They cannot be mixed with the r21/r22 benchmark-aware
   source selector, 16-network scope, exact rigid ablation, or the new target
   information budgets.
4. The Results currently make safety-region claims from aggregate queue
   values. The final claim must use the preregistered collision-incident gate,
   teleport audit, worst-city effect, and paired city/network estimates in
   addition to average queue.
5. The passive transit dataset section is disconnected from the TSC estimator
   and action protocol. It should move to a separate bus paper or a clearly
   labelled motivation appendix; it cannot serve as TSC counterfactual
   evidence.
6. RESCO and LibSignal native RL rows remain external-protocol cross-checks.
   They must not appear in a common ranking table with same-protocol CFCMT
   phase control unless state, action timing, reward, demand, seeds, horizon,
   and metric aggregation are made identical.

The replacement order is: freeze the estimator, pass the development gates,
run an untouched 3,600-second confirmatory matrix, generate tables and figures
from the audited JSON only, then rewrite Abstract, Results, Discussion, and
Methods against a single evidence ledger.
