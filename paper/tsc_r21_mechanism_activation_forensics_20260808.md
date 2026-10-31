# TSC r21 Mechanism-Activation Forensics

Recorded on 2026-08-08 from the physically validated r21/r17 development
matrix. This note is an implementation audit, not a new performance result.

## Audited artifacts

- Run: `tsc_v21r17_exact_ablation_dev600_20260808b`
- Budgets: 0, 8, 16, 32, 60, and 120 target counterfactual groups
- Scope per budget: 16 target networks, six held-out city/benchmark groups,
  three evaluation seeds
- Source implementation SHA-256:
  `e53c8ec17cee2e687025c06146478ca81e2677bb6c27bf39fa38e976e26d2793`

The frozen source model is identical across target-label budgets by design, so
the following activation counts are identical at all six budgets.

## Observed activation

Across the 16 target fits:

- the source mechanism stack was enabled for 6 targets and rejected for 10;
- selected stack weights were `1.0` for 3 targets, `0.75` for 3 targets, and
  `0.0` for 10 targets;
- 96 mechanism components were evaluated (16 targets times 6 auxiliary
  mechanisms), but only 11 components passed source-city validation;
- `red_accumulation` accounted for 9 of the 11 enabled components;
- `mobility` and `served_movement` were enabled once each;
- `queue_propagation`, `spillback`, and `terminal_clearance` were never
  enabled;
- every enabled component used latent rank 0.

Consequently, the r21 full path usually reduces to the rigid action-contrast
core. Where a source mechanism stack is active, it is normally a one-mechanism
red-queue refinement rather than a functioning multi-mechanism latent world
model.

## Estimand audit

The current mechanism stack is not an analytic-simulator residual estimator.
`_normalized_mechanism_dataset` replaces every mechanism prior with zero after
domain-label normalization, and `_mechanism_features` again predicts from a
zero-prior dataset. Therefore `prior_only` means a zero action-effect feature;
it does not mean that the uncalibrated simulator prior was retained. The stack
learns direct normalized mechanism action effects and then maps them to the
normalized control-cost contrast.

This distinction is material. A future residual-world-model claim requires a
target-available physical normalization applied consistently to both the SUMO
prior and the observed mechanism outcome, followed by explicit fitting of
`normalized outcome - normalized simulator prior`. Normalization by a target
label-derived city scale is not available at zero-shot deployment and cannot
be hidden inside that residual definition.

## Claim boundary

The r21 evidence can support a sparse, source-validated action-contrast model
with a pressure-policy fallback. It cannot by itself support claims that
multi-mechanism MC-WM factors or a target-instantiated latent mechanism state
caused the closed-loop gains. The manuscript must distinguish module presence
from selected module activity.

The r22/r18 joint target gate can reject harmful active source mechanisms, but
it cannot make source mechanisms that collapsed to prior-only become useful.
Therefore a successful r22 safety result would validate the gate, not erase
this activation audit.

## Precommitted redesign boundary

If r22 does not establish a non-degenerate mechanism contribution, the next
development module is a physically normalized simulator-residual estimand plus
a deterministic lane- and movement-normalized local coordinate representation.
It must be evaluated in stages:

1. rigid action-contrast core;
2. physically scaled simulator prior without a learned residual;
3. deterministic local-coordinate mechanism residual with latent rank 0;
4. source-validated static-context latent factor, enabled only when held-out
   source-city action regret improves;
5. the existing target-group OOF mechanism gate and target specialist.

No r21/r22 acceptance threshold will be weakened post hoc. Each stage must
report mechanism activation, city breadth, worst-city harm, collision
incidents, and exact-rigid model-level ablations.

## Runtime evidence

The r22 execution log independently exposes a computational duplication:
`cfcmt_mechanism` and `cfcmt_fused` refit the same frozen source stack and the
same source-LOO predictors. The confirmation implementation may reuse those
artifacts only under exact dataset, target-exclusion, label, group, and
reference-row equality checks. Prediction-equivalence tests are required
before the optimization is admitted.
