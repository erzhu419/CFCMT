# TSC V158 Native-Prefix Ranking Calibration Protocol

**Status: prepared, not run.** V158 is the last frozen attempt to correct the
Jinan native-prefix action-ranking mismatch identified by V157C. It is a new
experiment, not a completion or replacement of the incomplete V157C result.
The configuration protocol is
`tsc-v158-jinan-native-prefix-ranking-calibration-v1`.

## Fixed premise and inputs

V154O--P showed that restored-state labels can reverse native-prefix action
rankings. V155K--N then showed that adding more time points from the same few
seeds does not ensure fresh-seed generalization. V157C found the same practical
failure in Jinan: among the four complete windows where the frozen source and
target arms selected different actions, their predicted ordering agreed with
the native 450-second ordering only once.

V158 therefore keeps the network, objective and frozen V157B base predictions,
but learns a small residual ranking model exclusively from newly collected
native-prefix outcomes distributed over ten new development seeds. The fixed
inputs are:

- parent execution snapshot
  `102b77f2df56f9400f85702f548fe849e0524ca0b7063f93ebc648c1e9f1be4c`;
- Jinan v9 manifest
  `636e09d8e2f47d5fefaa26b33b0552373b01f185e77e18896c97751c267c356c`;
- V157B runtime bundle
  `a6bdea34175f06a399dc00c1c1df06781e578035a2806c434e0f4eb9be61ad93`;
- V157B arm manifest
  `02419e1b9b2370978ed0fe4525c1d4600fd13470c7a76edd625d9c1d126ce52f`;
- SUMO 1.22.0, a 10-second action interval, PhasePressure as the reference
  policy, and the repaired safe-phase executor.

The frozen base arms are `target_only`, `uniform_source`, and
`source_label_placebo`. Their models and predictions are inputs only; V158 does
not refit them or alter their source weights.

## Native development bank

The development seeds are fixed as:

```text
180314, 188625, 127178, 177524, 163775,
150945, 173412, 161481, 150744, 134310
```

They are the first ten pre-existing V157A selector seeds plus 100000. Each seed
contributes ten action groups: four from `jinan_3x4_real` and three each from
`jinan_3x4_real_2000` and `jinan_3x4_real_2500`. The frozen half-open time bins
are `[300,600)`, `[1050,1350)`, and `[1800,2100)` in all three scenarios, plus
`[2550,2850)` in `jinan_3x4_real`.

Within each bin, the uninterrupted PhasePressure probe selects the first
control interval containing at least one action-eligible traffic light with
eight distinct feasible states. All eligible lights are scored before a focal
light is selected. If source and target choose different states anywhere, the
focal light comes from that disagreement set; otherwise all eligible lights
remain candidates. The largest frozen predicted advantage determines the
light, followed by source advantage and traffic-light ID as deterministic tie
breaks. Native outcomes do not enter interval or light selection. A bin that
cannot supply a complete group invalidates the seed; its time, light, or seed is
not replaced.

The probe supplies the PhasePressure action and its 450 samples. Each of the
other seven actions is evaluated by starting SUMO again at t=0, following
PhasePressure to the selected checkpoint, applying the focal action for 10
seconds, and returning to PhasePressure for 440 seconds. SUMO state save/load is
never used. The complete bank therefore contains 100 groups, 800 candidate
rows, 30 uninterrupted PhasePressure trajectories, and 700 fresh non-reference
branches.

The label is candidate cost minus the same-group PhasePressure cost, where cost
is the mean halted-vehicle count per controlled lane over seconds 1 through 450
after the checkpoint. Every branch must reproduce the candidate dataset and
physical prefix exactly, contain one effective intervention at the fixed focal
light, complete all 450 samples, preserve active-population accounting, and
report no teleport. Collisions are retained as diagnostics and do not select
groups or decide the efficiency gate.

## Fixed calibrators and cross-fitting

All three calibrated arms use `CausalReferenceResidualRegressor` with the same
fixed settings: learning rate 0.05, 100 iterations, 15 leaves, minimum leaf size
20, L2 regularization 10, uncertainty quantile 0.9, and random state 20260803.
The causal state/action parent sets are unchanged.

Each arm receives the same native candidate features and the three frozen
target-only prediction fields: score, uncertainty, and context trust. It then
receives one capacity-matched auxiliary triple:

- `target_native`: three zeros;
- `source_native`: frozen uniform-source score, uncertainty, and trust;
- `placebo_native`: frozen source-label-placebo score, uncertainty, and trust.

Thus source versus placebo tests whether the true source prediction contains
useful information after both pass through the same native calibration model;
source versus target tests the value of adding that information to the common
target prediction.

Five whole-seed folds are fixed as:

```text
(180314,188625), (127178,177524), (163775,150945),
(173412,161481), (150744,134310)
```

Each fit uses 80 groups and predicts the 20 groups from its two unseen seeds.
All 100 OOF groups are retained, and the ten equal-weight seed means are the
analysis units. The decision threshold is fixed at zero. A tie or non-negative
predicted improvement executes PhasePressure. A learned stay is also vetoed
when it would cancel a PhasePressure switch.

Reserve execution is authorized only if all three OOF comparisons pass:

| Comparison | Mean requirement | Additional requirements |
|---|---:|---|
| source - target | at most -0.0005 | bootstrap upper 95% bound below 0; at least 7/10 seed wins |
| source - placebo | at most -0.0005 | bootstrap upper 95% bound below 0; at least 7/10 seed wins |
| source - PhasePressure | at most 0 | bootstrap upper 95% bound below 0; at least 7/10 seed wins |

The paired bootstrap uses 10,000 seed-level draws and random seed 20260911.
Full-data calibrator artifacts may be written for audit, but a failed OOF gate
forbids their reserve use and closes this remediation route.

## Fresh-seed closed-loop reserve

The reserve seeds are fixed as:

```text
280314, 288625, 227178, 277524, 263775,
250945, 273412, 261481, 250744, 234310
```

They apply the same pre-existing seed rule with an offset of 200000 and are not
used for collection, fitting, threshold selection, or model choice. Each seed
runs four 3600-second trajectories on `jinan_3x4_real`: `target_native`,
`source_native`, `placebo_native`, and PhasePressure. The learned controllers
use sparse coordination, a 44-interval cooldown, the fixed zero threshold, and
the same no-stay veto. This yields a minimum 450-second separation between
effective learned interventions.

Every reserve cell must complete the evaluator and 3600-second horizon, have a
finite `mean_queue_per_lane`, and report no teleport. Any invalid or missing
cell prevents the ten-seed aggregate; no seed is replaced. Collisions remain a
reported diagnostic. The reserve applies the same three comparisons, bootstrap
limits, and 7/10 seed-win requirements as the OOF gate, using
`mean_queue_per_lane` as the primary metric.

## Decision boundary

An OOF failure ends V158 without reserve simulation. A complete reserve that
fails any one of the target, placebo, or PhasePressure gates is a scientific
failure and closes this remediation route without changing seeds, thresholds,
features, source subsets, cooldown, or action-selection rules.

A reserve pass supports only Jinan closed-loop native-prefix ranking
calibration and a source-prediction contribution under this frozen sparse
controller. It is not unseen-city evidence and does not repair, complete, or
reinterpret V157C.
