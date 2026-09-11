# V154 Rigid Feature Alignment Correction

## Confirmed implementation failure

The original Cologne V150K runtime model is a `FrozenAnchoredBlendModel` with
`constant_alpha_0`: the rigid anchor supplies its score. The anchor is a
`PairwiseActionAdvantageRegressor` trained with 29 named features, but its
prediction method reads their stored training-time column numbers. The training
helper builds the shorter `CONTRAST_FEATURES_V3` schema; V150K evaluation and the
V150L runtime build an expanded contrast schema containing right-of-way fields.
Those extra reference fields shift the start of the delta block.

Nine of the anchor's 29 inputs consequently come from the wrong columns. For
example, `delta_green_q` reads `reference_protected_green_ratio`, and
`delta_green_down_occ` reads `delta_green_q`. This is a schema implementation
error, independent of whether the underlying traffic representation is adequate.

The retained model and read-only replay observations at 25780 and 26700 reproduce
the original eight candidate scores exactly when read using the erroneous
positions: maximum error is zero at both times. Each candidate reaches the same
leaf at both times in all 100 trees (800 action/tree pairs). Reading the same
model by the intended feature names, without refitting, changes both selected
actions from the blocked shared-lane phase (index 3) to the straight-green
pressure reference (index 2). No claim about training minimum/maximum feature
ranges is made; only the model's actual split thresholds were inspected.

Evidence:
`cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/rigid_score_plateau_analysis.json`.

## Frozen correction replay

The correction makes the actual model consumers resolve their existing stored
feature names against the current dataset schema when predicting. It retains
the fitted coefficients, trees and normalization. Regression tests must show
unchanged predictions on the original schema and unchanged predictions when
the supported right-of-way fields are inserted into a contrast dataset.

One corrected closed-loop replay uses the original V150L Cologne1 seed 41242
rigid target-only launch arguments: original model, network, demand, SUMO 1.22.0,
3600-second duration from 25200, 60-second warmup, ten-second control interval,
450-second prediction horizon, direct control and zero cooldown. The sole
controller change is feature-column resolution. Occupancy formulas, vehicle
parameters, signal clearance and collision handling remain as in the original
run. The result receives a separate path and immutable correction source
snapshot; the original V150L result remains retained.

Compare arrival completion against due demand, pending insertion, system load,
waiting, collision/teleport events, and the phase-holding trace against the
retained original rigid result. The retained PhasePressure result provides
service context and includes 20 collision incidents. The new run tests whether
correcting model inputs removes the observed service failure in this fixed
scenario; it does not authorize a full source-aware matrix.

Only result JSON and short logs are retrieved. The runtime model and any SUMO
state remain on the server.

## Corrected Cologne result

Task `t90756` completed the 3600-second replay using snapshot
`589fd20266b7265b6f2f`. The original fitted runtime model and all original rollout
arguments were retained. The corrected result explicitly records
`feature_binding=stored_feature_names` and protocol
`tsc-v154-cologne-rigid-feature-alignment-rollout-v1`.

| Metric | Original rigid | Feature-aligned rigid | Original PhasePressure |
|---|---:|---:|---:|
| Mean tripinfo waiting (s) | 1484.364 | 19.713 | 14.991 |
| Departed / due demand | 409 / 2015 | 2015 / 2015 | 2015 / 2015 |
| Arrived | 210 | 1997 | 1999 |
| Pending insertion at horizon | 1606 | 0 | 0 |
| Collision incidents | 0 | 18 | 20 |
| Starting teleports | 0 | 0 | 0 |

Correcting the column binding removes the severe service collapse in this fixed
scenario. The corrected controller makes 253 switches and 102 same-phase
requests, compared with 22 and 338 originally. It executes 2351 seconds in green
and has 18 active vehicles left at the horizon. Arrival completion against all
due demand increases from 10.42% to 99.11%.

The corrected waiting time is still above PhasePressure, and the zero-incident
diagnostic is **FAIL**: 18 collision incidents (36 recorded event observations)
remain. The original rigid run's zero collisions coexisted with almost complete
loss of service. This result is evidence of a substantial implementation repair,
not a safe-controller admission or superiority to PhasePressure.

Result:
`cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/rigid_feature_alignment_v1/result.json`.
Launch:
`cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/rigid_feature_alignment_launch_v1.json`.
Retrieval was limited to the 378147-byte result JSON and a short log.

## Remaining collision evidence

Both corrected rigid and PhasePressure retain only the first 20 collision
reports, corresponding to ten distinct collision keys in each sample. All
sampled reports are `junction` collisions. Eight corrected-rigid keys and nine
PhasePressure keys involve straight link 2 and opposing left-turn link 13;
one key in each involves straight link 12 and opposing left-turn link 3. In
these sampled cases the left-turn vehicle is nearly stationary around the
internal waiting point, while the through vehicle is moving. The remaining
corrected-rigid key involves straight link 16 and the second internal segment
of left-turn link 8 (`_22_0`), as resolved by the subsequent static extraction.

These samples motivate inspecting the shared internal waiting positions,
vehicle geometry and right-of-way connections. They do not establish the
cause. The sample cap prevents comparison of full-run collision types or time
distributions, and executor state is recorded after simulation and executor
advancement, so a report showing all-red does not establish a collision caused
by an all-red transition. The full-run zero-incident failure remains unchanged.

The subsequent original-start first-collision replay `t90912` reproduced all
23 original action records and both retained collision reports exactly, with
108 observations across three execution stages. At 25657, the stationary
left-turn vehicle's passenger polygon intrudes into the straight vehicle's
polygon by a minimum separating-axis penetration of 0.056509 m. Both vehicles
entered before all-red; the next green starts at 25665. This locates a concrete
internal waiting-position and vehicle-geometry incompatibility, while the
static opposing-movement conflict and yield relations are present. No network
or physics parameters were changed in this diagnostic. See
`paper/tsc_v154_cologne_first_collision_diagnostic_result.md` and the measured
geometry figure under `collision_replay_v1/collision_geometry.png`.

## Source evidence affected

V150K uses the same inconsistent contrast construction during offline scoring,
so this defect affects more than online deployment. Its utility payload remains
unrecomputed and cannot simply be paired with the corrected rigid scorer.
The separate V153 v3 correction has now recomputed all seven targets with the
same target groups, candidate grid, controls, thresholds and true nested folds.
It rejected source admission in all seven cities, with exact corrected-rigid
fallback and zero selected effects. The intermediate v2 Ingolstadt admission is
not retained as source evidence. See
`paper/tsc_v153a_target_calibrated_hierarchical_prior_v3_result.md`.

## Limitations

Shared-lane accessible-service information and occupancy-unit consistency remain
separate issues. The new replay isolates the column error; it does not combine
those changes or infer that correcting one implementation defect establishes
cross-city control quality. At protocol freezing, the corrected replay and the
new source recomputation had not run; their outcomes are reported separately
above and in the V153 v3 result document.
