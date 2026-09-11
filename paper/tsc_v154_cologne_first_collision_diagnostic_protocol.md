# Cologne corrected rigid: first collision diagnostic

Frozen on 2026-09-09 before the diagnostic replay is submitted. Reference:
corrected rigid task `t90756`, snapshot `589fd20266b7265b6f2f`, Cologne1,
seed 41242. This is a read-only mechanism observation of an existing failure.

## Retained observations and static context

The corrected rollout has 18 unique collision keys and 36 raw reports; original
PhasePressure has 20 keys and 40 reports. Both results retain only the first
20 reports, covering ten keys each. All retained reports are junction collisions.
Eight corrected keys and nine pressure keys involve internal lanes `_1_1` and
`_13_0` (straight link 2 and opposing left turn 13). One key in each involves
`_11_1` and `_3_0` (straight 12 and opposing left turn 3). The remaining corrected
sample at 26701 involves straight link 16 and the second internal segment of
left turn 8 (`_22_0`). This first-collision replay does not cover that later event.

A 20,659-byte server stdout extraction retained this junction's internal lanes,
connections, internal junctions and request matrix in
`cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/collision_static_geometry.json`.
No complete network, model or trajectory package was downloaded.

For both main pairs, the `foes` bit is set in both directions. Left turns 13/3
have `response` set for straight 2/12 and `cont=1`; the straight movement's
response bit for the left turn is zero. The internal junction after each left
turn lists the opposing straight lane among its internal prohibiting lanes.
Thus the static file contains the intended conflicting and yielding relations.
SUMO defines these request bits from right to left and uses the internal
junction as a waiting position.
[SUMO network semantics](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html#requests)

Left-turn lane 13 is 8.76 m long and lane 3 is 8.62 m. At the first corresponding
retained events, the left-turn front positions are 8.659 m and 8.518872 m:
approximately 0.101 m before the internal lane endpoint. Their front points are
1.635517 m and 1.639969 m from the opposing straight lane centreline. XML omits
lane width; the replay queries the native width and each actual vehicle's
dimensions. These static distances alone do not establish vehicle overlap.

## Frozen replay

- Start from original SUMO configuration time 25200; seed 41242; stop at 25666
  (466 seconds). No saved-state restart.
- Preserve the original corrected rigid originator, stored model, feature
  binding, 60-second warmup, 10-second control interval, 450-second prediction
  horizon, direct deployment and zero residual cooldown.
- Preserve the original SUMO 1.22.0 runner arguments, vehicle physics, demand,
  lane/network definition, collision `warn` handling and native safe executor.
- Observe times 25630 through 25665 inclusive at three stages: before each
  simulation step, after the step before executor advancement, and after
  executor advancement. Exactly 108 stage samples are required.
- Expected first native report: time 25657, collider `102219_396_0`, victim
  `129962_409_0`, lane `:cluster_357187_359543_1_1`.

The only shared-runner addition is optional `step_observer=None` and three
calls per second around the existing simulation/advance calls. The independent
observer uses getters; it neither requests phases nor changes returned actions.
The before-step observation follows any action request at that time and records
the signal actually supplied to the next simulation step. Collision objects are
queried only after the simulation step; other stages store null, avoiding triple
counting. The original runner's ordinary collision sampling remains untouched.

Each sample records actual TLS and executor state, both focal vehicles when
active, all vehicles on this junction's internal lanes, positions, angles,
speeds, accelerations, dimensions, route/road/lane/link information, and active
and pending counts.
Static output contains original request/foe/continuation data plus native lane
shape, length and width. Tripinfo remains in server `CFCMT_SCRATCH` and is
removed after metric extraction.

## Geometry and decision rules

SUMO 1.22 first checks bounding boxes and then actual bounding polygons before
reporting a junction collision. Its passenger polygon has cut corners; a box
intersection or car-following minGap violation alone is insufficient.
[SUMO 1.22 collision implementation](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSLane.cpp#L1592-L1628)

For a passenger vehicle fully on its current lane with zero lateral offset,
the diagnostic obtains the back point using native `convert2D` at
`lane_position - vehicle_length`; this calls the same lane geometry conversion
as that branch of native `getBackPosition`. It constructs the native zero-margin
eight-vertex passenger shape from the actual front/back points and width, then
records the minimum overlap across separating axes. Native reports remain the
primary collision evidence.
[Vehicle geometry](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSVehicle.cpp#L1454-L1488),
[passenger polygon](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSVehicle.cpp#L6817-L6848),
[native coordinate conversion](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/libsumo/Simulation.cpp#L481-L488).

A PASS requires exact equality of every retained intervention-trace field
before 25666 and all ordinary collision samples through that endpoint against
`t90756`, the expected first collision identity/time, a complete three-stage
window, successful runner completion and zero teleports. There is no floating
tolerance, warmup substitution or retuning after a failure. PASS means the
failing trajectory was reproduced, not that the controller is safe.

Interpretation separates an already-overlapping waiting vehicle, a moving
vehicle entering the conflict, and a change visible only after executor
advancement. It does not infer a missing yielding relation from collisions when
the static relation is present, or infer insufficient clearance merely from a
post-advance all-red sample. A repair needs the resulting geometric and temporal
evidence; no repair or efficacy experiment is part of this replay.

## Implementation and validation

Module: `cf_h2o.eval.traffic_signal_cologne_collision_replay`.

```sh
python -m cf_h2o.eval.traffic_signal_cologne_collision_replay \
  --launch-record /remote/frozen_inputs/original_v150l_launch.json \
  --reference-result /remote/rigid_feature_alignment_v1/result.json \
  --tripinfo /remote/CFCMT_SCRATCH/cologne_first_collision_v1/tripinfo.xml \
  --out /remote/cologne_first_collision_v1/result.json
```

Root stages and submits one CPU/8 GB task using the original SUMO 1.22 wrapper;
this module neither stages nor submits. Three local tests passed: the actual
runner preserves action writes and metrics with/without the observer while
exposing the three stages in order; changed/missing retained trace fields fail
exact comparison; passenger cut-corner geometry and native lane-following back
position differ from a naive rectangular approximation.

## Limitations

Derived passenger polygons are null when the rear spans lanes, lateral position
is nonzero, or the vehicle uses another shape. Actual poses and native collisions
are still retained. One-second observations cannot distinguish substep ordering
inside SUMO's collision stages. The retained original samples are capped, and
this bounded replay concerns only the first corrected rigid incident, not all
18 incidents or the full safety performance of either policy.
