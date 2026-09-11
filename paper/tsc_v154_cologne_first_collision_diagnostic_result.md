# Cologne first collision: internal waiting-point clearance failure

The first collision after correcting rigid feature binding is a physical
overlap between a stopped left-turning vehicle and an opposing straight vehicle
already inside the junction. The observed left-turn waiting position leaves
approximately **5.65 cm of vehicle-body intrusion** into the straight vehicle's
path. The result identifies a specific incompatibility between this network's
internal waiting geometry and the actual vehicle bodies. It does not establish
a general SUMO bug or a validated repair.

![Actual passenger polygons at the first collision](../cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/collision_replay_v1/collision_geometry.png)

## Exact reproduction

Task `t90912`, source snapshot `85b2b43cd12832c5a1dd`, replayed corrected rigid
reference `t90756` from the original start time 25200 to 25666 with SUMO 1.22.0,
Cologne1 and seed 41242. All six frozen checks passed:

- All 23 retained action-trace rows matched every original field exactly.
- Both retained native collision reports matched every original field exactly.
- The expected first collision identity and time were reproduced.
- All 108 stage observations were present; the runner completed without teleport.

The diagnostic ran in 1.427 seconds. Its PASS certifies faithful observation of
the failing trajectory; the collision safety requirement remains failed. No
network, controller, fitted model, vehicle physics or collision threshold was
changed by this replay.

## Observed geometry

At time 25657 immediately after `simulationStep`, before executor advancement:

| Quantity | Straight vehicle | Left-turn vehicle |
|---|---|---|
| ID | `102219_396_0` | `129962_409_0` |
| Internal lane | `:cluster_357187_359543_1_1` | `:cluster_357187_359543_13_0` |
| Movement | link 2, straight | link 13, opposing left turn |
| Lane position | 28.460661 m | 8.659000 m |
| Speed | 9.248266 m/s | 0 m/s |
| Length × width | 4.3 × 1.8 m | 4.3 × 1.8 m |
| Lateral offset | 0 m | 0 m |

Both vehicles have the native passenger shape and lie fully on their respective
current lanes. Their actual front positions, native lane-converted back
positions and widths therefore support the frozen eight-vertex passenger
polygon reconstruction. The minimum separating-axis overlap is
**+0.056508558 m**, coinciding with the native `junction` collision report.

The left-turn lane is 8.76 m long. Its vehicle front remains 0.101 m before the
lane endpoint and 1.635517 m from the opposing straight lane centreline. The
closest vertex of its body, however, is only **0.843491442 m** from that
centreline. The straight vehicle extends 0.9 m from the centreline, producing
the observed 0.056508558 m intrusion. The vertex projects to 25.604824 m along
the straight lane's shape, within the straight vehicle's full-width side segment
at 24.579406–28.017832 m. This is an overlap of the finite vehicle polygons,
not merely their extended centre lines or rectangular bounding boxes.

The network's `foes` and `response` records contain the intended conflict and
left-turn yielding relation; `cont=1` permits the internal waiting location.
Its native lane width is 3.2 m. The local failure is therefore compatible with
an internal waiting point whose vehicle-body clearance is insufficient even
though its static yielding connection exists.

## Observed sequence

| Time | Observation |
|---:|---|
| 25644, after executor | `GGGggrrrrrGGGggrrrrr` opens: straight link 2 is `G`, left-turn link 13 is `g`. |
| 25645, after simulation step | Left-turn vehicle enters internal lane 13. |
| 25649, after simulation step | Left-turn body already intrudes 0.040412 m into the straight body's swept strip; both movement signals are still green. The straight vehicle is approaching link 2 with `G`. |
| 25650, after simulation step | Straight vehicle has entered internal lane 1_1. Left-turn vehicle has accrued one second of waiting; intrusion is 0.053841 m. |
| 25650, before next simulation step | A control request starts the yellow transition. Both focal vehicles are already internal. |
| 25653, after executor | All-red starts. |
| 25655–25658 | Left-turn lane position is exactly 8.659 m at every sampled time. |
| 25657, after simulation step | Straight vehicle passes the stationary left-turn body; native collision and +0.056509 m polygon overlap occur before executor advancement. |
| 25658, after simulation step | A second raw report retains the same collision fields. Current vehicle polygons are separated; this is the same collision key, not evidence of another independent impact. |
| 25659, after simulation step | Left-turn vehicle begins crossing its internal continuation after the straight vehicle has passed. |
| 25665, after executor | The next green phase finally opens. |

All-red is present in all three observations at the first collision time. It
has already been active for four seconds, and the next phase remains closed
for another eight seconds. The executor continues extending all-red while
internal occupancy remains. Across the 36 observed timestamps, executor
advancement changes none of the recorded physical vehicle fields; `nextTLS`
signal characters change when the signal itself changes.

This sequence places the body intrusion before the yellow transition and the
collision before any new green release. The relevant vehicles entered under
the previous green phase and were already inside the junction. Extending the
existing all-red interval alone does not address the observed waiting body's
intrusion into the already-entered straight vehicle's path.

## Implication and next diagnostic

The immediate defect is concrete: the realized internal waiting position and
passenger body are incompatible with the opposing straight vehicle's passage.
This is more specific than an unexplained safety penalty or a learning-score
calibration problem. The available evidence places the next repair target in
the network's internal waiting-point clearance. It does not identify whether
the original position arose from a network-generation default, a particular
conversion choice, or a wider SUMO modeling limitation.

The next bounded step is a **separate geometry package** correcting the
internal waiting-point clearance against actual vehicle bodies and foe paths,
followed by the same frozen collision-window check. The original network and
this failure evidence remain intact. The 5.65 cm overlap is not itself a
validated distance by which to move the stop position: moving along a curved
lane also changes the vehicle orientation and polygon. Any wider safety
admission must follow an independently specified repair and validation.

## Limitations and evidence

This replay identifies one collision key from a rollout containing 18 keys.
The original PhasePressure samples show a similar dominant lane pair, but a
successful fix for this first corrected-rigid incident would not by itself
prove that all pressure or rigid collisions are removed. One-second samples
do not resolve SUMO's internal substep detection order. Derived polygons are
unavailable when a vehicle rear spans lanes or lateral offset is nonzero;
neither restriction applies to the two vehicles at the first collision.

- [Frozen protocol](tsc_v154_cologne_first_collision_diagnostic_protocol.md)
- [Raw replay result](../cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/collision_replay_v1/result.json)
- [Compact derived analysis](../cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/collision_replay_v1/analysis.json)
- [Editable diagnostic figure](../cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/collision_replay_v1/collision_geometry.svg)
- [SUMO 1.22 passenger polygon implementation](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSVehicle.cpp#L6817-L6848)
- [SUMO 1.22 junction collision implementation](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSLane.cpp#L1592-L1628)
