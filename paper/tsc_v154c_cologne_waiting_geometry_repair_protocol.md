# V154C Cologne 13→24 waiting-point geometry repair

Frozen before the repaired-network rollout. The repair follows the exact first
collision in `t90912` and applies only to opposing left-turn movement 13 versus
straight movement 2 at `cluster_357187_359543`. The original network remains
unchanged. A separate network file carries the repair.

## Physical rule

Use the recorded passenger dimensions: length 4.3 m and width 1.8 m for both
vehicles. Place the left-turn front **at the candidate internal split**, with
zero lateral offset. Compute its back point 4.3 logical metres earlier along
the original lane-13 shape and construct the original SUMO passenger octagon.
The polygon must remain outside the opposing straight vehicle's swept strip,
whose half-width is 0.9 m about the actual lane-1_1 centreline. All waiting-body
vertices at the selected split project within the finite straight shape.

The criterion does not subtract the observed 0.101 m stopping offset and does
not use collision counts, waiting time, or a subsequent simulation to choose
the position. The observed 5.65 cm overlap is evidence of the defect, not a
prescribed distance to move the split.

Find the most downstream geometric boundary satisfying nonnegative clearance.
Then round its logical lane position toward the upstream side on the original
network's 0.01 m length grid. No positive clearance target is tuned after a
rollout. New split coordinates retain twelve decimal places to preserve the
computed margin.

| Quantity | Frozen value |
|---|---:|
| Unrounded admissible boundary along original lane 13 | 8.487193932616 m |
| Lane 13 logical length | 8.76 → **8.48 m** |
| Lane 24 logical length | 19.77 → **20.05 m** |
| New split x | 11788.046881453862 m |
| New split y | 13325.914587635907 m |
| Retreat along original geometric shape | 0.280173623788 m |
| Body clearance when front reaches new split | **0.002367516 m** |
| Original combined logical length | 28.53 m, preserved |
| Original combined geometric length | 28.531026892008 m, preserved |

If the original 0.101 m stopping offset repeats, the geometric prediction is
approximately 0.035618127 m clearance. This prediction is separate from the
repair criterion and requires observation in the paired validation.

## Network semantics and exact scope

SUMO models internal waiting with a lane split: vehicles can approach the end
of the first internal lane and then wait before entering its continuation.
`contPos` supports customizing this position; SUMO 1.22's builder implements it
by splitting the connection shape into first and continuation shapes.
[SUMO waiting semantics](https://sumo.dlr.de/docs/Simulation/Intersections.html#waiting_within_the_intersection),
[SUMO 1.22 split implementation](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/netbuild/NBEdge.cpp#L1897-L1914).

The repair directly changes the independently copied compiled network, with
exactly six semantic attribute edits:

- Lane `:cluster_357187_359543_13_0`: `shape` and `length`.
- Lane `:cluster_357187_359543_24_0`: `shape` and `length`.
- Internal junction `:cluster_357187_359543_24_0`: `x` and `y`.

Lane 13 loses its final 0.28 logical metres; lane 24 gains that part. The former
split point `(11788.28, 13326.07)` is retained as an interior vertex of lane 24.
Every original vertex remains in the same order along the combined path, with
one new collinear split point. All connections, `via` values, identifiers,
`foes`, `response`, `cont`, signal plans, speed limits and ordered incoming or
internal lane lists retain their original values. The mirrored movement 3→20
is outside this repair.

Changing only the internal junction coordinates would not move the actual
waiting point: the lane-13 endpoint and its continuation must move together.
The synchronized edit targets internal junction 24, not the first lane's ID.

## Existing distance representation

The original geometric/logical ratios differ slightly: lane 13 has
`8.765431944233 / 8.76 = 1.000620084958`, while lane 24 has
`19.765594947775 / 19.77 = 0.999777185016`. The new split uses lane 13's original
ratio. Lane 13 retains that ratio, preserving its earlier logical-position to
geometry mapping. Lane 24 receives the remaining total logical length.

Its local distance mapping necessarily changes slightly when two portions
with different original ratios are combined into one lane. The repair preserves
the complete physical path and total logical length; it does not claim an
identical logical-position mapping at every downstream point.

## Artifact and validation contract

```sh
python scripts/data/repair_cologne_internal_waiting_geometry.py \
  --net /shared/original/cologne1.net.xml \
  --out /shared/new_geometry_package/cologne1.net.xml \
  --report /shared/new_geometry_package/geometry_report.json
```

The script records source/output paths, frozen dimensions, geometric values,
the six actual semantic changes and invariant results. It verifies the
serialized output against the original parsed network.

Three tests passed using the retained actual junction geometry: the six-attribute
scope and complete path are preserved; 8.48 m is admissible while the next
centimetre at 8.49 m fails the same body-clearance condition; writing an
independent network leaves the source bytes unchanged and preserves the
validated semantic delta.

## Frozen paired simulation

Write a separate network file and SUMO configuration on the server. The new
configuration references the original absolute route file and changes only
its network path. Configuration begin/end remain 25200/28800; the validation
runner stops at 25666.

Use SUMO 1.22.0 and controller source snapshot **85b2b43cd12832c5a1dd**,
including its original occupancy equations. Run the new standalone validation
tool by file path while importing `cf_h2o` from that frozen source. The new
tooling receives its own immutable source snapshot. This keeps the subsequent
occupancy correction out of the geometry comparison.

Both simulations start at 25200 and stop at 25666: 466 seconds, seed 41242,
60-second warmup, 10-second control interval, 450-second prediction horizon,
original rigid model, stored-feature-name binding, direct deployment and zero
cooldown. Collision warnings, junction checks, no-teleport execution, vehicle
parameters and native phase clearance retain their original settings.

One CPU/8 GB task performs two simulations sequentially:

1. Run the new driver with the original network. Require exact equality of
   the 23 original action records, two collision reports and 108 stage
   observations against `t90912` and its retained `t90756` reference. Require
   complete duration and zero teleports. This prerequisite detects mistakes
   in the new driver or frozen imports; failure stops before repaired simulation.
2. Run the repaired network. Require completion of all 466 seconds and all
   108 stage observations from 25630 through 25665; zero native collision
   events/incidents and zero teleports over the full bounded run; and actual
   passage of both focal vehicles along their original incoming → internal
   → outgoing route sequence. The focal IDs are `102219_396_0` (straight) and
   `129962_409_0` (left turn). Non-insertion or non-passage cannot pass.

Record waiting-body clearance, passage times, endpoint traffic metrics, longest
matching action prefix, first action difference and first observed physical
difference. Geometry may affect preceding vehicles and subsequent feedback
decisions; these differences are measured without an arbitrary unchanged-prefix
time cutoff. Native collision records determine the collision outcome;
passenger polygons provide mechanism evidence. If the new trajectory contains
no stationary focal left-turn sample, record that absence rather than inventing
a clearance observation.

The baseline output retains a compact comparison record; its existing detailed
trace is reused. New network, demand and model files remain on the server.
Retrieve the small repair report and validation JSON for adjudication.

Protocol: `tsc-v154c-cologne-waiting-geometry-validation-v1`.
Artifact directory: `tsc_v154c_cologne_waiting_geometry_20260909`.
Freeze the package before observing repaired dynamics. Retain a failed
validation; do not choose a different shift or extend this window after
observing its result. A pass establishes collision removal with actual focal
passage in this bounded paired test.

## Limitations

The rule covers the observed 4.3 × 1.8 m passenger class at this movement pair.
It does not claim to protect different vehicle dimensions, mirrored movement
3 or all 18 collisions of the corrected rollout. Any repaired-network trajectory
can change because the waiting point changes; the pair must preserve control
code and runtime parameters rather than require identical repaired-arm actions
after the first geometry-induced state difference.
