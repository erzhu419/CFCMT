# V154D Cologne bilateral internal waiting geometry

Frozen before the bilateral repaired-network rollout. This is a separate
package following V154C, which retained a collision at the unmodified mirrored
waiting point. V154C's network, result, fixed window and FAIL remain intact.

## Repair rule and scope

Retain V154C's 13→24 waiting split exactly (8.48 m / 20.05 m). Apply the same
physical rule to waiting lane `:cluster_357187_359543_3_0`, continuation
`:cluster_357187_359543_20_0`, and opposing straight lane
`:cluster_357187_359543_11_1`. The V154C stationary vehicle at this mirrored
point intruded into the straight swept strip by 0.049633247 m; the same movement
pair also collided later in the original full rollout.

Use the recorded 4.3 m length and 1.8 m width passenger octagon. Put its front
at the candidate split, with its back 4.3 logical metres earlier along the
original waiting-lane geometry and zero lateral offset. Require nonnegative
body clearance from the actual straight centreline's 1.8 m swept strip. Solve
for the most downstream admissible position, then round upstream to the
original 0.01 m logical-length grid. Do not subtract a stopping offset or use
rollout outcomes to choose the position. Record that the body projects within
the finite straight segment.

The static calculation, completed before running the bilateral simulation, is:

| Mirrored split quantity | Frozen value |
|---|---:|
| Unrounded admissible lane-3 position | 8.374454261820352 m |
| Waiting lane 3 logical length | 8.62 → 8.37 m |
| Continuation lane 20 logical length | 19.58 → 19.83 m |
| New split x / y | 11804.545222726876 / 13329.842575789198 m |
| Geometric retreat | 0.249888421684 m |
| Body clearance with front exactly at split | +0.001530323108 m |

The original 13→24 split stays at 8.48 / 20.05 m, with its front-at-split
clearance of +0.002367516 m. Both shifts are determined by the same geometry
criterion; neither uses observed repaired-arm traffic performance.

The original waiting geometry/logical ratio determines the new split. Retain
all original vertices along the combined path, including the former split as
a continuation vertex. Preserve each pair's complete geometry and combined
logical length. Each waiting prefix retains its original position mapping;
the continuation absorbs the shifted nominal length and its local mapping
changes slightly, as in V154C.

Build the independent bilateral package from the original network. The only
semantic edits versus that original are the shape/length of lanes 13, 24, 3
and 20, and x/y of internal waiting junctions 24 and 20: exactly 12 attributes.
The six attributes on 13→24 must equal the retained V154C report exactly;
the additional six belong only to 3→20. Connections, IDs, via links, ordered
incoming/internal lane lists, conflict and yielding responses, signal plans,
speeds, demand and vehicle parameters are preserved.

The geometry report records the calculated position and clearance before any
bilateral dynamics are observed. The serialized network must retain the same
semantic delta. These checks detect a misplaced split or unrelated network
change and stop preparation if either occurs.

## Paired runtime and fixed observations

Use SUMO 1.22.0, original controller source snapshot
`85b2b43cd12832c5a1dd`, the original rigid model and stored-feature-name binding.
The original occupancy equations remain in use. Execute the new standalone
tool from its own immutable snapshot while importing the four recorded
controller modules from the old source.

The frozen tooling snapshot is `8b6203057536f22e5707`. Three geometry tests
and 13 validation tests passed before staging. They verify the scoped path
edit, upstream rounding, exact baseline prerequisite, both focal-pair passage
conditions, full-window collision accounting and actual demand/pose handling.

Both arms start at 25200 and end at 25666: 466 one-second steps. Seed 41242,
60-second warmup, 10-second control interval, 450-second prediction horizon,
direct execution, zero cooldown, native collision settings and no-teleport
settings remain unchanged. The new SUMO configuration references the original
route file and a separate bilateral network; its begin/end remain 25200/28800,
with the runner enforcing the fixed endpoint.

One task requesting one CPU and 8192 MB runs two arms sequentially:

1. Original network prerequisite: reproduce all 23 retained action records,
   two collision reports and 108 three-stage detailed observations exactly
   against passing original replay `t90912`. Require complete duration and no
   teleports. A failure stops before bilateral simulation.
2. Bilateral network: complete all 466 steps and all 108 detailed observations
   from 25630 through 25665; require zero native collision events/incidents
   over the entire 466 seconds and zero teleports. Count collisions before
   the detailed window as well.

Prespecify actual incoming→internal lane(s)→outgoing passage by 25666 for all
four vehicles involved in the two retained incidents:

| Pair | Vehicle | Required internal lane(s) |
|---|---|---|
| Original straight | `102219_396_0` | 1_1 |
| Original left | `129962_409_0` | 13_0, 24_0 |
| V154C mirrored straight | `121463_406_0` | 11_1 |
| V154C mirrored left | `168358_425_0` | 3_0, 20_0 |

Read their actual routes from the original server-side route XML and cross-check
available retained route observations. Track passage over all 466 seconds.
Report the two pair conditions separately; both must pass for overall PASS.
This prevents non-insertion or unfinished passage from being counted as a
successful collision test. The window is not extended in response to late
passage or any other observed result.

For both waiting lanes, retain actual passenger polygons, dimensions, speed,
position and clearance for every reconstructible after-step observation in
the detailed window. Keep stationary and moving evidence distinguishable;
these measurements explain geometry and do not add a tuned acceptance
threshold. Retain endpoint traffic metrics, native collision records and the
first action/sample difference. Geometry may change preceding traffic and
controller feedback under the same parameters; action equality is required
only for baseline reproduction.

## Artifacts and interpretation

Protocol: `tsc-v154d-cologne-bilateral-waiting-geometry-validation-v1`.
Artifact directory: `tsc_v154d_cologne_bilateral_waiting_geometry_20260909`.
Inventory existing matching tasks before submitting one new paired task.
Keep route, network and model files on the server. Retrieve the small geometry
report and validation JSON; use logs if execution fails.

Zero collisions with completed focal passage establishes a bounded paired
validation result. Positive waiting-body clearance supports the local geometric
mechanism. Neither alone establishes elimination of all 18 full-rollout
incidents, service superiority over PhasePressure, efficacy of new occupancy
equations, or a source-transfer benefit. Preserve any failed condition without
changing the shift, thresholds or fixed endpoint after observing dynamics.
