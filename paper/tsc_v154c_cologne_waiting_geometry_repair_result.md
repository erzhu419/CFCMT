# V154C Cologne Waiting-Point Geometry Repair Result

**The frozen paired validation is FAIL.** The new driver exactly reproduced
the original baseline, and the lane-13 repair satisfied its static geometry
rule. The repaired simulation nevertheless retained one native collision
incident at the unmodified mirrored waiting point, and the original straight
focal vehicle had not left the controlled junction by the fixed endpoint.
Both are failures of the conditions frozen before this run.

Task `t90940` ran the baseline and repair sequentially under protocol
`tsc-v154c-cologne-waiting-geometry-validation-v1`. Each simulation covered
25200–25666, exactly 466 seconds. The window was not extended after the result.

## Baseline And Geometry Verification

The baseline prerequisite passed exact comparison of all **23 retained action
records, two collision reports and 108 three-stage observations** against
`t90912` and its original `t90756` reference. It also completed all 466 steps
without teleportation. This confirms that the new driver reproduced the
retained trajectory before running the repaired network.

The repair report confirms exactly six attribute changes: shape and length of
lanes `:cluster_357187_359543_13_0` and `:cluster_357187_359543_24_0`, and the
coordinates of their internal waiting junction. Lane 13 changes from 8.76 to
8.48 m, while lane 24 changes from 19.77 to 20.05 m. The complete physical path
and combined logical length remain unchanged; all eight reported invariants
pass. Connections, conflict responses, signals and the mirrored 3→20 waiting
point retain their original network attributes.

For the frozen 4.3 × 1.8 m passenger body with its front exactly at the new
split, clearance from the opposing straight swept strip is **+0.002367516 m**.
That is the pre-simulation geometric result. The baseline's observed stationary
focal body intrudes into the strip by **0.056508558 m**. The repaired focal
left-turn vehicle does not stop in the retained detailed window, so the paired
result contains no repaired stationary sample for that vehicle.

The corrected side nevertheless has direct pose evidence from another vehicle.
Vehicle `153000_419_0` approaches the new waiting boundary during 25659–25665.
At 25665 its front is at 8.378916346 m and speed is 0.000365513 m/s, with actual
body clearance **+0.035645676 m**. This closely matches the pre-simulation
prediction of +0.035618127 m for a 0.101 m stopping offset. All eight retained
after-step lane-13 observations with reconstructible passenger polygons have
positive clearance, with this final observation the minimum. This supports
the repaired waiting geometry under an observed approach to rest. The vehicle
is nearly stationary; the frozen stationary threshold is unchanged, and this
auxiliary observation does not turn either failed admission condition into a
pass.

![Original body intrusion and measured clearance at the repaired waiting point](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v154c_cologne_waiting_geometry_20260909/waiting_clearance_comparison.png)

The comparison uses the original stationary focal vehicle and a different,
nearly stationary vehicle in the repaired run. The blue region is the 1.8 m
straight vehicle's swept strip. It visualizes local body clearance, not a
recreated paired encounter. All eight measured repaired-side poses are retained
in [the observation summary](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v154c_cologne_waiting_geometry_20260909/changed_waiting_lane_observations.json).

## Paired Outcome

| Quantity | Original baseline | Repaired network |
|---|---:|---:|
| Simulation duration | 466 s | 466 s |
| Detailed three-stage observations | 108 | 108 |
| Native collision events / incidents | 2 / 1 | 2 / 1 |
| First native collision time | 25657 | 25629 |
| Starting / ending teleports | 0 / 0 | 0 / 0 |
| Departed vehicles | 322 | 322 |
| Arrived vehicles | 256 | 247 |
| Active vehicles at endpoint | 66 | 75 |
| Pending insertion at endpoint | 1 | 1 |

The repaired run completed normally and retained the full observation window.
Its collision outcome therefore cannot be attributed to missing simulation
time or to dropping vehicles through teleportation. Collision counts apply
to the entire 466-second simulation, including the collision at 25629, one
second before the detailed window begins. Native events and incidents are
reported separately: two retained reports represent one incident in each arm.

Both original focal vehicles were inserted and retained their original routes.
Their repaired passage evidence is:

| Focal vehicle | Incoming first seen | Internal milestones | Outgoing reached by 25666 |
|---|---:|---|---|
| Left turn `129962_409_0` | 25623 | Lane 13 at 25630; lane 24 at 25632 | Yes, `32038051#0` at 25634 |
| Straight `102219_396_0` | 25616 | Lane 1_1 at 25662 | No |

The left turn had already left the junction when the straight vehicle entered
its internal lane. The original focal collision was not observed in this
changed trajectory, but this does not isolate the effect of repaired body
clearance during the original encounter. The frozen requirement for both
vehicles to traverse their controlled junction is unmet.

## The Remaining Collision Is At An Existing Mirrored Defect

At 25629, stationary vehicle `168358_425_0` on
`:cluster_357187_359543_3_0` collided with `121463_406_0` travelling at
9.689219 m/s on `:cluster_357187_359543_11_1`. The report was retained again at
25630. The stationary front was at logical position 8.519 m on the unchanged
8.62 m waiting lane. Its actual passenger polygon at 25630 has **−0.049633247 m**
clearance to the 1.8 m straight vehicle's swept strip; its full longitudinal
projection lies within the finite straight segment.

This movement pair already collided in the original full `t90756` rollout:
at 25973, straight vehicle `152995_419_0` on lane 11_1 hit nearly stationary
vehicle `132821_410_0` on lane 3 at position 8.518871930 m. Those are different
vehicles and a later encounter. Together with the unchanged mirrored geometry,
this supports an existing mirrored waiting-body conflict being triggered
earlier under the changed traffic trajectory. It is not evidence that the
six-attribute edit introduced a new connection or conflict topology.

The static rule for movement 13 is satisfied, but repairing that movement
alone did not satisfy the full-window collision criterion. The remaining
mirrored defect provides a concrete reason for the collision failure; the
incomplete straight focal passage independently prevents a paired PASS.

## Controller Feedback And Scope

The first retained action already differs at 25260, the end of the 60-second
warmup. The original model chose candidate 6 with a stay request; the repaired
trajectory chose candidate 4 with an accepted switch. Recorded mean speed
changed from 11.445977 to 11.221109 m/s. There are 23 original versus 25 repaired
retained action records, with no exactly matching leading record. The first
retained physical sample also differs at 25630; detailed observations do not
locate the first physical divergence before then.

These are feedback differences under the same controller parameters. Both
arms use the original rigid model, seed 41242, 10-second control interval,
450-second prediction horizon, direct execution and zero cooldown. The four
recorded runtime modules are loaded from frozen controller source
`85b2b43cd12832c5a1dd`; the standalone driver comes from tooling snapshot
`d07d45b3d0700d993910`. The original occupancy equations remain in use, and the
repair has zero utility-gate selections and zero source-induced action changes.

## Limitations

This is one seed and one fixed 466-second window. Static clearance of the
modified waiting point is established for the specified passenger dimensions,
with measured positive clearance for another vehicle approaching that point.
The original focal pair no longer recreates its stationary encounter, and the
straight vehicle has not completed junction passage, so this run does not
establish causal elimination of that collision by body clearance. It also
does not establish full-rollout safety or service improvement: the bounded
repair has nine fewer arrivals and the same pending count. The original FAIL,
window, geometry shift and thresholds are retained.

## Execution And Artifacts

The geometry repair and paired validation tests total **13 passes**. The
simulation's scientific FAIL is independent of those implementation checks.

The paired runner took 3.689 seconds in one task requesting one CPU and
8192 MB RAM. The retrieved files were the 4,486-byte geometry report and
672,985-byte paired result JSON, totalling **677,471 bytes**. Network, route and
model files stayed on the server.

The next concrete target is the mirrored waiting point under the same
body-clearance rule, implemented and evaluated as a separate version. This
result retains the original one-sided geometry shift and fixed window.

- Frozen protocol: `paper/tsc_v154c_cologne_waiting_geometry_repair_protocol.md`
- Geometry report: `cf_h2o/results/cluster/tsc_v154c_cologne_waiting_geometry_20260909/geometry_report_v1.json`
- Paired result: `cf_h2o/results/cluster/tsc_v154c_cologne_waiting_geometry_20260909/paired_validation_v1/result.json`
- Launch and snapshots: `cf_h2o/results/cluster/tsc_v154c_cologne_waiting_geometry_20260909/launch_v1.json`
- Original observed baseline: `cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/collision_replay_v1/result.json`
- Original full rollout, including the later mirrored collision: `cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/rigid_feature_alignment_v1/result.json`
