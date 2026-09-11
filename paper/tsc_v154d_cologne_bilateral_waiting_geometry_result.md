# V154D Cologne Bilateral Waiting-Point Geometry Result

**The fixed-window validation is FAIL because the original straight focal
vehicle has not traversed the junction by 25666.** The bilateral repair
completed all 466 seconds with **zero native collision events, zero collision
incidents and zero teleports**. Both vehicles from the V154C mirrored collision
completed their required passage. The measured mirrored waiting-body clearance
also became positive.

Task `t90968` ran one original-network baseline followed by one bilateral
repair under protocol
`tsc-v154d-cologne-bilateral-waiting-geometry-validation-v1`. The V154C result
was reused as retained evidence. The original 25200–25666 simulation window,
25630–25665 detailed observation window and geometric shifts were retained.

## Geometry And Baseline

The independent network keeps the V154C 13→24 split exactly and applies the
same passenger-body clearance rule to the mirrored 3→20 split. With a
4.3 × 1.8 m passenger body and its front exactly at the split, the most
downstream feasible position is rounded upstream to the 0.01 m logical grid.

| Quantity | Primary 13→24 | Mirrored 3→20 |
|---|---:|---:|
| Waiting logical length, original → repaired | 8.76 → 8.48 m | 8.62 → 8.37 m |
| Continuation logical length, original → repaired | 19.77 → 20.05 m | 19.58 → 19.83 m |
| Geometric retreat | 0.280173624 m | 0.249888422 m |
| Front-at-split static body clearance | +0.002367516 m | +0.001530323 m |

The report confirms exactly 12 allowed attribute changes versus the original
network, of which the six primary-side changes equal V154C and the six new
changes belong to the mirrored side. Both complete physical paths and their
combined logical lengths remain unchanged. Each waiting prefix preserves its
original position mapping; each continuation absorbs the shifted length.
Connections, via links, conflict and yielding responses, signal plans, speeds,
demand and vehicle parameters retain their original values. All reported
geometry invariants pass.

The new runner reproduced the original baseline exactly: **23 action records,
two collision reports and 108 three-stage physical observations** match the
retained `t90912` replay and its `t90756` reference. It also completed all 466
steps without teleportation. Only after this prerequisite passed did the
bilateral simulation run.

## Fixed-Window Outcome

| Quantity | Original baseline | V154C one side, retained | V154D both sides |
|---|---:|---:|---:|
| Simulation duration | 466 s | 466 s | 466 s |
| Detailed observations | 108 | 108 | 108 |
| Native collision events / incidents | 2 / 1 | 2 / 1 | **0 / 0** |
| Starting / ending teleports | 0 / 0 | 0 / 0 | **0 / 0** |
| Departed vehicles | 322 | 322 | 322 |
| Arrived vehicles | 256 | 247 | 241 |
| Active vehicles at endpoint | 66 | 75 | 81 |
| Pending insertion at endpoint | 1 | 1 | 1 |

Collision counts cover the entire 466-second simulation, including the time
before detailed observation begins. The repaired runner completed normally
and retained the complete 108-observation window.

All four prespecified vehicles were observed and retained their original
routes. Their passage results are:

| Vehicle | Required internal milestones observed | Outgoing reached by 25666 |
|---|---|---|
| Original straight `102219_396_0` | No internal entry | **No** |
| Original left `129962_409_0` | Lane 13 at 25631; lane 24 at 25632 | Yes, `32038051#0` at 25633 |
| Mirrored straight `121463_406_0` | Lane 11_1 at 25622 | Yes, `32038056#0` at 25627 |
| Mirrored left `168358_425_0` | Lane 3 at 25622; lane 20 at 25631 | Yes, `32324544#0` at 25635 |

The mirrored-pair passage condition passes. The original-pair condition fails
solely because `102219_396_0` remains on incoming edge `-32038056#3` at the
endpoint. Its last detailed after-step observation, at 25665, places it on
lane `-32038056#3_1` at 340.459246 m with speed **5.575170 m/s**. It is moving;
the failure is unfinished junction passage within the fixed window.

## Actual Waiting-Body Clearance

The detailed observer reconstructs SUMO's passenger polygon from the actual
vehicle front and converted back position, then measures its signed distance
from the opposing straight vehicle's 1.8 m swept strip. Both reported minima
project entirely within the finite straight segment.

| Waiting side | Vehicle and sample | Speed | Actual clearance |
|---|---|---:|---:|
| Mirrored lane 3 | `168358_425_0`, 25630 | 0.000817815 m/s | **+0.036372646 m** |
| Primary lane 13 | `153000_419_0`, 25665 | 0.022504264 m/s | **+0.036526383 m** |

The mirrored observation uses the same focal left-turn vehicle that had
**−0.049633247 m** clearance at 25630 in V154C. Its body now has **+3.6373 cm**
of clearance, compared with **4.9633 cm intrusion** on the unchanged mirrored
geometry in V154C. This directly supports the local mirrored waiting-point
correction. Its speed is close to zero, but it does not meet the frozen
stationary threshold; both runs' actual pose and speed are retained.

On lane 13 the minimum measured clearance is **+3.6526 cm** for a different
vehicle, `153000_419_0`, approaching the already repaired boundary. There are
11 after-step lane-13 observations, seven with reconstructible bodies, and
two lane-3 observations, both reconstructible. Every reconstructible body in
these samples has positive clearance. The four unavailable lane-13 polygons
are retained as unavailable observations. Neither side has a sample meeting
the strict stationary threshold in this run.

## Controller Feedback And Execution

Both simulations import the four recorded controller modules from frozen
source `85b2b43cd12832c5a1dd`; the new standalone tooling comes from snapshot
`8b6203057536f22e5707`. They use SUMO 1.22.0, the original rigid model with
stored-feature-name binding, seed 41242, a 60-second warmup, a 10-second
control interval, a 450-second prediction horizon, direct execution and zero
cooldown. The original occupancy equations remain in use.

The first retained repaired action already differs from the original at
25260, and the first detailed physical observation differs at 25630. Each arm
has 23 retained action records, with no exactly matching leading record.
Geometry therefore changes traffic and subsequent controller feedback within
this fixed-model comparison.

The three geometry tests and 13 runner tests passed before staging:
**16 implementation tests in total**. The two sequential simulations took
**3.499 seconds** in a task requesting one CPU and 8192 MB RAM. The scientific
FAIL is separate from successful execution and implementation testing.

The retrieved geometry report is 13,240 bytes and the paired result is
800,684 bytes, totalling **813,924 bytes**. Network, route and model files
remained on the server.

## Limitations And Next Experiment

The bilateral repair removes native collisions from this one fixed window
and supplies positive measured body clearance at both waiting points. The
original straight focal vehicle has not yet traversed the junction, so the
prespecified four-vehicle passage requirement remains unmet and the overall
result stays FAIL. The mirrored vehicles also follow a changed encounter
timing. This result does not isolate collision elimination under the exact
original timing, establish full-rollout safety, or demonstrate service
improvement. Arrivals at the fixed endpoint decrease from 256 in the original
baseline to 241, while pending insertion remains one. No superiority over
PhasePressure, benefit of the newer occupancy equations or source-transfer
benefit is established here.

The original full `t90756` rollout reports 36 native collision events and 18
incidents. Its retained collision samples are capped at 20 reports,
representing ten incidents: eight involve lanes 13/1_1, one involves 3/11_1,
and one involves 16/22. Thus eight original incidents remain unclassified from
the retained sample alone. At 26701, the third retained movement pair includes
vehicle `229091_450_0` travelling at 8.350075 m/s on lane 16 and
`202023_438_0` travelling at 8.672521 m/s on lane 22. Both vehicles are moving;
that incident requires its own mechanism analysis and cannot be assigned to
the two waiting-point defects merely because it occurs at the same junction.

The next bounded experiment is a separate full 3600-second closed-loop
confirmation of this frozen bilateral network, with complete incident
inventory and endpoint service metrics. It should retain the same model,
seed, demand, physics and geometry, and freeze its conditions before launch.
The V154D window, geometric shifts and FAIL remain unchanged.

## Artifacts

- [Frozen V154D protocol](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154d_cologne_bilateral_waiting_geometry_protocol.md)
- [Geometry report](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v154d_cologne_bilateral_waiting_geometry_20260909/geometry_report_v1.json)
- [Paired validation result](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v154d_cologne_bilateral_waiting_geometry_20260909/paired_validation_v1/result.json)
- [Task launch record](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v154d_cologne_bilateral_waiting_geometry_20260909/launch_v1.json)
- [Retained V154C result](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154c_cologne_waiting_geometry_repair_result.md)
- [Original full rollout](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/rigid_feature_alignment_v1/result.json)
