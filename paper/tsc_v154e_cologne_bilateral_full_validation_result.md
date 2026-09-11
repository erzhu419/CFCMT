# V154E Cologne Bilateral Geometry: Full-Duration Result

**The frozen 3600-second validation is PASS.** With the unchanged V154D
bilateral waiting geometry, the original rigid controller completes the full
Cologne scenario with **zero native collision events, zero collision incidents
and zero teleports**. All four prespecified focal vehicles traverse the
junction on their original routes. The original-network baseline reproduces
exactly and has 18 collision incidents.

The service cost is modest in this run: four fewer arrivals out of 2015 due
vehicles, mean waiting increased by 1.2496 seconds, and system vehicle hours
increased by 0.1731%. The full-run collision outcome is a substantial positive
result for this fixed geometry and controller comparison.

Task `t90988` ran the original and bilateral networks sequentially under
protocol `tsc-v154e-cologne-bilateral-full-validation-v1`, from 25200 to 28800.
The bilateral network is the existing V154D package: lane lengths remain
13/24 = 8.48/20.05 m and 3/20 = 8.37/19.83 m. Its 12-attribute geometry delta
was retained without rebuilding or shifting either waiting point.

## Exact Reproduction And Completed Passage

The original-network prerequisite reproduces **all 77 metrics fields** and
the deterministic originator diagnostics exactly against corrected original
run `t90756`. This includes all **175 retained action records**, all **20
retained collision reports**, traffic metrics, and execution audits. The
baseline completes all 3600 one-second steps without teleportation.

Within the new bilateral full run, the first 466 seconds also reproduce
V154D exactly: **23 retained actions, 108 three-stage physical observations
and all four passage prefixes** match. This establishes that the longer run
continues the previously observed bilateral trajectory under the same inputs.

| Bilateral focal vehicle | Internal milestone(s) | Outgoing edge reached |
|---|---|---|
| Original straight `102219_396_0` | Lane 1_1 at 25667 | `-28198821#4` at **25681** |
| Original left `129962_409_0` | Lane 13 at 25631; lane 24 at 25632 | `32038051#0` at 25633 |
| Mirrored straight `121463_406_0` | Lane 11_1 at 25622 | `32038056#0` at 25627 |
| Mirrored left `168358_425_0` | Lane 3 at 25622; lane 20 at 25631 | `32324544#0` at 25635 |

The original straight focal vehicle enters its internal lane one second after
V154D's endpoint and reaches its outgoing edge at 25681. Its previously
unfinished passage is now directly observed. **V154D remains FAIL at its
original endpoint of 25666; V154E passes its separately frozen full-duration
conditions.**

## Collision And Service Comparison

| Quantity | Original network | Unchanged bilateral network | Change |
|---|---:|---:|---:|
| Simulation duration | 3600 s | 3600 s | Same |
| Native collision events | 36 | **0** | −36 |
| Native collision incidents | 18 | **0** | −18 |
| Starting / ending teleports | 0 / 0 | **0 / 0** | Same |
| Due demand | 2015 | 2015 | Same |
| Departed vehicles | 2015 | 2014 | −1 |
| Arrived vehicles | 1997 | 1993 | −4, −0.2003% |
| Active vehicles at endpoint | 18 | 21 | +3 |
| Pending insertion at endpoint | 0 | 1 | +1 |
| Mean tripinfo waiting | 19.712655 s | 20.962264 s | +1.249609 s, +6.3391% |
| System vehicle hours | 41.079722 | 41.150833 | +0.071111, +0.1731% |

Waiting includes all departed vehicles, including unfinished tripinfo records
at the horizon, and excludes the pending insertion vehicle. System vehicle
hours include active and pending vehicles under the existing 3541-sample
post-warmup accounting. Thus the pending vehicle is represented in the
system-wide service metric even though it has no departed-vehicle tripinfo.

The bilateral run has 256 phase switches versus 253 originally, and 264
occupancy-clearance extension seconds versus 237. These are recorded traffic
and executor responses to the fixed geometry change. The collision reduction
is accompanied by slightly higher waiting and nearly unchanged total system
vehicle hours.

## Complete Original Incident Inventory

The new read-only observer retains all **36 original native reports** and
groups them into **18 incidents** using the evaluator's existing key: sorted
participant IDs, native collision type and native reported lane. Its event,
incident and collision-step totals agree with the evaluator. Classifications
use the participant lanes in each incident's first report, so later reports
after a vehicle moves onto an outgoing lane do not change the incident's
movement pair.

| Original incident movement pair | Complete incident count |
|---|---:|
| Waiting lane 13 / straight lane 1_1 | **15** |
| Mirrored waiting lane 3 / straight lane 11_1 | **2** |
| Lane 16 / continuation lane 22 | **1** |
| Total | **18** |

All 18 incidents now have observed participant lanes; the eight incidents
previously outside the 20-report retention cap are classified. Seventeen
incidents occur on the two movement pairs addressed by the waiting geometry
repair. The remaining original incident at 26701 involves moving vehicles
on lanes 16 and 22, travelling at approximately 8.3501 and 8.6725 m/s.

The bilateral inventory is empty over all 3600 seconds, and its zero totals
agree with the evaluator. The moving 16/22 incident also does not recur in
this changed trajectory.

Inventory observations record collision-time vehicle poses and executor
states immediately after the simulation step, before executor advancement.
The unchanged evaluator's capped debug reports remain sampled after executor
advancement and are compared to their original reference separately. This
preserves both the exact baseline check and the complete collision-time
record.

## Runtime And Artifacts

The controller modules load from frozen source `85b2b43cd12832c5a1dd`; the
standalone full-validation tooling uses snapshot `95293dee78371b53157d`.
Both arms use SUMO 1.22.0, the original rigid model with stored-feature-name
binding, the original occupancy equations, seed 41242, a 60-second warmup,
10-second control interval, 450-second prediction horizon, direct execution
and zero cooldown. Demand, vehicle dimensions, collision handling and signal
execution rules retain their original values.

All **16 focused implementation tests** passed before staging. The paired
simulation took **10.8436 seconds** in one task requesting one CPU and
8192 MB RAM. Only the **756,788-byte result JSON** was retrieved. The compact
paper artifact was derived locally. Network, route and model files stayed on
the server; temporary tripinfo files were generated and removed there.

- [Frozen protocol](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154e_cologne_bilateral_full_validation_protocol.md)
- [Paired full result and uncapped collision inventory](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v154e_cologne_bilateral_full_validation_20260909/paired_full_v1/result.json)
- [Compact paper artifact](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154e_cologne_bilateral_full_validation_v1.json)
- [Launch record](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v154e_cologne_bilateral_full_validation_20260909/launch_v1.json)
- [Retained V154D result](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154d_cologne_bilateral_waiting_geometry_result.md)

## Limitations And Next Comparison

This PASS applies to one prespecified seed, one Cologne scenario and its
original 3600-second duration. The two repaired waiting geometries have
direct clearance evidence from V154C/V154D and the complete bilateral run has
zero native collisions. The unmodified 16/22 movement pair's non-recurrence
does not establish that its geometry or collision mechanism has been directly
repaired: the changed traffic and controller feedback also change encounter
timing. These results do not establish safety for other seeds or cities,
superiority over PhasePressure, benefit from the newer occupancy equations,
or source-transfer efficacy.

The next comparison should keep this bilateral geometry fixed for both the
rigid controller and PhasePressure and use a seed roster specified before
execution. That will evaluate whether the collision outcome persists and
whether the service difference is attributable to controller choice under
the same repaired network. No new geometric adjustment or seed selection is
part of V154E.
