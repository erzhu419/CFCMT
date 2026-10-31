# Boston v18 Microscopic Admission Result

## Decision

Boston package v18 is **rejected for microscopic TSC evaluation**. It passed
static package checks and complete-route admission, but failed the frozen
zero-collision trigger at simulation time `17,265 s`. The failure was reproduced
exactly by a bounded read-only diagnostic. No v19 parameter or topology repair
is authorized, and no full-day admission will be launched from v18.

The diagnostic task reports `passed=true` only because it reproduced the
predeclared collision identity and trace contract. It does not admit the
package.

## Evidence Chain

| Stage | Task | Result | Canonical artifact SHA-256 |
|---|---|---|---|
| v18 package construction | `t90219` | Passed static admission | `f5a527ab537b34c0676fefecdecbfa4175291b596880cea3bb5dd19f6d5a4fb3` |
| Complete-route admission | `t90226` | All `3,806,510` trips passed | `3884dab564c4e92ffc92b4368b815bae6672729163b8add78be7ad0eb798a2dc` |
| Trigger admission | `t90227` | Rejected at `17,265 s` | `d2f82ff1e325c4d60d94e566b2c78847b011efe4760e3552a321784061bce401` |
| Collision-lane static audit | `t90328` | No static lane merge; downstream priority relation resolved | `82b703a347620c52a7430dad59a659f2610d9d0d8d03dd33aa6be2dd829ce45a` |
| Network-wide visibility audit | `t90331` | Focus visibility is not unique | `ca06cd976efd58d4bbb3e3b7278b5c5e2dffe06101b98072dfb0d61112433a32` |
| Read-only dynamic reproduction | `t90329` | Exact time, lane, collider and victim reproduced | `0a0d950f404433260a95484c4090ad0dcf196a1c17693e537a19169bd1c54777` |
| Original derived kinematic summary v1 | local deterministic builder | Retained; gap interpretation superseded by v2 | `892e073da2de47a3fc208ca39eaf90075a01fe3f6a1b01d47c6076a79e033bb6` |

The artifacts are under
`cf_h2o/results/cluster/tsc_v150_source_identifiability_20260908/`.

The corrected derived artifact is
`boston_v18_collision_diagnostic_v1/trace_summary_v2.json`, generated from the
same retained `result.json` with `scripts/data/summarize_sumo_collision_trace.py`
under `sumo-collision-trace-summary-v2`. The raw diagnostic and original
`trace_summary.json` remain unchanged.

## Reproduced Sequence

SUMO registered a same-lane rear collision event between collider `271533` and
victim `289458` on `591691873#0_1`. This event violated the frozen minimum-gap
criterion; the recorded vehicle bodies did not overlap.

| Time (s) | Collider | Victim | Observed relation |
|---:|---|---|---|
| `17,263` | lane 1, position `21.930 m`, speed `12.860 m/s` | lane 0, position `24.324 m`, speed `15.856 m/s` | Vehicles are on parallel lanes. |
| `17,264` | lane 1, position `34.411 m`, speed `12.481 m/s` | changes from lane 0 to lane 1, position `41.699 m`, speed `17.375 m/s` | The bumper gap is `2.288 m`, leaving `0.788 m` beyond the collider's `1.5 m` minimum gap. |
| `17,265` | acceleration `-0.828 m/s2` | acceleration `-7.068 m/s2` | The bumper gap is `0.943 m`, a `0.557 m` deficit against the minimum; SUMO registers the collision event. |

SUMO's `getLeader` distance already excludes the follower's `minGap`. The v1
summary treated that clearance as a bumper gap and compared it against
`minGap` a second time. Version 2 reports both quantities explicitly:

`bumper_gap_m = raw_leader_clearance_m + collider_min_gap_m`.

The saved lane positions independently give `41.698856 - 5 - 34.410517 =
2.288338 m` at `17,264 s` and `52.005571 - 5 - 46.062779 = 0.942792 m` at
`17,265 s`. Only the latter is below `1.5 m`. The static lane-context audit
already recorded this final positive bumper gap and absence of body overlap.
The corrected interpretation matches the
[SUMO 1.22 getLeader implementation](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSVehicle.cpp#L6358)
and the [documented collision criterion](https://sumo.dlr.de/docs/Simulation/Safety.html#collisions).

The victim's braking exceeded its declared comfortable deceleration of
`4.5 m/s2` but remained below its declared emergency deceleration of
`9.0 m/s2`. The collider never exceeded comfortable deceleration in the
observed window. The watched conflicting major lane `9114638_0` was empty in
all 26 one-second samples. Vehicle `270873` entered the collision lane behind
the collider at the final sample and was not the collider's leader.

## Static Context

The collision lane has a unique upstream and downstream route mapping. Its two
lanes map to two distinct downstream lanes, so there is no static 2-to-1 merge.
At downstream junction `67457199`, lane 0 continues as a major movement (`M`),
whereas the lane-1 route is correctly encoded as a minor movement (`m`) yielding
to the major movement from `9114638_0`.

The lane is `59.22 m` long, and the victim was `7.214 m` from its downstream
boundary in the collision sample. The relevant minor connection has explicit
visibility `9.0 m`. A whole-network audit found the same value on 605 minor
connections and on 255 of 4,538 connections matched by state, direction and
incoming speed (`5.619%`).

## Interpretation and Model Boundary

The published Boston configuration is a mesoscopic scenario
(`mesosim=true`, `2 s` step, route errors ignored, collision action disabled).
The CFCMT admission protocol instantiates it as a strict microscopic scenario
(`1 s` step, route errors rejected, junction collisions checked). The source
network contains zero internal edges. The native configuration also starts at
`14,400 s`, permits departure discards after `1,200 s` and teleports after
`360 s`; the complete-demand microscopic protocol starts at `7,201 s` and
disables both mechanisms. This conversion therefore changes the operational
requirements as well as the simulation model.

SUMO's mesoscopic model uses coarser junction and lane-changing dynamics and is
more tolerant of network modelling errors. However, simulation without internal
links is also a supported simplified microscopic model: vehicles still obey
right-of-way rules while intersection traversal and blocking are omitted.
The absence of internal edges does not by itself establish why this collision
event occurred or that microscopic evaluation is impossible.
See [SUMO Meso](https://sumo.dlr.de/docs/Simulation/Meso.html) and
[SUMO Intersections](https://sumo.dlr.de/docs/Simulation/Intersections.html#internal-links).

The trace establishes a lane change followed by strong braking and a minimum-gap
violation. One specific unresolved explanation is the change from the lane-0
major approach to the lane-1 minor approach: SUMO can decelerate for a minor
connection before reaching its visibility distance even when no prioritized
vehicle is nearby. Thus, the empty watched major lane does not exclude
precautionary junction braking. The trace does not expose the internal reason
for the final braking command, and the visibility inventory establishes
frequency rather than dynamic safety.
See [SUMO vehicle speed at intersections](https://sumo.dlr.de/docs/Simulation/VehicleSpeed.html#intersections).

The supported conclusion is **failure of v18 under the current microscopic
admission protocol**. No new localized right-of-way defect has been identified;
mesoscopic-to-microscopic model compatibility remains an explanation to
investigate, not a uniquely established cause.

## Frozen Consequences

1. Do not change `tau`, `minGap`, lane-change aggressiveness, visibility,
   collision thresholds or collision handling to make v18 pass. Those would be
   post-observation behavioral calibration, not a declared network repair.
2. Do not launch v18 full-day admission because its prerequisite trigger failed.
3. Exclude Boston from the admitted microscopic external-city cohort and retain
   this rejection as negative package-admission evidence.
4. Any reconsideration of Boston requires a separately declared protocol. A
   bounded reproduction of the lane-change and braking interaction can first
   distinguish an implementation or time-discretization issue from a network
   representation issue. Reconstructing internal junction geometry and defining
   a native mesoscopic environment remain possible separate projects; neither
   is established as the necessary or sufficient repair for this event.

Boston admission is independent of the seven-city source-transfer experiments.
Its rejection neither rescues nor negates the rigid/action-contrast CFCMT result.
