# TSC v104: Nagoya NUMo independent unseen-city safety protocol

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This protocol is frozen before any Nagoya SUMO 1.22 input, safety, source
selection, or controller outcome is inspected. Nagoya has not been used to
develop or select the v93/v98 causal components, strict target-only anchor,
B100 offline selector, pressure guard, controller parameters, or efficacy
thresholds.

The NUMo authors report zero collisions and 561 teleports in their SUMO 1.19
full-day run. Both figures are recorded in advance. The v104 admission asks
whether the complete published demand can finish with time-based teleportation
disabled. Zero reported teleports alone is not sufficient: all published
vehicles must depart and arrive by the fixed completion cap.

## Frozen source and scope

- Dataset: *Nagoya Urban Mobility (NUMo) Scenario*.
- Repository: `ToyotaInfoTech/numo`, frozen master commit
  `5409c4696ad9b4d81e0665d1de51e327beda0505`.
- Spatial scope: the complete 326 km2 Nagoya network in `nagoya.net.xml`.
- Demand: every vehicle in all 48 published half-hour route files from
  `routes/nagoya_00_00.rou.xml` through `routes/nagoya_23_30.rou.xml`.
- Traffic lights: every network signal and the complete published
  `nagoya_waut.add.xml` program schedule.
- Published statistics: 24 hours, 1,627,151 departed vehicles, 7,833
  junctions, 11,520 edges, 1,593 traffic lights, zero collisions, and 561
  teleports under SUMO 1.19.
- Licenses: ODbL/DbCL for the OpenStreetMap-derived network and CC BY 4.0 for
  the other published files, as stated by the repository.

## Execution contract

Acquisition preserves the complete fixed commit. Static admission requires
all files named by `nagoya.sumocfg`, all 48 route files, exact unique vehicle
identity, a complete 24-hour departure span, valid route-edge references, and
the full network and WAUT signal inventory. No route, vehicle, time interval,
edge, signal, or vehicle type may be sampled, scaled, repaired, or removed.

The strict headless configuration preserves the published network, demand,
WAUT programs, eager insertion, and dynamic rerouting settings. It may omit
output destinations, set the deterministic development seed to 5058, enable
junction collision checks, change `collision.action` from `none` to `warn`,
disable time-based teleportation, and extend the end time to 108,000 seconds
to provide six hours of drain time after the final departure. Demand scale is
one and route errors are fatal.

Admission uses libsumo 1.22. Safety events are checked every simulation step;
active and pending populations may be sampled at a coarser interval for
runtime efficiency, but final populations are exact. The run stops at the
first collision or teleport, or when no vehicle remains expected, and fails
if the fixed cap is reached with any running, pending, or expected vehicle.

## Sequential gate

1. Acquire the complete fixed Git commit directly to shared cluster storage.
2. Statically require the published 48 route files, exactly 1,627,151 unique
   route vehicles, valid edge references, full-day coverage, and at least one
   controllable traffic light.
3. Freeze the exact SUMO 1.22 package manifest and strict configuration before
   simulation.
4. Run the complete native signal condition with development seed 5058.
   Require exact demand loading, departure, arrival, and completion; zero
   unique junction-collision incidents; and zero teleport events.
5. Any identity, parser, demand, route, signal, collision, teleport,
   insertion, completion, or conservation failure rejects Nagoya before
   method evaluation. No simulator, route, demand, vehicle, or threshold
   amendment follows the first result.
6. Only after admission may Nagoya enter the unchanged causal transfer ladder:
   zero-shot, B100 target-offline source selection, strict target-only CFCMT,
   selected causal mixture, dense/H2O+-style residual, simulator-only MPC,
   fixed time, and pressure-rule baselines.
7. Source selection uses seed-blocked target-offline folds only; confirmation
   uses disjoint seeds. Target-only fallback is an accepted negative-transfer
   decision.

No Nagoya result is yet available under this protocol.
