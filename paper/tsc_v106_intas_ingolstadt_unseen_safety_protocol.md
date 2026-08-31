# TSC v106: InTAS Ingolstadt independent unseen-city safety protocol

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This protocol is frozen before the fixed InTAS repository is acquired and
before any SUMO 1.22 parser, safety, source-selection, controller, or efficacy
outcome is inspected. Ingolstadt has not been used to develop or select the
v93/v98 causal source components, strict target-only anchor, B100 offline
selector, pressure guard, controller parameters, or efficacy thresholds.

InTAS is a target evaluation environment, not the method's uncalibrated
auxiliary simulator. Its authors modeled demand with activitygen, real traffic
information, and demographic data; selected a rerouting probability by
comparison with real observations; and report validation at 24 measurement
points. Those target calibration records, detector traces, and optimization
outcomes are not method inputs. They make InTAS a stronger pseudo-ground-truth
environment but cannot be used to fit, select, or guard the zero-shot method.
B100 receives only the target-offline transition budget defined by the existing
transfer protocol.

## Frozen source and scope

- Dataset: *InTAS - The Ingolstadt Traffic Scenario for SUMO*.
- Repository: `silaslobo/InTAS`, master commit
  `0f7951ba01dda8483f0a852f2c3e4ff0d8a1c0ee`.
- Paper: Lobo et al., *SUMO Conference Proceedings* 1, 73-92,
  DOI `10.52825/scp.v1i.102`.
- License: GPL-3.0 for the repository; the proceedings article is CC BY 3.0.
- Frozen Git tree: 35 blobs and 983,190,873 uncompressed blob bytes.
- Network: complete `scenario/ingolstadt.net.xml`.
- Demand: every route source named by `InTAS_buildings.sumocfg`: pedestrian
  demand, the complete public-bus flow/timetable file, and all 22 files from
  `InTAS_001.rou.xml` through `InTAS_022.rou.xml`.
- Facilities: complete bus-stop and parking definitions required by the route
  demand.
- Traffic lights: every signal and program embedded in the published network.
- Published scope: full 24 hours, approximately 87,500 vehicles, real bus
  routes/timetable, 20 modeled real traffic-light systems, and validation at 24
  traffic measurement points with reported NRMSE 0.33.

No route file, vehicle, person, bus service, departure interval, edge, traffic
light, or vehicle type may be sampled, scaled, repaired, or removed. Passive
polygons, E1 output definitions, GUI settings, and output destinations may be
omitted because they do not alter demand, topology, signals, or dynamics.

## Static admission

Acquisition preserves the complete fixed Git commit and records archive size,
SHA-256, entry inventory, and Git-tree identity on shared cluster storage.
Static packaging must:

1. resolve every file referenced by the published configuration;
2. preserve all 24 route sources (pedestrian, bus, and 22 road-demand files);
3. require unique vehicle, flow, trip, route, person, and person-flow identities
   wherever the SUMO schema requires global uniqueness;
4. expand flow/person-flow counts deterministically for conservation reporting
   without rewriting the published demand;
5. require valid route, edge, stop, parking, vehicle-type, and signal references;
6. confirm departures across the complete 00:00-24:00 source interval;
7. record exact vehicle, flow, person, bus, edge, lane, junction, signal, and
   traffic-light-program counts; and
8. freeze the strict SUMO package manifest before microscopic execution.

Any duplicate identity, missing reference, parser error, unsupported legacy
construct, demand loss, or incomplete time coverage rejects InTAS before a
simulation result. No `duarouter --repair`, route substitution, or permissive
parser option is allowed.

## Strict execution contract

The native 0.1-second step, complete route set, car-following model, parking
maneuvers, pedestrian model, signal programs, and published dynamic rerouting
probability/period remain unchanged. The strict headless configuration may
remove passive outputs, set deterministic seed `5107`, set
`max-depart-delay=-1` so insertion delay cannot silently discard demand,
enable junction collision checks, change `collision.action` to `warn`, disable
time-based teleportation, and extend the completion cap to 108,000 seconds.
Demand scale is one and route errors are fatal.

Admission uses libsumo 1.22 only. Collision and teleport events are checked at
every 0.1-second simulation step. Active and pending populations may be sampled
less frequently for runtime efficiency, but loaded, departed, arrived,
unfinished, and final populations are exact. The run stops at the first
collision or teleport, or when no vehicle/person remains expected. Reaching the
fixed cap with any running, pending, or expected entity fails even when
teleportation is disabled.

## Sequential gate

1. Acquire the complete fixed Git commit directly to shared cluster storage.
2. Complete the static full-network/full-route audit and freeze the package
   manifest.
3. Run the complete native-signal condition with seed `5107`; require exact
   demand loading and completion, zero unique junction-collision incidents,
   zero teleport events, and no silent max-depart-delay loss.
4. Any source identity, parser, demand, reference, signal, collision, teleport,
   insertion, completion, or conservation failure rejects Ingolstadt before
   method evaluation. No simulator, route, demand, signal, vehicle, seed, or
   threshold amendment follows the first result.
5. Only after admission may Ingolstadt enter the unchanged causal transfer
   ladder: source-only zero-shot components, B100 target-offline source
   selection with target-only fallback, strict target-only CFCMT, selected
   causal mixture, dense/H2O+-style residual, simulator-only MPC, native fixed
   signal control, phase pressure, and MaxPressure-style baselines.
6. Source selection uses seed-blocked target-offline folds only. Closed-loop
   confirmation uses disjoint seeds. Target-only fallback is an accepted
   negative-transfer decision.

No Ingolstadt result is yet available under this protocol.
