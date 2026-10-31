# TSC v106: InTAS pre-execution identity freeze

Date frozen: 2026-08-31 (Asia/Shanghai)

This record was written after static input admission and before any InTAS
microscopic safety result was observed. It instantiates the sequential gate in
`tsc_v106_intas_ingolstadt_unseen_safety_protocol.md`; it does not amend the
source, demand, simulator, safety threshold, or method protocol.

## Frozen identities

- Parent protocol commit:
  `7988916d7be1387750da54708473ac1c30904cfb`.
- Source repository: `silaslobo/InTAS` at commit
  `0f7951ba01dda8483f0a852f2c3e4ff0d8a1c0ee`.
- Acquisition protocol: `cfcmt-intas-fixed-git-acquisition-v1`.
- Package implementation commit:
  `e210246cc6b7ed2a7d482907b2e7ec53ece7e15f`.
- Admission implementation and committed manifests:
  `3edf7304d431524d516097432ada6e8d4ba15ee4`.
- Package protocol: `cfcmt-intas-full-day-sumo122-package-v1`.
- Package manifest SHA-256:
  `4f79a34946addde7efd0ba88b4a3318716de907f8a8b5d30b50ef096771fc282`.
- Admission protocol: `intas-full-day-strict-libsumo-safety-admission-v1`.
- Server package root:
  `/home/zhengliang01/scheduleurm_work/CFCMT_DATA/intas_v106_e210246/package_v1`.

The SHA-256 above is an execution identity: the admission runner rejects a
different manifest before starting SUMO.

## Static result

Static input admission passed with no failed gate:

- all 35 fixed Git blobs, 983,190,873 uncompressed bytes;
- all 22 private-road-vehicle route files plus the complete bus and pedestrian
  route sources;
- 185,923 private-road vehicles;
- 1,581 buses obtained by exact expansion of all 172 published flows;
- 254 pedestrians;
- 23,648 edges, 33,204 lanes, 4,687 junctions, and 98 traffic lights;
- zero duplicate vehicle identities and zero missing edge, lane, stop, route,
  vehicle-type, or signal references; and
- source departures spanning 00:00 through the final minute of the day.

No route, entity, interval, traffic light, or demand was sampled, scaled,
repaired, or removed.

## Frozen microscopic run

The first and only admission run uses:

- libsumo/SUMO 1.22.0;
- the complete native network, signal programs, and all 24 demand sources;
- native 0.1-second steps and published rerouting probability 0.82;
- deterministic seed 5107;
- demand scale 1;
- `max-depart-delay=-1` and time-based teleportation disabled;
- junction collision checks at every simulation step; and
- a fixed 108,000-second completion cap.

The run stops and rejects at the first collision or teleport. Otherwise it
must account exactly for 187,504 departed and arrived road vehicles and 254
departed and arrived persons, with no running, pending, or expected entity at
completion. Static PASS alone does not admit InTAS to efficacy evaluation.
