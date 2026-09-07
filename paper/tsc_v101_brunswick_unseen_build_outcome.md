# TSC v101: Brunswick unseen-city build outcome

## Frozen source and build

- Source: DLR `sumo-scenarios`, commit
  `00f6eb479a9dc0fbaeb731495c11d48d7a9661d3`, complete `brunswick`
  subtree `04bc93e619e60faa95dc5b18432fb07a5434634c`.
- Acquisition: 66 tracked files, 234,720,146 bytes, streamed directly to
  shared cluster storage.
- Build runtime: SUMO 1.22.0 with the official Brunswick MIV build sequence.
- Compatibility changes were limited to two declared build migrations:
  the source OSM already contains `tram4.diff`, and the two stale connections
  targeting `299910664` were mapped to its unique SUMO 1.22 split edge
  `299910664#0`.
- GTFS build dependency: isolated `pyproj==3.7.1` wheel for CPython 3.10;
  the shared SUMO runtime and conda environment were not modified.

## Pre-simulation gate

The source contains 672,252 road geotrips spanning 74,670--187,250 seconds.
The official `fromgeo.duarcfg` uses `ignore-errors=true`. Under the frozen
SUMO 1.22 build it emitted 671,975 mapped trips and reported 277 source trips
with no valid directed route. Retention was therefore 99.9587952%, below the
predeclared exact-retention requirement of 100%.

The missing records were not confined to one route or one vehicle class. They
span 103,809--174,306 seconds and multiple passenger-car classes plus one
trailer. The build also produced 4,010 GTFS vehicles and 176 controlled
intersections, but those successful checks do not override the failed demand
gate.

## Decision

**REJECT before simulation.** No collision/teleport admission and no CFCMT,
baseline, zero-shot, or B100 controller evaluation is authorized for Brunswick.
The rejection does not alter the positive strict target-only Jinan confirmation;
it only means Brunswick cannot serve as the independent full-demand city under
the frozen no-drop protocol.
