# TSC v107: Chicago passenger-graph implementation correction

Date frozen: 2026-08-31 (Asia/Shanghai), before the v3 static scan.

## v2 diagnostic outcome

The v2 streaming implementation activated the frozen projection dependency and
completed its network parse, but rejected before writing an anchor set because
community area 1 had zero anchor candidates. No demand, safety, source-selection,
controller, or efficacy result was available.

A source-free static diagnostic found:

- 290,495 non-internal passenger-permitted edges;
- only 7,594 edges in the implementation's largest strongly connected component;
- a component geographic extent of approximately
  `[-87.9973, 41.7027, -87.5270, 41.9857]`; and
- a frozen community-area extent of approximately
  `[-87.9401, 41.6445, -87.5241, 42.0230]`.

The OSM network therefore overlaps Chicago correctly. The fragmentation came
from applying the 15 m **anchor** length criterion before SCC construction.
Short netconvert split edges are road-graph connectors; removing them before
SCC analysis disconnects otherwise reachable long endpoint edges.

## Frozen v3 correction

The parent protocol says that the largest strongly connected component is
computed over non-internal passenger edges and that demand endpoints are chosen
from eligible anchors inside that component. The v3 implementation restores
that order:

1. include every shaped, positive-speed, passenger-permitted non-internal edge
   when constructing the directed connection graph;
2. compute the largest strongly connected component over that full graph; and
3. apply the unchanged 15 m endpoint-quality criterion only when forming the
   area-specific anchor candidate set.

The correction does not relax the requirement of 32 selected anchors in every
one of the 77 community areas. It adds fail-closed per-area candidate counts to
any rejection message. OSM bytes, SUMO network bytes, signal programs,
projection, passenger permissions, connection directions, capacity score,
farthest-point selector, and all later demand/safety/efficacy gates remain
unchanged. v3 again adopts the byte-identical successful v1 netconvert output.
