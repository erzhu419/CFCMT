# TSC v107: Chicago full-network static outcome

Date completed: 2026-08-31 (Asia/Shanghai)

## Outcome

Chicago passes the frozen v107 full-network static gate under implementation
v3. The admitted artifact is
`/home/zhengliang01/scheduleurm_work/CFCMT_DATA/chicago_v107_409aec2/network_v3`.
It uses SUMO 1.22.0 and the byte-identical network produced by the frozen OSM
conversion:

- network size: `362471296` bytes;
- network SHA-256:
  `45250772dcff2cd79bd4472e0657d09c20eb32be280a1099d5ba93b831ac42ae`;
- edges: `4,492,660` total and `1,538,949` non-internal;
- lanes: `4,655,236`;
- junctions: `1,050,674`;
- directed connections: `5,674,164`;
- traffic-light systems/programs: `4,960 / 4,960`;
- traffic-light phases: `30,664`.

The full passenger graph contains 409,848 shaped positive-speed non-internal
edges. Its largest strongly connected component contains 408,373 edges
(99.6401%). After applying the endpoint-quality length rule, 290,495 passenger
edges remain eligible as anchors. All 77 community areas pass: each receives 32
deterministically selected anchors, for 2,464 anchors total. Area 47 has the
smallest candidate pool (298); area 28 has the largest (4,312). The total number
of within-area anchor candidates is 122,795.

The v3 streaming scan took 209.93 seconds. The earlier v1 dependency failure and
v2 graph-order failure remain preserved and did not expose demand, safety,
source-selection, controller, or efficacy outcomes. Their corrections are
documented in the two pre-v2/v3 amendments. No OSM byte, SUMO network byte,
signal, area, or scientific admission threshold changed.

## Next gate

This is a static network PASS, not a safety or efficacy result. Demand expansion
must next account for every grouped TNP/taxi row, explicitly classify publicly
unlocatable endpoints, conserve every included trip, and route all seven full
days without repair before microscopic safety admission begins.
