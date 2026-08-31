# TSC v107: Chicago pre-demand identity

Date frozen: 2026-08-31 (Asia/Shanghai), before any trip expansion or routing.

The admitted `network_v3/anchors.json` is fixed at `857511` bytes with SHA-256
`3c3ee079d62ebc0a4df0f7798b464b1934f962b8a74caaf17742059937f8c805`.
It contains exactly 32 unique anchors in each of 77 areas. Demand preparation
rejects any anchor or network identity drift.

For each published TNP/taxi aggregate endpoint, a finite released centroid
inside one of the published 77 polygons takes precedence. It receives the eight
nearest same-area anchors, ordered by longitude-adjusted squared distance and
edge ID. If no released in-city centroid exists, a valid released community
area receives all 32 area anchors ordered by edge ID. Otherwise the endpoint is
publicly unlocatable. A trip is included only when both endpoints are locatable;
all other counts remain in source/day/missing-endpoint denominators.

Within each aggregate group, the Cartesian origin/destination anchor pairs are
ordered by origin and destination edge ID, with same-edge pairs removed. Vehicle
ranks cycle through that fixed pair list. Departures are evenly stratified at
`(rank + 0.5) / count` inside the released 900-second privacy bin. Seven calendar
days are assigned disjoint 86,400-second offsets for one later `duarouter` run;
the offsets are removed when routed files are split back into full days.

No aggregate count, timestamp bin, source identity, area, anchor, or demand scale
may be sampled or changed. Demand preparation is only a count/location gate and
cannot inspect traffic safety, controller, source-selection, or efficacy results.
