# V118r ETH Boston SUMO 1.22 Zipper Compatibility Amendment

## Trigger

The complete-demand microscopic admission terminated at simulation time 9,518
s with:

```text
FatalTraCIError: Zipper junctions with more than two conflicting lanes are not supported (at junction '61394761')
```

Before the exception, 11,362 vehicles had loaded, 6,425 had departed and 4,876
had arrived. No controller outcome was observed and no efficacy comparison was
performed.

## Root Cause

SUMO 1.22 raises this exception in `MSLink::getZipperSpeed` when a zipper link
has more than one foe link. The published Boston network contains 477 zipper
junctions. A structured audit of their request logic found 138 junctions with
at least one request whose `response` bitset contains two foe links;
`61394761` is one of them. The other 339 have at most one response foe per
request.

The v5 package changed those 138 junction `type` attributes to `priority`, but
left all 551 associated connection states as `state="Z"`. SUMO constructs the
microscopic link behavior from this connection state: its 1.22 implementation
continues through the zipper-specific path when the link state is `Z`,
independently of the already changed junction attribute. At junction
`61394761`, all four connections retained `Z`; two request rows each listed two
response foes. The 600 s preflight did not reach a vehicle interaction at this
junction, whereas the full run did at time 9,518 s.

## Frozen Remediation

The first attempted remediation asked SUMO 1.22 `netconvert` to apply 138
plain-node overrides. The admission gate rejected that output: although all
lane IDs were retained, `netconvert` changed 177,706 lane permission strings
and re-rounded 73,022 lane lengths. No simulation result from that package is
admissible.

The second attempted remediation (package v5) performed a structured
`zipper`-to-`priority` junction-type substitution. Its static audit was too
narrow because it did not include connection state. The full admission exposed
that omission, and no v5 simulation result is admissible.

Package v6 identifies the same 138 unsupported junctions and jointly changes:

- their junction `type` attributes from `zipper` to `priority`; and
- the 551 corresponding connection states from zipper `Z` to minor priority
  `m`, because every corresponding request row has at least one response foe.

The connection-to-request mapping follows SUMO's junction connection order.
The request and foe matrices themselves remain unchanged, so the published
conflict relation is retained while unsupported zipper speed adaptation is
replaced by ordinary minor-link yielding. Topology, geometry, permissions,
signal programs and demand are not rebuilt.

Admission requires all of the following before simulation:

- zero remaining zipper requests with more than one response foe;
- exact XML element order and exact attributes except the 138 declared junction
  type and 551 declared connection-state substitutions;
- exact preservation of external edge and lane semantics;
- exact preservation of external lane-to-lane movements;
- exact preservation of traffic-light programs; and
- junction-type changes equal exactly the declared unsupported set.

The v6 static package gate passed with 3,806,510 trips, 161,096 external edges,
189,716 external lanes, 352,275 external lane-to-lane movements and 4,203
traffic-light programs. Its local manifest copy is
`cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/package_v6/package_manifest.json`
(SHA-256
`7dae903cf5f3b77263140ac4509d4d6bc9c22c1a2394def35b9a6e21b729636a`).
This is pre-simulation input admission only; route, operational and complete
microscopic admission remain separate gates.

The package-v6 route gate subsequently routed all 3,806,510 trips with SUMO
1.22 `duarouter`, TAZ routing enabled, 20 routing threads, return code zero and
`ignore-errors=false`. Its result is
`cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/route_admission_v4/result.json`
(SHA-256
`01cb52cb21595f8eaed9511057671906d1cee3c5b3425b9d00250e9c29053807`).
This authorizes operational preflight only; it is not evidence of microscopic
safety or controller efficacy.

The package-v6 3,000-s trigger-window preflight then ran from simulation time
7,201 through 10,201, crossing the former failure time 9,518. It completed in
4,151.56 s with no runtime exception, collision, starting teleport or ending
teleport and exposed exactly 4,203 controllable traffic lights. At the window
boundary, 11,390 vehicles had departed and 9,297 had arrived. The result is
`cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/operational_trigger_window_v4/result.json`
(SHA-256
`3b585570eba671969b6a41341327b787b4cc3f8b636db343096b4d4199a95fd1`).
This is a passing operational gate, while
`scientific_admission_passed=false` remains explicit.

The package configuration freezes 15 synchronized SUMO rerouting threads.
The earlier full run declared only two CPU cores and the trigger run declared
four, so both were under-resourced relative to their executable configuration.
The package-v6 complete admission is scheduler task `t88337`; it declares 16
cores, 24,576 MB RAM under a new v6-complete RAM history family, and permits
only `node001,node002,node003,node005,node006`. Its signature is
`CFCMT/v118r/eth-boston-full-admission-v4-junction-link-state`. The task is the
only result that can authorize complete microscopic admission; it remains
pending at this amendment.

The source archive and original `BOS.net.xml` remain unchanged. Demand,
departure times, TAZ schema canonicalization, seed, simulation horizon,
controller arms and outcome gates are unchanged. The simulator uses only the
package-owned `microscopic_network.net.xml` produced by this declared
compatibility normalization.

## Technical basis

- SUMO 1.22 `MSLink::getZipperSpeed` rejects more than one foe link:
  <https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSLink.cpp#L1939-L1948>
- SUMO documents `m`/`M` as minor/major connection states and reserves zipper
  behavior for link state `Z`:
  <https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html#plain-connections>
