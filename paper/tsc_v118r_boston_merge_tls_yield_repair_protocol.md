# TSC V118r Boston Merge TLS Yield Repair Protocol

## Purpose

Repair the single localized control-priority defect exposed by Boston package
v6 without changing demand, topology, signal timing, controller state/action
spaces or scientific acceptance criteria.

## Declared transformation

For TLS program `joinedS_726` only:

| Phase index | Link index | Before | After |
|---:|---:|:---:|:---:|
| 10 | 12 | `G` | `g` |
| 12 | 14 | `G` | `g` |

In SUMO signal semantics, `G` grants priority and `g` requires the movement to
yield to conflicting priority traffic. Phase durations, yellow transitions,
all other phase characters, connections, lanes, edges and junctions remain
unchanged.

## Static admission

The structured repair completed on immutable source snapshot
`4116a350cd8f6e2805ef` and produced package v7. XML event-by-event comparison
observed exactly the two declared character substitutions and no other
attribute difference. The remaining eight files reuse the admitted v6 files
exactly through same-filesystem hard links.

- Package root:
  `/home/zhengliang01/scheduleurm_work/CFCMT_DATA/eth_five_city_584669/boston_complete_published_package_v7`
- Package manifest:
  `cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/package_v7/package_manifest.json`
- Manifest SHA-256:
  `b39d2676ae72f57f7494d908640ac697c96397b09c587dafa96e4c038e0ef2b9`
- Static decision: pass

## Sequential gates

1. Complete-demand duarouter admission for all 3,806,510 trips.
2. A 5,400 s libsumo trigger window from 7,201 to 12,601 s, crossing both the
   former zipper failure and the v6 collision time.
3. Complete-day microscopic admission with exact demand conservation, zero
   collisions, zero teleports and all 4,203 traffic lights.
4. Controller evaluation only after all preceding gates pass.

Short trigger execution is an operational diagnostic. It cannot authorize a
scientific Boston result by itself.
