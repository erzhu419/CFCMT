# TSC V118r Boston Signalized Ramp Merge Yield Repair Protocol

## Failure evidence

Boston package v8 passed static and complete-demand route admission, but its
5,400 s libsumo trigger task `t88714` stopped at simulation time 11,304 s after
one collision on lane `618556460#8_0`, 2.30 m downstream of junction
`cluster_73110325_73135282`. No teleport occurred before termination.

Structured inspection of the server-side network found three movements into
edge `618556460#8`:

- two mainline lanes from `618556460#7`, both `uncontrolled="1"` and
  `state="M"`;
- one signalized ramp from `9504239#0` into the same receiving lane 0, controlled
  by `joinedS_777`, link index 8.

Phase 2 of `joinedS_777` assigned link 8 protected green `G`, even though the
two mainline movements remain uncontrolled. This is the localized priority
conflict consistent with the observed downstream merge collision.

## Declared transformation

For TLS program `joinedS_777` only:

| Phase index | Link index | Before | After |
|---:|---:|:---:|:---:|
| 2 | 8 | `G` | `g` |

The ramp remains green but must yield to the uncontrolled mainline. Phase
duration, every other signal character, all connections, lanes, edges,
junctions, demand and configuration files remain unchanged.

## Static admission

Immutable snapshot `93367d6e9149eb55b330` produced package v9 from the admitted
v8 package. Event-by-event XML comparison observed exactly one declared
character substitution. The remaining eight package files reuse the v8 files
through same-filesystem hard links.

- Package root:
  `/home/zhengliang01/scheduleurm_work/CFCMT_DATA/eth_five_city_584669/boston_complete_published_package_v9`
- Local manifest:
  `cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/package_v9/package_manifest.json`
- Manifest SHA-256:
  `d2b3918188a14ad68552d7f10204076d9f7cf1a3cd27c444ea745c2eade90a3f`
- Static decision: pass

## Sequential gates

1. Complete-demand duarouter admission for all 3,806,510 trips.
2. A 5,400 s libsumo trigger window from 7,201 to 12,601 s, crossing failure
   times 9,518, 11,304, 11,504 and 12,279 s.
3. Complete-day microscopic admission with exact demand conservation, zero
   collisions, zero teleports and all 4,203 traffic lights.
4. Controller evaluation only after all preceding gates pass.

The trigger window remains an operational diagnostic and cannot replace the
complete microscopic admission.
