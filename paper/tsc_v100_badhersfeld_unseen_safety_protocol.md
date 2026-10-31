# Bad Hersfeld independent unseen-target safety protocol

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This protocol is frozen before any Bad Hersfeld SUMO safety or controller
outcome is inspected. It governs source acquisition, full-demand packaging,
and simulator-integrity admission only. No CFCMT, H2O+, source selection,
target adaptation, reward, queue, delay, or throughput result from Bad
Hersfeld is available at this point.

## Frozen source and scope

- Repository: `DLR-TS/sumo-scenarios` at commit
  `00f6eb479a9dc0fbaeb731495c11d48d7a9661d3`, EPL-2.0.
- Source unit: the complete `BadHersfeld/` directory, 47 tracked files and
  28,258,195 bytes at the fixed commit.
- Network: the published `osm_edited.net.xml.gz`, without topology or
  traffic-light rebuilding.
- Runnable road-demand conditions: the complete `present` and
  `future_no_prt` 86,400 s scenarios. Both published full-day activity-based
  demand files are retained.
- Additional traffic: every route/additional input named by each configuration,
  including the complete GTFS public-transport routes and stops, truck demand,
  vehicle types, parking areas, rerouters, obstacle definition, and calibration
  input. Passive polygon and output destinations may be omitted from the
  canonical admission configuration, but no network, route, vehicle, person,
  stop, or time interval may be sampled or dropped.

The PRT configurations are not execution conditions because their joined PRT
network and generated PRT demand are build products not present as complete
tracked runnable inputs at the fixed commit. Their source files remain in the
complete acquisition. This exclusion is based on source availability before
any simulation result.

## Execution contract

All source network and demand inputs are identity-checked before simulation.
The package may make path-only configuration changes and remove passive output
destinations. Compressed XML may be read directly or decompressed losslessly;
neither operation may change XML traffic semantics.

Admission uses libsumo 1.22 and the same cross-network execution contract as
the existing targets: junction collision checking remains enabled, collision
events are reported rather than teleported away, and time-to-teleport is
disabled. Thus the source configuration's permissive collision/teleport
recovery settings cannot hide a safety failure.

## Sequential gate

1. Before simulation, inventory every configured network, route, vehicle,
   person, public-transport, stop, and traffic-light input and freeze the exact
   counts and package identities.
2. Run the complete `present` scenario with development seed 5057 for 86,400 s.
3. Require exact deterministic demand loading, every declared controllable
   traffic light, the full horizon, zero unique collision incidents, and zero
   teleports.
4. Any parser failure, demand loss, missing signal, collision, teleport, or
   premature termination rejects the city before `future_no_prt`, additional
   seeds, or controller efficacy.
5. Only if the candidate passes, admit both full-day conditions under the
   remaining development seeds `6067, 7103, 8171, 9277, 12001, 13007, 14009`
   and disjoint confirmation seeds `15101, 16001, 17011, 18013, 19001, 20011,
   21013, 22003`. Every condition-seed run must pass.

No threshold, source condition, signal plan, route, demand record, or vehicle
parameter is revised after the first safety outcome.

## Pre-simulation input-semantic amendment

This amendment was recorded after static package validation and before any
Bad Hersfeld SUMO run or controller outcome. The published truck-demand file
contains exactly 29 `lkw_*` vehicles whose three-edge route skeletons encode an
origin, the fixed loading edge `-902659463`, and the reverse origin edge. These
29 skeletons are intentionally disconnected and rely on the published
`ignore-route-errors=true`, `device.rerouting.probability=1`, 300 s rerouting
period, and 300 s pre-period. All other 18,675 audited routes are connected and
all route edges exist.

The package therefore preserves this source-declared dynamic-routing behavior
and reports the 29 skeletons separately rather than silently treating them as
connected routes. No route is edited or removed. Admission still requires SUMO
to load the exact full road-demand inventory; any silently dropped truck,
parser/runtime failure, collision, teleport, or demand-conservation mismatch
rejects the candidate. This amendment changes no controller, safety threshold,
seed, horizon, demand record, or post-outcome decision rule.
