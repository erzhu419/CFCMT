# TSC v103: Paderborn independent unseen-city safety protocol

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This protocol is frozen before any Paderborn SUMO 1.22 safety or controller
outcome is inspected. Paderborn has not been used to develop or select the
v93/v98 causal source components, strict target-only anchor, B100 offline
selector, pressure guard, controller parameters, or efficacy thresholds.

The source README reports nine teleports under its SUMO 1.6 full-day run. This
known source result is recorded before admission and is not hidden. The v103
test asks whether the same complete demand can finish under the project's
stricter no-teleport execution contract; disabling the timeout alone is not a
pass because every loaded vehicle must eventually depart and complete.

## Frozen source and scope

- Dataset: *Paderborn Traffic Scenario*, release v0.1, Zenodo record
  `4522059`, version DOI `10.5281/zenodo.4522059`, concept DOI
  `10.5281/zenodo.4522058`.
- Target unit: all 11 files published in the fixed Zenodo record, with each
  repository size and MD5 checked before packaging.
- Spatial scope: the complete administrative region of Paderborn represented
  by `network.net.xml`.
- Demand: all precomputed DUA routes in `allroutes.rou.xml`, derived from the
  complete `alltrips.trp.xml`; no route, vehicle, departure interval, edge,
  signal, or vehicle type may be sampled, scaled, rewritten, or removed.
- Traffic lights: the complete published `trafficlightlogic.tll.xml` in
  addition to all programs embedded in the network.
- Source-reported full-day statistics: 203,387 trips, 89,015 simulated seconds,
  approximately 42 minutes runtime on an i7-7700K, and nine teleports under the
  authors' SUMO 1.6 configuration.

Zenodo metadata labels the license GPL-2.0 while the README says GPLv3. The
published `LICENSE.md` is acquired unchanged and governs reuse; the manuscript
must report the metadata discrepancy rather than silently choosing one label.

## Execution contract

The network, DUA routes, traffic-light logic, and vehicle types are preserved.
The headless admission configuration may omit passive polygons and output
destinations, change the seed, and strengthen validation/safety options. It
sets `ignore-route-errors=false`, enables junction collision checks, uses
`collision.action=warn`, disables time-based teleportation, and forbids demand
scaling. The one-second microscopic step and published actuated signal plans
remain unchanged.

Admission uses libsumo 1.22. It records loaded, departed, arrived, running,
pending, insertion-delayed, unfinished, collision, teleport, and controllable
traffic-light counts. The run continues until no vehicle is expected, subject
to a fixed cap of six hours after the last published departure. Reaching the
cap with any pending or running vehicle is a failure, even if SUMO reports zero
teleports because teleportation was disabled.

## Sequential gate

1. Acquire all 11 fixed-record files directly to shared cluster storage and
   verify every published size and MD5.
2. Statically require exactly 203,387 unique precomputed route vehicles,
   complete route-edge references, a full-day departure span, no route or
   vehicle loss relative to the published trip set, and at least one
   controllable traffic light.
3. Freeze the exact SUMO 1.22 configuration and input audit before simulation.
4. Run the complete native actuated-signal condition with development seed
   `5057`. Require exact demand loading, all vehicles completed by the fixed
   cap, zero unique junction-collision incidents, and zero teleport events.
5. Any source identity, parser, demand, route, signal, collision, teleport,
   insertion, completion, or conservation failure rejects Paderborn before
   method evaluation. No route, signal, vehicle parameter, simulator threshold,
   or gate is revised after this first result.
6. Only after safety admission may Paderborn enter the unchanged v98 ladder:
   zero-shot, B100 target-offline source selection, strict target-only CFCMT,
   selected causal mixture, dense/H2O+-style residual, simulator-only MPC,
   fixed time, and pressure-rule baselines.
7. Source selection uses seed-blocked target-offline folds only. Confirmation
   uses disjoint seeds; target-only fallback is an accepted negative-transfer
   decision, not a failed method.

No Paderborn result is yet available under this protocol.
