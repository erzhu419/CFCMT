# TSC v102: ETH Lisbon independent unseen-city microscopic protocol

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This protocol is frozen before inspecting any Lisbon microscopic SUMO safety,
controller, CFCMT, source-selection, or target-adaptation outcome. Lisbon has
not been used to develop or select the v93/v98 causal source components,
target-only anchor, offline source selector, pressure guard, or efficacy
thresholds.

The ETH release describes a five-city data product for 24 hours of car traffic.
Its published study used mesoscopic SUMO and analyzed an extended morning
period. The v102 primary candidate is therefore a **new microscopic
instantiation of published Lisbon network and demand**, not a reproduction of
the authors' mesoscopic result. Any accepted claim must retain that distinction.

## Frozen source and target unit

- Dataset: *Traffic Simulations for Boston, Lisbon, Los Angeles, Rio de
  Janeiro, San Francisco*, DOI `10.3929/ethz-b-000584669`.
- Repository bitstream: `sumo_sim_perc_mfd.zip`, complete archive, expected
  size 658,810,696 bytes and repository MD5
  `802321f1d2c61de527ed579ce9c54842`.
- License: Creative Commons Attribution-NonCommercial 4.0 International.
- Target unit: every Lisbon (`LIS`) network, route, additional, TAZ, and
  configuration entry contained in the fixed archive. No route, departure
  interval, edge, junction, traffic signal, or vehicle class may be sampled or
  removed.
- Demand horizon: the complete published 24-hour Lisbon route demand. A drain
  interval may be appended after the last departure, but the evaluated demand
  begins at the earliest published departure and retains every source record.

The other four cities remain in the acquired archive for source provenance but
are not target observations and may not be used to tune the Lisbon method.

## Microscopic instantiation contract

The source Lisbon network, routes, vehicle types, traffic-light programs, and
additional inputs are preserved. A new configuration may change only simulator
mode from mesoscopic to microscopic, execution horizon, random seed, output
destinations, and audited safety options. It may not repair topology, rewrite
routes, scale demand, change signal plans, enable permissive route-error
handling, or discard failed insertions.

Execution uses libsumo 1.22 with junction collision checks enabled,
`collision.action=warn`, and time-to-teleport disabled. Starting and ending
teleport counts, unique physical collision incidents, loaded/departed/arrived
vehicle counts, unfinished vehicles, and controllable traffic-light inventory
are recorded. Disabling teleportation does not make a gridlocked run pass:
every source vehicle must load, the full horizon must complete, and the drain
must leave no unexplained unfinished demand.

## Sequential admission gate

1. Acquire the complete fixed archive directly to shared cluster storage and
   verify its repository size and MD5 before extraction.
2. Inventory every archive entry and every Lisbon input. Require a complete
   24-hour demand, exact route/vehicle identity retention, valid network
   references, and at least one controllable traffic light.
3. Freeze the generated microscopic configuration and its input audit before
   running SUMO.
4. Run one complete development-seed (`5057`) full-demand Lisbon simulation.
   Require exact demand loading, full-horizon completion, zero unique junction
   collision incidents, zero teleports, and no unexplained unfinished vehicle.
5. Any failed source identity, parser, demand, signal, collision, teleport,
   insertion, completion, or conservation gate rejects Lisbon before method
   evaluation. No source input or threshold is changed after the first result.
6. Only after admission may v98 target-only and causal source-selector models
   be instantiated. Selection uses the same B100 target-offline,
   seed-blocked five-fold protocol and thresholds frozen for v98; no Lisbon
   closed-loop evaluation seed is used for selection.
7. Confirmation uses disjoint seeds and reports target-only fallback as a valid
   negative-transfer decision. Efficacy is compared against fixed time,
   max-pressure/phase-pressure, dense residual/H2O+-style transfer,
   simulator-only MPC, strict target-only CFCMT, and the selected causal source
   mixture under one safety protocol.

No Lisbon result is yet available under this protocol.
