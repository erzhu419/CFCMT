# Luxembourg LuST unseen-target safety protocol

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This protocol precedes all Luxembourg controller outcomes. It governs immutable
source selection, full-demand packaging, and simulator-safety admission. No
CFCMT, H2O+, source-selection, target-adaptation, reward, queue, delay, or
throughput result from Luxembourg has been computed or inspected.

## Frozen source and variant

- Repository: `lcodeca/LuSTScenario` at commit
  `5edb7ecb9ad196c39172b6eb95d19ed789f4b1a6` (MIT license).
- Scenario variant: `due.static.sumocfg`, the dynamic-user-equilibrium demand
  with the source static traffic-light programs.
- Coverage: all five demand files named by that configuration: three complete
  local-route shards, all bus-line vehicles, and all transit vehicles.
- Demand: 288,250 vehicles with 288,250 embedded routes, departures from 0 to
  86,395 s. No route, vehicle, line, shard, or time interval is sampled or
  omitted.
- Network: 5,779 external edges and 201 traffic-light programs. The native
  network, demand files, vehicle types, bus stops, and static TLS file are
  preserved byte for byte. Passive detector/output and polygon files are not
  loaded.
- Package protocol: `cfcmt-lust-native-sumo-due-static-package-v1`.

The source authors state that LuST was generated and validated with SUMO 0.26
and that mobility under newer SUMO versions must not be described as retaining
that validation. Accordingly, the experiment is an unseen-network stress test
under SUMO 1.22, not a claim that its modern-version trajectories remain
real-world calibrated.

## Candidate gate

Run the complete 86,400 s scenario with libsumo, the fixed SUMO 1.22 execution
protocol, and development seed `5057`. Admission requires:

- all 288,250 vehicles loaded consistently;
- all 201 traffic-light programs exposed;
- the full horizon reached;
- zero unique collision incidents; and
- zero teleports.

Any parser incompatibility, demand loss, collision, teleport, missing TLS, or
premature termination rejects LuST before controller efficacy. The source files,
network, static TLS programs, full-day horizon, and zero-event thresholds will
not be repaired or relaxed after this run.

## Multi-seed gate after candidate admission

If and only if seed `5057` passes, rerun the unchanged full-day package with the
remaining development seeds `6067, 7103, 8171, 9277, 12001, 13007, 14009`, then
with disjoint confirmation seeds `15101, 16001, 17011, 18013, 19001, 20011,
21013, 22003`. Every run must pass. Only then may Luxembourg enter target-offline
construction and controller efficacy evaluation.
