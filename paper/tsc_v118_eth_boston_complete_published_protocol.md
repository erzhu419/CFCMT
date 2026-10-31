# TSC v118: ETH Boston independent complete-published-demand protocol

## Status and purpose

This protocol is frozen before inspecting the Boston network inventory,
microscopic safety outcome, controller outcome, CFCMT source choice, or target
adaptation outcome. The only Boston-specific facts examined before this freeze
are that `BOS` is one of the five units in the fixed ETH archive and the v102
streaming screen reported departures from 7,201.78 to 50,395.74 seconds.

V118 is an **independent complete-published-demand external stress test**. It is
not a 24-hour experiment: the published Boston departure span is approximately
12 hours. The experiment therefore cannot replace a full-day external-city
confirmation and must always be labelled as complete published demand rather
than full day.

## Frozen source and target unit

- Dataset: *Traffic Simulations for Boston, Lisbon, Los Angeles, Rio de
  Janeiro, San Francisco*, DOI `10.3929/ethz-b-000584669`.
- Archive: `sumo_sim_perc_mfd.zip`, 658,810,696 bytes, MD5
  `802321f1d2c61de527ed579ce9c54842`, CC BY-NC 4.0.
- Target unit: every network, trip, TAZ, additional, and configuration entry
  under `BOS/` in that archive.
- Demand scope: every published Boston trip from the earliest to the latest
  departure, followed by a fixed six-hour drain allowance. No trip, route,
  time window, edge, or vehicle may be sampled, scaled, repaired, or removed.
- Runtime: SUMO 1.22.0 through `libsumo`; TraCI is not permitted.

The microscopic instantiation may remove passive output declarations only. It
must set `ignore-route-errors=false`, `time-to-teleport=-1`,
`max-depart-delay=-1`, `collision.action=warn`, and
`collision.check-junctions=true`. Source demand and network semantics otherwise
remain unchanged.

## Sequential admission

The following stages are fail-closed and must be completed in order.

1. **Static package audit.** Verify the fixed archive identity, exact complete
   `BOS/` entry set, unique and complete trip identities, valid edge and TAZ
   anchors, monotonic departures, the previously screened departure bounds,
   at least one controllable traffic light, and removal of passive outputs
   only.
2. **Microscopic safety admission.** Run the complete published demand with
   seed `5057`. Require exact trip loading, departure, arrival, and final
   conservation; zero pending vehicles; zero collision incidents; zero
   starting or ending teleports; and no runtime exception. Any irreversible
   collision or teleport failure stops the run immediately and rejects Boston.
3. **Passive target construction.** Only after admission passes, collect
   phase-pressure trajectories on adaptation seeds `5057` and `6067`. The
   target-label-free arms may use static topology, complete demand metadata,
   and contemporaneous state distributions but no target next-state labels.
   B100 arms receive exactly 100 deterministic target transition groups.
4. **External confirmation.** Only after the V115 fits and V116 selector are
   frozen, run all applicable arms on the 32 fresh seeds in the machine-readable
   protocol. Failed seeds may not be excluded.

## Control scope and information parity

The entire Boston network and all published trips remain active in every run.
To preserve the fixed four-intersection model interface, the controlled unit is
the four traffic lights with greatest mean incoming halted-queue exposure in
the two passive adaptation trajectories, breaking ties by traffic-light ID.
This rule is fixed before those trajectories are observed and is shared by all
arms. All other lights execute their published programs.

The confirmation arms are PhasePressure, architecture-matched target-only
B100, source-selected CFCMT B100 and B0, and same-estimand H2O+ dense B100 and
B0. The source selector and all source profiles are frozen before Boston is
opened. A rejected or absent source profile falls back to PhasePressure; it is
not replaced after observing Boston efficacy.

## Primary estimand and gate

The paired unit is the same complete Boston demand and fresh seed. The primary
metric is full-network mean trip waiting time. The primary CFCMT B100 arm must:

- have a paired-bootstrap 95% upper relative-delta bound no greater than zero
  versus H2O+ B100 and architecture-matched target-only B100;
- have a paired-bootstrap 95% upper relative-delta bound no greater than
  `+1%` versus PhasePressure;
- activate a source profile on at least half of the confirmation seeds; and
- add no aggregate collision or teleport incident relative to PhasePressure.

All applicable primary conditions must pass jointly. B0 and budget-improvement
comparisons are secondary. A static or microscopic rejection is itself a valid
V118 outcome and authorizes no controller claim for Boston.
