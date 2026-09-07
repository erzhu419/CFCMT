# Bad Hersfeld independent unseen-target safety outcome

Date evaluated: 2026-08-31 (Asia/Shanghai)

## Frozen identities

- Source: `DLR-TS/sumo-scenarios` commit
  `00f6eb479a9dc0fbaeb731495c11d48d7a9661d3`.
- Acquisition: 47 files, 28,258,195 bytes, with every file checked against
  its fixed Git blob identity.
- Package protocol: `cfcmt-dlr-badhersfeld-native-sumo-package-v1`.
- Package manifest SHA-256:
  `1d5738ee6a7f7f1eac0c16ba0e829e5181c02b5743aeff4580712f8bca2b53b4`.
- Package tree SHA-256:
  `f9482c593aed8aed0912bc11e281b5025d9edcdf9ba5363d7996a51eddc064fe`.
- Runtime source-tree SHA-256:
  `27e2acd76b202d7f47dcd8bdf24bea7ca66f65b37c1c77ce9068ded2a78bb34c`.
- Runtime: libsumo 1.22.0 on `node002`.

## Pre-simulation validation

Both full-day packages passed XML, input-order, source-identity, and zero-step
SUMO loading checks. The present condition contains 18,582 road vehicles,
including 17,899 person-triggered vehicles, and 26,144 person plans. It contains
18,704 audited routes, no missing edge, 18,675 connected routes, and the 29
published LKW dynamic-reroute skeletons recorded in the protocol amendment.
The source network has 21 explicit signal programs and libsumo exposes 35
controllable traffic-light objects.

## Sequential-gate result

The first and only authorized run was `badhersfeld_present`, seed 5057, with a
frozen 86,400 s horizon, zero allowed collision incidents, and zero allowed
teleports. The run was rejected at simulation time 22,186 s after the first
unique junction collision:

- collider: `all_all_7426_tr`;
- victim: `all_all_26750_tr`;
- lane:
  `:cluster_1170610905_2421325896_296320406_3281721216_3281721218_8_0`;
- departed/arrived at termination: 573/16;
- active vehicles: 557;
- starting and ending teleports: 0/0;
- runtime exception: none.

The demand-conservation and horizon checks are false only because the frozen
fail-fast rule terminated the simulation immediately after the irreversible
collision threshold was exceeded. They are not interpreted as an independent
full-horizon demand failure.

## Decision

Bad Hersfeld is rejected as a confirmatory target. The `future_no_prt`
condition, remaining development seeds, confirmation seeds, and all controller
comparisons are not run. No route, vehicle, signal plan, safety threshold, or
simulator option is repaired after observing the result.

This rejection is a target-admission result. It does not alter the previously
confirmed positive Jinan transfer result, and it supplies no evidence about
CFCMT efficacy because no controller was evaluated on Bad Hersfeld.

## Evidence

- Local result:
  `cf_h2o/results/cluster/tsc_v100_badhersfeld_unseen_20260831/first_candidate/results/badhersfeld_present_seed5057/admission.json`.
- Local shard record:
  `cf_h2o/results/cluster/tsc_v100_badhersfeld_unseen_20260831/first_candidate/results/shard_00_summary.json`.
- Frozen package inventory:
  `cf_h2o/results/cluster/tsc_v100_badhersfeld_unseen_20260831/package_v2/frozen_inventory.json`.
