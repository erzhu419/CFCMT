# TSC V118r Boston Full Admission V4 Outcome

## Decision

Reject package v6 for controller evaluation. The complete-demand microscopic
admission stopped at simulation time 12,279 s after one collision on lane
`-426602284_0`. An automatic retry with the same deterministic package was
cancelled because it could not provide independent evidence.

## Frozen evidence

- Scheduler task: `t88337`
- Cancelled deterministic retry: `t88390`
- Package manifest SHA-256:
  `7dae903cf5f3b77263140ac4509d4d6bc9c22c1a2394def35b9a6e21b729636a`
- Result:
  `cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/full_admission_v4/result.json`
- Result SHA-256:
  `de4f558dd3ad8a6a9b762e6cafe6e60bc0d3def7184b68e09df35b67806d7d32`
- Elapsed execution: 4,646.66 s
- Vehicles departed/arrived before termination: 28,398/25,266
- Teleports: 0
- Controllable traffic lights: 4,203
- Collision: vehicle `56953` with vehicle `32599`, lane
  `-426602284_0`, position 4.355 m

## Failure localization

Edge `-426602284` is a 15.85 m single-lane receiving edge at junction
`61351006`. Three movements enter that lane:

1. An uncontrolled major straight movement from `-426602280#0`.
2. A signalized left turn from `426602275#1`, TLS `joinedS_726`, link 12.
3. A signalized right turn from `8641349#2`, TLS `joinedS_726`, link 14.

The source network grants uppercase priority green to link 12 in phase 10 and
to link 14 in phase 12 while the straight movement remains uncontrolled. This
creates a reachable priority conflict at the observed three-to-one merge. It
is a network-conversion admission defect, not evidence about CFCMT efficacy.

## Authorized successor

Package v7 may lower only the two declared TLS characters from priority green
`G` to yielding green `g`: `(phase 10, link 12)` and `(phase 12, link 14)` of
`joinedS_726`. It must preserve every other network XML attribute and all
other package files. Route, targeted trigger-window and complete microscopic
admission must pass again before any Boston controller evaluation.
