# TSC V118r Boston V11 Full Admission Outcome

## Decision

Reject Boston package v11 for microscopic controller evaluation. Scheduler task
`t89321` executed the intended complete-demand libsumo admission and stopped at
simulation time 13,146 s after one collision. This is a scientific admission
failure, not a runtime exception or scheduler failure. It does not authorize
V149 or any Boston controller evaluation.

## Bound evidence

- Scheduler task: `t89321`
- Immutable source snapshot: `4187c3ab21a61280e249`
- Package protocol: `cfcmt-eth-boston-complete-published-package-v11`
- Package manifest SHA-256:
  `e46d49518dac1596c93fba04839f5a4508fc6296ee2d8bbb557e6baf37f7604e`
- Local terminal result:
  `cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/full_admission_v8/result_t89321.json`
- Backend and runtime: libsumo/SUMO 1.22.0 on `node006`
- Requested scope: BOS, 3,806,510 unique trips, 4,203 controllable traffic
  lights, begin 7,201 s, stop 71,996 s, no shortened operational horizon
- Execution before gate termination: 3,759.18 s wall time; 38,081 vehicles
  departed, 33,850 arrived, 4,231 remained active, 14 remained pending and
  12,977 vehicles were still expected
- Teleports: 0
- Unique collisions: 1

Package identity, city identity, SUMO version, traffic-light count and full-day
request all match the predeclared contract. The evaluator deliberately stopped
when the zero-collision gate became irreversibly false. Consequently, full-day
completion and exact loaded/departed/arrived demand gates are false and cannot
be interpreted as separate data defects.

## Failure localization

The collision occurred on receiving lane `8647416#3_0`, 2.63 m downstream of
junction `61400161`, between vehicles `77188` and `95307`. Two movements enter
that same lane:

1. Vehicle `77188` arrived from `8647416#1` through an uncontrolled connection
   with state `M`.
2. Vehicle `95307` arrived from `128013681#0` through `joinedS_514`, link index
   0. At the incident, phase 6 used state `GrrrrrrrGGGGsrrr`, so link 0 had
   priority green `G`.

This is another reachable protected-versus-uncontrolled shared-receiving-lane
conflict in the converted network. It is a package-admission defect, not
evidence about CFCMT performance.

## Successor boundary

Scheduler automatically created retry `t89359` with the same package, seed and
command. That retry is not independent scientific evidence and cannot reverse
the package-v11 rejection. A successor package may be declared only after
structured network inspection confirms a narrowly scoped repair for
`joinedS_514`; static, complete-route, trigger-window and complete-day gates
must then run again in order. V149 remains closed.
