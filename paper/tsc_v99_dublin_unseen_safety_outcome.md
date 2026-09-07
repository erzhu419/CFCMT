# Dublin Urban unseen-target safety outcome

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This is a simulator-integrity result only. No CFCMT, H2O+, target adaptation,
source selection, controller reward, queue, delay, or throughput result from
Dublin was computed or inspected.

## Frozen source and corrected package

- Source: `maxime-gueriau/ITSC2020_CAV_impact` at commit
  `a2fe1d2f67f546099b4b44dc81ae7ba82a60e054`.
- Coverage: the full 86,400 s Urban demand, all 418,078 explicit vehicles, all
  38,573 source route definitions, all 435 controlled intersections, and all
  six published vehicle-composition variants A--F.
- Correct package protocol: `cfcmt-dublin-urban-native-sumo-package-v2`.
- Package manifest SHA-256:
  `cbc1fb7366a31b0ed816ce621f5c3622ea68e01c0003b1194d4430e4a2661241`.
- Package tree SHA-256:
  `7be9c3032581037dd7366ad4e99c2fee7dff0e02d52939da3a9d006a628208b6`.

The first v1 package is invalid evidence. It moved the route-definition file
from the source `additional-files` sequence into `route-files`, so SUMO loaded
an emitter route distribution before its referenced route. The simulation
stopped at time zero. Version 2 preserves the source order
`TLS -> vehicle types -> route definitions -> emitters`, omitting only passive
detector/POI output definitions. A SUMO 1.22 parser run then loaded the scenario
successfully.

## Fixed gate

Scenario A and seed 5057 were the predeclared first candidate. Admission used
libsumo 1.22 for the complete 86,400 s horizon and required exact demand
loading, at least 400 controllable traffic lights, zero collision incidents,
zero teleports, and completion of the full horizon. Failure terminates Dublin
before variants B--F or any controller efficacy run.

## Result

The valid v2 simulation exposed all 435 traffic lights and loaded all 418,078
vehicles. At 139.5 s, after 430 departures and 76 arrivals, SUMO reported a
junction collision between vehicles `767_2.0` and `771_4.0` on internal lanes
at junction `1431043259`. No teleport or runtime exception occurred. The run
terminated immediately because the zero-collision limit had become
irreversibly false.

SUMO's step API reported zero newly loaded definitions because the route file
was eagerly parsed at startup, whereas the summary output correctly reported
418,078 loaded vehicles. The admission implementation was subsequently fixed
to accept either startup or stepwise loading while retaining exact demand
conservation. This bookkeeping correction cannot change the observed
collision or the rejection decision.

## Decision

Dublin is rejected as the independent closed-loop confirmation city. Variants
B--F are not run, the collision threshold is not relaxed, and the native
network, traffic-light programs, routes, demand, and vehicle parameters are not
modified after observing the result. The compact evidence is retained at
`cf_h2o/results/cluster/tsc_v99_dublin_unseen_20260831/admission_v2_a_seed5057/result.json`.
