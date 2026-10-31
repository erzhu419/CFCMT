# TSC v42/r38 External LA and Nanchang Confirmation Preregistration

## Trigger

The integrity-audited v41 development stage passed its frozen gate:

- v41 audit SHA-256: `5bcae9467cdb4cc157676e94078d204702ebdca6d51007ffca4bc85d13951066`
- nested LOCO relative improvement: `9.8379%`
- improved held-out cities: `6/7`
- maximum absolute held-out-city regression: `0.003881324`

Therefore LA and Nanchang may be opened for conversion and safety admission.
No external CFCMT outcome has been computed at the time this protocol is
frozen.

## External data and conversion

All raw-file hashes and the conversion-script hash are frozen in
`cf_h2o/config/traffic_signal_tsc_v25_external_la_nanchang_confirmation.json`.
No corridor, route, OD pair, time slice, or intersection may be removed.

- LA uses all 30 CityFlow intersections, 52 directed roads, 2,203 vehicle
  records, lane links, and supplied signal phases.
- Nanchang uses all 2,048 intersections, 6,024 directed roads, 859 signalized
  intersections, and 9,786 flow definitions (an estimated 126,669 vehicles).

Conversion and network admission are not efficacy tests. They may inspect raw
topology, routes, demand, collisions, teleports, controllable TLS coverage, and
safe complete action-group capacity. They may not fit CFCMT or inspect external
action-regret outcomes. The canonical conversion runtime is SUMO/netconvert
`1.22.0`, matching the frozen experiment runtime.

## Independent external folds

LA and Nanchang are evaluated as separate 19-network folds: the immutable 18
development networks plus exactly one external target. The two external
targets never source one another. Source labels remain the frozen 18-network
cache; target labels come only from the newly admitted target simulator.

For each external target:

- seeds 5057/6067 provide the B100 adaptation pool;
- five seed-balanced folds train five 80-group model pairs and produce OOF
  target-selection evidence;
- seed 7079 is untouched until the target candidate is frozen;
- seeds 8081/9091 are reserved for closed-loop rollouts.

The v41 selector is immutable:
`scope_all_tau_0_rho_0p1_lambda_0`. No external city, candidate, or threshold
tuning is allowed. The deployed model is the five-model score-level ensemble
with within-model plus between-model uncertainty.

## Offline confirmation gate

The external stage passes only if all conditions hold:

1. macro relative normalized action-regret improvement over the rigid anchor
   is at least `2%`;
2. both external cities strictly improve;
3. maximum absolute external-city regression is no greater than `0.01`;
4. at least one target selects a non-anchor candidate.

Failure stops closed-loop claims and triggers no post-hoc threshold change.

## Closed-loop stage

Only after the offline gate passes, run full 3,600-second libsumo rollouts on
seeds 8081/9091. Compare fixed-time, PhasePressure, MaxPressure, rigid-anchor
MPC, simulator-only MPC, H2O+-style dense-residual MPC, and the frozen CFCMT
ensemble. Waiting time is primary; travel time, queue, throughput, teleports,
and collisions are secondary/safety outcomes.

## Claim boundary

Passing this protocol supports external simulator-network confirmation of
target offline adaptation. It does not convert simulator counterfactuals into
real-world counterfactual evidence and does not establish zero-shot transfer.
