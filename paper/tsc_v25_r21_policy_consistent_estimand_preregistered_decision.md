# TSC v25/r21 Policy-Consistent Mechanism Estimand Preregistered Decision

Date frozen: 2026-08-09

## Evidence available before freezing

The admissible prior evidence is limited to the failed v23/r19 physical-residual
screen, the completed v24/r20 component attribution, unit tests, and a single
90 s Ingolstadt1 engineering smoke test. No six-city v25/r21 family result and
no CFCMT result on either Salt Lake network has been observed.

The v24/r20 result showed that mobility-only reduced six-city macro action
regret by 5.76%, while every terminal queue, red-accumulation, or spillback
variant produced a large Atlanta regression. Code inspection established a
temporal mismatch: the focal action was applied for 10 s, pressure control was
used for the remaining 50 s, but local mechanism labels and analytic priors
were both treated as if they represented the same 60 s focal-action mechanism.

## Frozen estimand repair

Counterfactual cache version:
`v19-policy-consistent-one-step-mechanisms-two-pass-20260809`.

Each branch keeps the existing paired SUMO snapshot, action set, safety censor,
and 60 s global rollout outcome. It now records two explicitly separate
estimands:

1. `one_step_*`: focal local queue, served queue, red queue, downstream
   occupancy, and speed immediately after the first 10 s candidate action.
   Their uncalibrated analytic priors are also evaluated at 10 s.
2. `interval_cost` and `terminal_system_load`: global vehicles per controlled
   lane under the candidate for 10 s followed by phase-pressure control through
   60 s. These remain the action-ranking outcome and terminal safety signal.

Legacy terminal `next_*` fields are retained only for exact historical
diagnostics. The v25/r21 physical families are forbidden from reading them.
Target-city action labels, target residuals, and target family selection remain
absent at budget zero.

## Frozen production artifacts before Stage A

The six Stage-A jobs were submitted only after the following production
artifacts had passed a physical read audit and their cache files had been made
read-only:

- immutable algorithm snapshot SHA-256:
  `8f60f48e0aac184941bf76bec91e04836ea73ea263b389ca1eaa42366a2d70be`;
- source-tree SHA-256:
  `997f072412977fb40420c3ecceeba48016b098142e71ded1bb32d265df625ffa`;
- 768-file counterfactual-cache aggregate SHA-256:
  `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`;
- 992-file source-rule-cache aggregate SHA-256:
  `5d5ce0f8a609216b6525009feb5076d195912818075bcc33417cb160a0018ca5`;
- deterministic audited source-rule baseline SHA-256:
  `25e6291ba9ff7348642baf09498c9c7bea36ef3fd77a7ee6680b289a03201acb`.

The source-rule baseline contains 16 scenarios and 31 generalized pressure
rules per scenario. Its 496 mean costs are exactly equal, value for value, to
the independently generated r22 baseline, while its provenance points to the
current source tree and current 992-file cache. The initial hashes observed
before multiprocessing children had stopped writing are invalid and are not
admissible experiment identities.

## Stage A: six-city development screen

The development targets are fixed to one representative network per existing
city group:

- RESCO grid4x4;
- Cologne1;
- Ingolstadt1;
- Atlanta 1x5;
- Hangzhou 4x4;
- Manhattan 28x7.

All fits use complete city-group holdout and target group budget zero. The
reference is the unchanged `causal_rigid_advantage` model. The exact
machine-readable candidate and tie-break order, frozen in
`cf_h2o.eval.traffic_signal_stage_a_selection.CANDIDATES`, is:

- one-step queue only;
- one-step red-accumulation only;
- one-step spillback only;
- one-step served-movement only;
- one-step mobility only;
- one-step mobility plus queue;
- one-step mobility plus red accumulation;
- one-step mobility plus spillback;
- one-step mobility plus served movement;
- the full five one-step mechanisms.

A candidate passes Stage A only if all conditions hold:

- six-city macro normalized action regret improves by at least 10% relative to
  rigid;
- at least four of six cities improve beyond numerical tolerance;
- no city's absolute normalized-regret increase exceeds 0.05.

If multiple candidates pass, form a tie set containing every passing candidate
within 0.005 absolute macro regret of the lowest passing macro regret. Select
the member with fewer mechanisms, then the earliest member in the exact order
above. If no candidate passes, v25/r21 fails and no family is promoted. The
best failing family may be reported only as a diagnostic.

## Stage B: unopened Salt Lake confirmation

The selected Stage-A family and every hyperparameter are frozen before opening
controller results on:

- `saltlake_400s_200w_q1_weekday_peak`;
- `saltlake_state_university_q1_weekday_peak`.

Both networks form the single held-out `salt_lake_city` domain. Their demand
construction, route hashes, and safety admission were fixed independently of
CFCMT outcomes. Salt Lake contributes no source fit, source-family selection,
target adaptation group, or residual label in the zero-shot confirmation.

The frozen family passes Stage B only if mean normalized action regret across
the two Salt Lake networks improves by at least 10% versus rigid, both networks
improve beyond numerical tolerance, and neither network regresses by more than
0.05. Failure is reported as a confirmatory failure; the Salt Lake result may
not be used to choose another family.

## Stage C: breadth, adaptation, and closed loop

Only after Stages A and B pass:

- evaluate all 18 networks under complete leave-one-city-group-out folds;
- run target adaptation budgets 0, 8, 16, 32, 60, and 120 complete action
  groups with disjoint model-adaptation and policy-selection groups;
- run paired 600 s closed-loop evaluation seeds 131, 337, and 911;
- run the frozen 3,600 s extension without changing family, guard, action set,
  reward, or metric;
- report normalized action regret, optimal-action rate, system vehicle-hours,
  queue per controlled lane, throughput/completion, collisions, teleports,
  uncertainty coverage, city wins, worst-city harm, and bootstrap intervals.

Offline action regret is a mechanism-development endpoint. Publication-level
control claims require the paired closed-loop stages. A passing offline screen
cannot substitute for closed-loop safety or efficacy.
