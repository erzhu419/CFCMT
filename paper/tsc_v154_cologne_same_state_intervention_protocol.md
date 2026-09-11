# V154 Cologne Same-State Service Intervention

This is a two-branch physical mechanism diagnostic at the recorded Cologne
`cologne1`, seed `41242` state at 26700 seconds. It is not a policy efficacy
experiment. The accepted uninterrupted replay and its server-side SUMO state
and executor context are the sole inputs.

Both branches reload the same state and RNG using the original SUMO 1.22.0
configuration, seed, `--thread-rngs 1`, and `--save-state.rng true`. Original TLS
candidates and executor timings are constructed before the reload. Each reload
is followed by executor `restore(..., write_state=True)` so its current signal,
green elapsed time and pending clearance match the saved controller state.

The control is one request to retain `rrrGGrrrrrrrrGGrrrrr`, followed by 60
seconds of holding. The intervention is one request, through the same original
safe executor, for existing phase `GGGggrrrrrGGGggrrrrr`, which permits the
straight movements 2 and 12, followed by 60 seconds of holding. The executor
ticks after each original one-second SUMO step. No model is trained or invoked;
there is no action reselection and no change to demand, routes, car-following,
lane-changing, collision handling or signal-clearance parameters.

## Reproduction prerequisite

The hold branch runs first and must reproduce the uninterrupted replay's seven
samples at 26700, 26710, ..., 26760. Comparison covers active and pending counts,
executor state, all incoming/receiving lane vehicle counts, halting, occupancy,
speed, movement counts, and head-vehicle identity, position, speed, waiting,
next-TLS movement/distance/signal, and next route edge.

The original state writer uses `--save-state.precision 8`. Accordingly, only
head positions and next-TLS distances permit an absolute difference up to
`0.5e-8` metres, the half-unit rounding bound of that serialization. All other
compared fields require exact equality. The result reports exact equality
separately, retains every nonzero numeric difference, and reports whether each
falls within that fixed bound. The bound is frozen before loading the saved
state; it is not expanded after observing a mismatch. A failed prerequisite
prevents the switch branch from running.
The restored hold must also complete all 60 seconds without collision or
teleport events, matching the incident-free uninterrupted rigid segment.

## Outcomes

Every one-second step records observed vehicle transitions from any incoming
lane into a TLS internal lane or a directly connected receiving lane. Changes
between incoming lanes and disappearance without an observed crossing are not
counted as intersection entries. The two original shared-lane head vehicles
are tracked individually until entering an internal lane, reaching their
expected receiving edge, or arriving. Per-second active/pending counts and
newly departed/arrived totals accompany the crossing events. Collision-event
counts and bounded incident samples are retained by the existing safety ledger;
teleportation terminates the branch as an invalid fixed-population run. A
completed intervention with collisions is explicitly marked as not incident-free.

The key interpretation is whether the original straight-moving head vehicles
remain trapped under the left-turn hold but cross after a safe straight-green
request while receiving space is available. This would support shared-lane
movement blocking in this state. It would not establish the performance of
repeated interventions or a new learned policy.

Occupancy values remain exactly those returned by SUMO, expressed as fractions.
Replay-v1's `occupancy_pct` name is a diagnostic labeling error; this module
reads its unchanged values as fractions and outputs `occupancy_fraction`.

The module is `cf_h2o.eval.traffic_signal_cologne_service_counterfactual`, with
arguments `--replay-result` and `--out`. It uses the remote scratch paths already
recorded by the replay and downloads neither the state nor a model. Only a small
result JSON is returned. At freezing, neither branch has been run.

## Result: reproduction prerequisite rejected

Task `t90749` completed with `REPRODUCTION_FAILED` and exit code 2. The hold
branch ran for all 60 seconds without collisions, teleports, intersection
entries, departures or arrivals. The switch branch was not run; its result is
`null`. This is a failed restoration prerequisite, not a negative result for
the proposed straight-green intervention.

The sole comparison failure was the pending-insertion getter immediately after
loading the state: at 26700, the uninterrupted run reported 526 pending vehicles
and the restored run reported zero. Active inserted vehicles were 199 in both.
After the first simulation step, the restored pending count was already 528.
All six subsequent ten-second pending observations matched the uninterrupted
run exactly:

| Time | Uninterrupted pending | Restored pending |
|---:|---:|---:|
| 26700 | 526 | 0 |
| 26710 | 540 | 540 |
| 26720 | 547 | 547 |
| 26730 | 552 | 552 |
| 26740 | 560 | 560 |
| 26750 | 571 | 571 |
| 26760 | 578 | 578 |

All other compared categorical/count/state fields matched. The 112 nonzero
position or next-TLS distance differences had maximum absolute value
`4.093166694474348e-9` metres, within the frozen `5e-9` serialization bound.
Both original shared-lane head vehicles remained in their incoming lanes for
the full hold branch, and the active vehicle count stayed at 199.

The immediate zero followed by restored pending counts is consistent with a
pending-insertion getter or state-initialization difference at reload, rather
than disappearance of the waiting demand. That interpretation comes from the
subsequent observations; it is not a source-level adjudication of SUMO's loader.
The frozen prerequisite nevertheless required the initial pending count to
match, so the rejection is retained. No observation is dropped, no warmup step
is inserted and no threshold is relaxed to run the switch branch.

Evidence:
`cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/same_state_intervention_v1/result.json`
(58,644 bytes). No state snapshot or model was retrieved.
