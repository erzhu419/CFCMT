# V154 Cologne Rigid Service Diagnostic

## Retained evidence

The V150L `cologne1`, seed `41242` failure already occurs in rigid target-only.
Its single controlled TLS, `cluster_357187_359543`, spends almost the whole
3,600-second run in green while repeatedly retaining its current phase. This
does not support excessive switching or clearance time as the main explanation.

| Arm | Arrived / due demand | Pending at horizon | Switches | Same-phase requests | Green seconds |
|---|---:|---:|---:|---:|---:|
| PhasePressure | 1,999 / 2,015 | 0 | 252 | 103 / 360 | 2,374 |
| Rigid target-only | 210 / 2,015 | 1,606 | 22 | 338 / 360 | 3,505 |
| Selected source | 128 / 2,015 | 1,652 | 21 | 339 / 360 | 3,509 |
| Matched placebo | 188 / 2,015 | 1,618 | 26 | 334 / 360 | 3,495 |

Rigid's yellow and all-red time total only 95 seconds, including seven seconds
of occupancy-clearance extension. The executor records zero busy or minimum
green rejections. Returning `request_accepted=false` for a same-phase request is
the executor's normal behavior; the corresponding stay remains effective.

The retained rigid trace repeatedly selects `rrrGGrrrrrrrrGGrrrrr`: first for
approximately 100 seconds from time 25460, then for 550 seconds from 25600,
and finally from 26350 through the trace cutoff at 27830. At the final sample,
the executor reports 1,476 seconds elapsed in that green state. From 26700 to
27830, all 114 consecutive ten-second samples have mean incoming-lane speed
zero and 145 vehicles on those lanes. Source reaches its terminal sampled
zero-speed segment earlier, at 26280, with 180 vehicles; placebo reaches its
terminal segment at 26810, with 154 vehicles.

The first rigid zero-speed samples occur at 25780–25800. Later brief recoveries
mean that this first episode must not be described as irreversible gridlock.
The terminal sampled segment is stronger evidence of persistent local loss of
service. The retained record still does not measure stop-line discharge.

The static TLS extraction identifies a specific explanation to test. The held
phase activates links 3, 4, 13 and 14, corresponding to left/U-turn movements
on `-32038056#3_1` and `28198821#3_1`. Each lane also carries a straight movement,
links 2 and 12 respectively, whose signal remains red in this phase. Existing
lane-aggregated action features can therefore count demand in a nominally
green lane even when its front vehicle wants the red straight movement.
The read-only replay below confirms red-bound front vehicles with empty
receiving lanes in the recorded failure.

## Read-only replay result

Task `t90447` completed successfully using source snapshot
`9f7d6123bde1ba41f561` and the original SUMO 1.22 runtime. Every field in all
154 retained original intervention rows before 26820 matched exactly, with zero
mismatches. The replay recorded all 144 specified additional observation times.
Its reported computation time was 3.66 seconds. Only the 865,198-byte result
JSON was retrieved; the saved SUMO state and executor context remain on the
server.

At time 26700 the selected phase was `rrrGGrrrrrrrrGGrrrrr`, whereas the pressure
reference was `GGGggrrrrrGGGggrrrrr`:

| Shared incoming lane | Vehicles / halted | Head movement and signal | Head waiting | Green-movement demand behind the head |
|---|---:|---|---:|---:|
| `-32038056#3_1` | 23 / 23 | Straight, link 2, red | 333 s | 10 |
| `28198821#3_1` | 8 / 8 | Straight, link 12, red | 339 s | 4 |

Both heads were stopped approximately one metre before their stop line. All
eight directly connected outgoing lanes contained zero vehicles. Nevertheless,
the selected phase received `green_q=41.872077` and the same positive
`service_pressure`; its predicted cost contrast was `-0.510726`, below the
pressure reference's zero. The unchanged rigid selector therefore retained the
phase that could not serve either shared-lane head.

This is a repeated observation: of 121 sampled decisions selecting that phase,
102 had both shared-lane heads stopped with red signals, and 101 also had all
eight outgoing lanes empty. From 26700 through 26810, the two head IDs remained
unchanged, all 145 incoming-lane vehicles were stationary, and pending insertion
vehicles rose from 526 to 613. This identifies shared-lane head-of-line blocking
and a mismatch between lane-level green demand and accessible movement service.
It rules out downstream receiving-lane saturation as the immediate explanation
at these observed empty-exit times. It does not yet quantify the effect of a
corrected policy over a full episode.

A subsequent exact model-input audit identified a more direct cause of the
rigid choice: nine of its 29 named features were read from the wrong columns
after right-of-way contrast fields expanded the dataset schema. The erroneous
inputs reproduce both observed score vectors exactly. Resolving the inputs by
name changes both inspected choices to the straight-green pressure reference,
without retraining. The physical blockage is observed, but it must not be
attributed solely to insufficient representation or tree extrapolation. See
`paper/tsc_v154_rigid_feature_alignment_correction.md` for the isolated correction
and its separate replay.

## Frozen read-only replay

Only the original rigid target-only `cologne1` seed `41242` is replayed, with
the original model, manifest, conversion root, SUMO runtime, initial seed,
60-second warmup, ten-second control interval and 450-second prediction horizon
from `state_conditioned_source_closed_loop_smoke_launch_v3_paired_safety.json`.
The original `evaluate_policy_v3` and original action selector/executor remain
unchanged. A subclass records additional getters and returns the original
selector's decision object.

The simulation begins at 25200 and stops at 26820, after 1,620 seconds. It records
144 pre-action samples at ten-second intervals from 25380 through 26810;
terminal aggregate counts refer to 26820. This window covers the first rigid
long hold, its temporary recovery, the final hold and the onset of sustained
zero-speed observations. It is one trajectory diagnostic, not another efficacy
matrix.

Each sample contains:

- All incoming and directly connected receiving lanes: vehicle count, halted
  count, occupancy, mean speed and sampled lane membership changes. Incoming
  lanes also retain the existing queue proxy unchanged.
- The front vehicle on each incoming lane: position, speed, waiting time,
  next-TLS movement index/current signal and next route edge. Counts of all
  current lane vehicles by their intended next-TLS movement distinguish demand
  for green turns from demand for red straight movements.
- Pre-action executor state; every candidate's rigid score, `green_q`, `red_q`,
  `service_pressure`, downstream occupancy, switch indicator and clearance
  fraction; the original selected candidate and pressure reference.
- Active inserted and pending insertion vehicle counts. Terminal departed,
  arrived and pending totals come from the original evaluator.

The primary inspection is the two shared lanes. If their front vehicles want
red links 2/12 while green-turn demand is counted behind them and receiving
lanes have room, the next correction concerns movement-specific accessible
service. If front vehicles want permitted movements but receiving lanes are
blocked, the next correction instead concerns receiving-space propagation.
If the action is appropriate but the executor fails to display it, the next
correction concerns execution. These outcomes require different changes; this
diagnostic applies none of them.

Every original retained intervention field before 26820 must exactly match
the replay, including action states, scores, local aggregates and executor
fields. Added diagnostic fields are ignored by that comparison. A mismatch
rejects the replay as a reproduction before its lane evidence is used to
attribute the original failure.

An optional native SUMO state and executor context are saved just before the
26700 action. The original SUMO startup enables RNG-state saving. These files
remain under remote `CFCMT_SCRATCH`, and only their paths appear in the result.
No state snapshot or model is downloaded.

## Limitations

The legacy intervention trace contains only the first 256 pressure-override
rows, not all policy decisions. Rigid has 352 accepted overrides over the full
run but only 256 retained rows. Source has two full-run source-induced action
changes but only one appears in its retained pressure-override rows; an absent
change cannot be placed in time from this trace. Pressure agreements are not
recorded and PhasePressure itself has no intervention trace. Full-run phase and
demand aggregates remain available beyond the cutoff.

The trace field `total_queue` is a composite queue proxy, not the number of
stopped vehicles. Its `mean_speed` is the unweighted mean of incoming lane mean
speeds. Sampled lane membership losses include lane changes and are not arrival
throughput. A green lane and zero aggregate speed do not alone establish which
movement is blocking service.

The replay v1 field named `occupancy_pct` contains the unmodified SUMO lane
occupancy **fraction**, despite its label. For example, 23 vehicles of length
4.3 m on the 351.23 m lane yield `0.2815818694`, equivalent to 28.158%. This is
also the canonical unit explicitly documented by the project's causal-mechanism
cache. The v1 JSON remains unchanged; analyses interpret that field as a
fraction. Empty-lane findings are unaffected by this label correction.
The implementation basis is SUMO 1.22
[`MSLane::getNettoOccupancy`](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSLane.cpp),
[`libsumo::Lane::getLastStepOccupancy`](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/libsumo/Lane.cpp),
and the shared TraCI lane handler; the
[official lane-value documentation](https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html)
also describes the length ratio.

A separate implementation defect uses percent-scale constants (`120`, `135`,
and an occupancy ceiling of `100`) with those fractions in the v2 service
pressure, analytic prior and spillback-pressure baseline. That weakens the
intended occupancy response and requires a separate, consistent feature/prior
correction. Multiplying every input by 100 would violate existing fraction
contracts. The original model, original features and original action decisions
were preserved in this replay. The executor's occupied-clearance test uses
vehicle counts and is unaffected. This unit mismatch cannot explain blockage
at the observed times when every receiving lane was empty.

Tripinfo includes inserted unfinished vehicles but excludes vehicles still
waiting to enter. Rigid's 51.34% `throughput_ratio` uses 409 departed vehicles as
denominator; its completion ratio against all 2,015 due vehicles is 10.42%.
The pending-demand figures must therefore accompany tripinfo performance.
PhasePressure has 20 collision incidents in this run; its strong service
numbers do not authorize it as a safe deployment policy.

## Artifacts and execution

Retained summary:
`cf_h2o/results/paper_artifacts/tsc_v150l_cologne_rigid_service_failure.json`.
Rebuild with:

```bash
python3 scripts/data/summarize_cologne_rigid_service_failure.py \
  --output cf_h2o/results/paper_artifacts/tsc_v150l_cologne_rigid_service_failure.json
```

Static context:
`cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/static_tls_context.json`.
The replay module is `cf_h2o.eval.traffic_signal_cologne_service_replay`; its CLI
accepts `--launch-record`, `--tripinfo`, `--out`, and optional
`--state-at-26700`. The launcher provides immutable source and runtime settings;
the module extracts the original experiment inputs from the supplied launch
record. Only the compact result JSON is retrieved after completion.

Replay result:
`cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/replay_v1/result.json`.
Launch record:
`cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/replay_launch_v1.json`.
