# V111 target-offline source/guard confirmation outcome

## Status

V111 is a completed, fresh-seed confirmatory experiment and its preregistered
efficacy claim is rejected. The failed result is retained as falsification
evidence; its 56 seeds are no longer eligible for confirmation in later method
development.

## Frozen comparison

The experiment evaluated three paired policies on each of the three Jinan
full-horizon demand scenarios:

- exact `phase_pressure`;
- the strict B100 target-only model under the B100-selected deployment guard;
- the same target anchor plus the B100-selected Hangzhou source component at
  weight 1.0 under the same deployment guard.

Source identity, source weight, and the guard were selected without reading a
target closed-loop outcome. The 56 PCG64 seeds were disjoint from all v93 and
v98 seeds. All 504 prescribed rollouts completed.

## Primary result

| Paired comparison | Mean waiting-time delta | Paired 95% CI | Improved seeds | Collision incidents |
|---|---:|---:|---:|---:|
| source guard - phase pressure | +9.110% | [+8.828%, +9.397%] | 0 / 56 | 0 vs 4 |
| target-only guard - phase pressure | +7.366% | [+7.077%, +7.651%] | 0 / 56 | 0 vs 4 |
| source guard - target-only guard | +1.630% | [+1.374%, +1.888%] | 4 / 56 | 0 vs 0 |

Positive deltas are worse. The source model therefore failed both the exact
PhasePressure comparison and the strict target-only source-contribution
comparison.

## Interpretation

The B100 selector optimized cross-fitted one-step normalized action regret.
That estimand did not identify long-horizon closed-loop waiting performance.
The selected aggressive guard executed roughly two hundred residual phase
changes per rollout and induced states absent from passive B100 support.

The lower observed collision count is not accepted as a causal safety benefit.
Inspection of the three collision-bearing PhasePressure seed-scenarios showed
that paired residual policies did not consistently intervene at the collision
TLS near the event. The difference is compatible with ordinary trajectory
divergence after many earlier phase changes. Safety must be enforced by a
policy-independent executor and evaluated separately from efficacy.

## Consequence for the method

One-step target-offline labels may select source mechanisms, but cannot by
themselves authorize a deployment intervention rate. The next method revision
must use a policy-relevant multi-step target and genuinely predictive runtime
information, then pass a new development gate before any new independent
confirmation is opened. V111 outcomes may be used for development only after
this disclosed rejection.

## Evidence

- Config: `cf_h2o/config/traffic_signal_tsc_v111_target_offline_source_guard_confirmation_v1.json`
- Launch: `cf_h2o/results/cluster/tsc_v111_target_offline_source_guard_20260831/fresh_confirmation_v1_launch.json`
- Audit: `cf_h2o/results/cluster/tsc_v111_target_offline_source_guard_20260831/fresh_confirmation_v1_audit.json`
