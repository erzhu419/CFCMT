# TSC V157C Jinan Native One-Action Branch Result

**Status: INCOMPLETE.** The frozen three-seed protocol did not produce its
third analysis unit. Seed 27178 reached the fixed 480-second checkpoint with no
scorable traffic light. The checkpoint remains at 480 seconds and the planned
window remains in the frozen roster as invalid and missing, rather than as an
outcome; it was not moved, dropped, replaced, or counted as zero. Consequently
no three-seed aggregate, confidence interval, or primary-gate decision is
reported.

## Execution outcome

The frozen roster was seeds 80314, 88625, and 27178 at checkpoints 300, 480,
and 660 seconds. Tasks `t92562` and `t92563` each returned a complete PASS seed
result with all three windows and 10 native simulations. Both reported zero
collision events and zero teleport increase.

Task `t92564` ran the PhasePressure probe/baseline for seed 27178 and then
failed before any learned branch with:

```text
ValueError: V157C checkpoint 480 has no scorable TLS
```

It wrote neither `result.json` nor the full trace. Automatic attempts `t92565`
and `t92566` encountered the existing-output/scratch refusal before simulation;
the former failed and the latter was cancelled. These attempts add no native
result and do not repair the missing seed.

## Two-seed description

Lower cost is better. These values average each available seed over all three
fixed checkpoints and then average the two available seed means. They are
descriptive only.

| Contrast | seed 80314 | seed 88625 | two-seed description |
|---|---:|---:|---:|
| uniform source - target only | -0.00496399 | +0.00969136 | +0.00236368 |
| uniform source - source-label placebo | -0.00463477 | 0.00000000 | -0.00231739 |
| target only - PhasePressure | +0.00771091 | -0.00520576 | +0.00125257 |
| uniform source - PhasePressure | +0.00274691 | +0.00448560 | +0.00361626 |

Across the six valid primary windows, uniform source versus target only has two
negative, two positive, and two exact-zero contrasts. The four windows where
the two arms selected different actions split two negative and two positive.
The two-seed mean therefore puts uniform source 0.00236368 cost units, or
0.191%, above target only. Uniform source is also above PhasePressure in both
available seed means, by 0.00361626 overall.

The favourable source-placebo description is not a broad source effect. The
two arms select the same action in five of six windows; the entire nonzero
contrast comes from seed 80314 at checkpoint 660.

The completed windows also locate the main failure more directly. Uniform
source made five effective overrides whose stored predicted advantage over
PhasePressure was positive; only two lowered native 450-second cost and three
raised it. Target-only and placebo each improved only one of their five
effective overrides. At the four windows where source and target chose
different actions, ordering their predicted PhasePressure advantages agreed
with the native source-versus-target outcome only once. The missing third seed
therefore is not the sole obstacle: action-benefit ranking is poorly calibrated
to the native continuation even within both complete seeds.

## Evidence and decision

- Incomplete aggregate:
  `cf_h2o/results/paper_artifacts/tsc_v157c_jinan_native_action_branches_incomplete_v1.json`,
  SHA-256 `cc2595f5a938309281bcb2aa32e262b1d7235f4f809b182c867cec7246e7a6fa`.
- Failed-seed scheduler evidence:
  `cf_h2o/results/cluster/tsc_v157c_jinan_native_action_branches_20260911/native_v4/failed_seed_27178_evidence.json`,
  SHA-256 `4b9f5e70335fecd0322419ccafe6267e96c050a490771dac6bb194930bfcb9d8`.
- PASS result for seed 80314: SHA-256
  `cd84ad4487c7417f4aacf65e128c0aa5909c3072d980ce0ff651882d173b98cf`.
- PASS result for seed 88625: SHA-256
  `91ac6f8d80fdfb068be82aae719aa050fc926ea7a1274d70fa721e5a65575d4d`.
- Frozen launch `launch_v7.json`: SHA-256
  `3a68bf6b9e3a9e8598e8d47c32c9a654219b38e0136a235cc37f4a17f2dd947d`;
  derivation `derived_snapshot_v7.json`: SHA-256
  `40a372b16d9d175eaf653857f7861fdf3244a5b2c2a7b5bf56ad48e11f0bcb8f`.

V157C is closed as an incomplete development diagnostic. Re-running the same
deterministic fixed checkpoint cannot create a valid missing analysis unit;
moving the checkpoint or replacing the seed would define a new protocol. The
two PASS seeds do not establish the V157B offline benefit in native execution
and do not support fresh-seed transfer, unseen-city transfer, full closed-loop
performance, simultaneous-network control, safety superiority, or superiority
to PhasePressure.
