# TSC v43/r39 full-budget execution audit

This note records every execution attempt before the untouched seed-8171
confirmation. Failed attempts are retained as protocol-engineering evidence;
they are not efficacy trials and produced no metrics used for method choice.

## Attempt v1: missing external diagnostic evidence

- Tasks: `t78189` (Los Angeles) and `t78190` (Jinan).
- Outcome: both failed before model fitting because the immutable source
  snapshot intentionally excludes `results/`, while the v42 seed-7079
  diagnostic failure was addressed through a snapshot-relative path.
- Correction: the v43 runner now requires an explicit, hash-checked
  `--diagnostic-failure-result` from the remote evidence store.
- Scientific boundary: no model artifact and no seed-8171 cache file was
  created.

## Attempt v2: inherited five-fold selector contract

- Tasks: `t78192` (Los Angeles) and `t78193` (Jinan).
- Outcome: all three fitting roles per city completed, then the process failed
  before serialization because the v40 selector rejected the preregistered two
  complete-adaptation-seed folds as not having five folds.
- Automatic retries `t78197` and `t78198` used the same immutable source and
  were cancelled as soon as they were detected. Direct process inspection on
  node003 and node004 confirmed that their parent and child PIDs terminated.
- Correction: `select_target_candidate` now accepts an explicit fold count;
  v40/v41 retain the default five-fold contract, while v43 passes exactly two
  folds and emits
  `target-seed-blocked-risk-adjusted-anchored-selection-v2`.
- Verification: focused regression tests cover both the original five-fold
  selector and the explicit two-fold selector.
- Scientific boundary: no model artifact and no seed-8171 cache file was
  created. In-memory OOF values from this failed run were not inspected or
  retained.

## Attempt v3: admissible confirmation freeze

- Tasks: `t78205` (Los Angeles) and `t78206` (Jinan).
- Immutable snapshot SHA-256:
  `2315f32e19e118af5af15644899927901ccb82d950b292c96fac22d77a0627b1`.
- Runtime source-tree SHA-256:
  `2da93b11804a44e273425903ff76e99987395ca43520f701ad3cc2652ca44623`.
- Outcome: both tasks terminated successfully and direct node inspection found
  no surviving parent or child PID. Los Angeles used all 165 groups (seed-fold
  counts 80/85) and selected `local_cap_1_z_0p25`; Jinan used all 911 groups
  (452/459) and selected `constant_alpha_0p9`.
- Deployment artifacts: Los Angeles method
  `f26ee8e5829e248ef411f77ad3e19ed0d63fd52cdfdae6d8291a9a3aeaaa7aff`,
  Los Angeles baselines
  `c4445ce9157e999bccd099694cfa75d0e5c4070dee54f0efd46220e0efc8657f`,
  Jinan method
  `a34970f1e4f023079650866097359ec3b2509e17787685249a21363cc7b03741`,
  and Jinan baselines
  `f1136d2d08ae2b8880bc0efba566a960dac0ab1cbe30152247fd41f24de0b625`.
- Independent joint-freeze audit: `PASS`, decision
  `authorize_external_seed8171_collection`, SHA-256
  `471e8c34eb7be96411c812c7fe961d5c07f1857e3fb50d2c8d7edd59fc23c596`.
  The audit observed zero seed-8171 cache files at freeze time.

The seed-role boundary remains unchanged throughout all attempts: 5057/6067
are adaptation seeds, 7079 is diagnostic-only, 8171 is the untouched offline
confirmation seed, and 8081/9091 are inaccessible closed-loop seeds until the
offline gate passes.
