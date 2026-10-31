# V127 JSON Serialization Amendment

## Trigger

The first V127 execution (`t87922`) completed its model fitting and fold
evaluation but failed before publishing a result. Strict JSON serialization
rejected the mathematical `+infinity` sentinel used for the intervention
threshold of a disabled calibration profile:

```text
ValueError: Out of range float values are not JSON compliant: inf
```

No result JSON or efficacy outcome was recovered from that process. Automatic
retry `t88263` was cancelled before launch because it would have repeated the
same deterministic writer failure.

## Correction

The scientific protocol, input artifacts, model family, folds, calibration
rules and pass gate are unchanged. A disabled profile is now represented as:

```json
{
  "enabled": false,
  "predicted_advantage_threshold": null
}
```

Execution explicitly accepts no intervention whenever `enabled` is false, so
`null` is only a standards-compliant representation of the previous
nondeployable `+infinity` sentinel. A regression test serializes a complete
fold with `allow_nan=false` before the rerun is authorized.

The rerun uses a new immutable snapshot, output directory and task signature
ending in `v2-json-safe`. The failed execution remains part of the audit trail
and is not an efficacy result.
