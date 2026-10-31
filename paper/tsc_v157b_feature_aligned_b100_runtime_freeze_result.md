# V157B v2 Feature-Aligned B100 Runtime Freeze Result

## Failed v1 reuse attempt

Scheduler task `t92500` ended with exit code 1 because the V115 runtime pickles
did not reproduce the corrected V157A B100 arrays. The remote `freeze_v1`
output directory was absent after failure, so no model, placebo, or partial
result was created. Automatic retry `t92511` was cancelled.

Server-side inspection found that the 209,690,586-byte V157A prediction
artifact contains score, uncertainty and context-trust arrays plus metadata,
but no serialized model or model path. Exact refitting was therefore required
to recover general runtime models.

## Corrected v2 result

V157B v2 completed the frozen 16-fit roster: one exact V157A target refit for
audit only, one domain-aligned target refit, seven exact source-component
refits, and seven matched source-label-placebo refits. Across all 158,808
selector rows, the audit target, every source component, their uniform mean,
and the runtime wrapper reproduced the corresponding V157A score, uncertainty,
and context-trust arrays with `numpy.array_equal`. The exact identity gate
therefore passed.

V157B also established why the earlier V123/V157A source attribution was not
valid. Their target-only arm retained three scenario-level Jinan domain
labels, while each source-component arm relabelled the same target-adaptation
rows to the common `jinan` domain. The V157B runtime target corrects that
difference by relabelling all selected B100 rows to `jinan`; as expected, its
scores differ from the historical V157A target scores. That expected diagnostic
difference is not an identity-gate failure.

On the reused 22-seed Jinan selector, the frozen uniform-source arm minus the
domain-aligned target-only arm has mean normalized cost `-0.0135734779`, paired
95% interval `[-0.0178352335, -0.0093450927]`, and improves on 20/22 seeds. It
passes the frozen mean-at-most-`-0.0005` and upper-bound-below-zero gate.

## PhasePressure boundary and authorization

The uniform-source arm still costs `+0.0253374874` relative to PhasePressure
(PP), so the absolute PP gate remains failed. V157B establishes only a corrected
B100 ranking signal on the reused 22-seed Jinan selector. It does not report a
matched-placebo outcome, native branch outcome, fresh-seed result, closed-loop
controller result, cross-city transfer, or superiority over PP.

After both required gates passed, V157B wrote the domain-aligned target-only,
uniform-source, and source-label-placebo rankers plus the PP reference. These
artifacts authorized only the frozen V157C single-focal-TLS, one-action,
450-second Jinan branch experiment. V157C was not run as part of V157B.

## Subsequent V157C outcome

The bounded authorization was later exercised without changing its models,
seeds, checkpoints, or selection rule. Seeds 80314 and 88625 completed, while
seed 27178 was invalid because its fixed 480-second checkpoint had no
action-eligible light. V157C is therefore incomplete and supplies no formal
three-seed inference. Across the two valid seed units, the descriptive
uniform-source-minus-target mean is `+0.0023636831`, with one seed improving
and one worsening. Uniform source is also worse than PhasePressure in both
valid seeds, with mean difference `+0.0036162551`. Thus the native evidence did
not confirm the V157B offline relative benefit and does not support controller
adoption. The missing frozen seed is retained as an invalid unit rather than
replaced or excluded.

## Provenance

The completed result uses protocol
`tsc-v157b-feature-aligned-b100-runtime-refit-freeze-result-v2` and has SHA-256
`fc731f16e9b2ffc4cacdfdd474f67ef91d71f4a7e64f86f10a92ff560d65d7b5`.
The local arm manifest has SHA-256
`02419e1b9b2370978ed0fe4525c1d4600fd13470c7a76edd625d9c1d126ce52f`.
The 7,377,567-byte runtime bundle remains server-side with SHA-256
`a6bdea34175f06a399dc00c1c1df06781e578035a2806c434e0f4eb9be61ad93`.

[Authoritative result](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v157b_feature_aligned_b100_runtime_freeze_result_v2.json)
[Arm manifest](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v157b_feature_aligned_b100_arm_manifest_v2.json)
