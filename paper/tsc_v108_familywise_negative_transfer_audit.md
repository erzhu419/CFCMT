# TSC v108: family-wise negative-transfer audit

Date: 2026-08-31.

This post-hoc robustness audit reads only the frozen v98 target-offline
cross-fitted action-regret matrices. It does not read the 56-seed closed-loop
confirmation outcomes.

The revised selector controls the one-sided family-wise error rate at 0.05
across all 24 nonbaseline source-city and source-weight candidates using a
Bonferroni critical multiplier of `2.9137263183`. Transfer candidates must also
improve by at least 0.5% at the simultaneous upper bound and regress by no more
than 1% in any fold. If no candidate passes, deployment is the exact target-only
model. Among simultaneously certified candidates, the selector minimizes mean
normalized action regret.

For Jinan, the revised selector retains Hangzhou with source mass 1.0. Its mean
regret delta is `-0.1018629065`, its simultaneous upper confidence bound is
`-0.0120414195`, and its worst fold delta is `-0.0267188499`. For Los Angeles,
no source passes and the selector returns source mass zero.

This strengthens the interpretation of the earlier v98 result: the Jinan
source effect is not an artifact of a pointwise 1.645 critical value, while the
same mechanism rejects source information where target-offline evidence does
not support transfer. The audit remains retrospective; the same selector is
frozen prospectively for the unseen-city experiment.
