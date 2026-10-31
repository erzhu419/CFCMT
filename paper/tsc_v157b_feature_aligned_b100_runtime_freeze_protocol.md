# V157B v2 Feature-Aligned B100 Runtime Refit Freeze Protocol

## Purpose

V157A selected the uniform seven-source B100 arm on all 22 selector folds, but
its source-minus-target comparison used different domain handling on the two
sides. The V157A target-only path retained the three Jinan scenario domain
labels. The source-component path relabelled those same target-adaptation rows
to the common `jinan` domain. The reported positive source contribution is
therefore not sufficient to authorize a runtime comparison.

V157B v2 first reproduces the V157A result for audit, then fits a
domain-aligned target comparator and repeats the fixed B100 source gate. It
prepares models only; it does not establish branch or controller performance.

The failed v1 task `t92500` reused older V115 model pickles. They did not
reproduce the corrected V157A arrays, so v1 failed before creating its output
directory; automatic retry `t92511` was cancelled. Their paths and hashes are
retained only as failure provenance. V157B v2 neither loads nor evaluates them.

## Sixteen shared B100 fits

V157B v2 loads the immutable V157A banks once and runs 16 fits in one
fork-based process pool with one numerical thread per worker:

1. one exact V157A-path target refit, used only for reproduction audit;
2. one runtime target refit after relabelling all selected B100 rows to
   `jinan`;
3. seven exact V157A-path source components; and
4. seven source-label placebo components.

The audit target and seven source components must reproduce every frozen V157A
B100 score, uncertainty and context-trust array with `numpy.array_equal`. The
uniform source component mean and feature-name binding must also match. The
exact V157A target refit is never written into the runtime arm.

If this exact identity check fails, execution aborts before writing
`result.json`, a runtime bundle, an arm manifest, a placebo mapping, or any
V157C authorization. A failed identity check cannot be interpreted through the
corrected source gate.

Full-selector arrays are materialized once for the audit target, domain-aligned
target, each individual source component, and the deployed uniform wrapper. The
wrapper itself is invoked once so its output remains separately checked against
the component mean. The exact audit, domain-alignment diagnostic, and corrected
source gate then reuse those arrays.

## Corrected source gate

The fixed candidate is `uniform_all_sources__source_weight_1`. On the same 22
V157A selector seeds, V157B computes each seed's mean normalized-cost difference
between the exact uniform source arm and the domain-aligned target-only arm.
Runtime freezing requires both:

- mean source-minus-target cost at most -0.0005; and
- paired-bootstrap upper 95% bound strictly below zero.

If this gate fails, V157B writes only `result.json`. It writes no runtime
bundle, arm manifest, placebo mapping, or V157C authorization.

## Matched source-label placebo

Within each source scenario and action-count stratum, seed 20260911 assigns
each action group another group's whole ordered vector of non-reference
waiting-cost deltas. The PhasePressure reference stays zero. The mapping uses
no selector or branch outcome, and Jinan B100 labels remain unchanged. The
seven placebo components use the same rows, B100 groups, model families and HGB
settings as the exact source refits.

## Frozen output

After all gates pass, the bundle contains `target_only` (the domain-aligned
target), `uniform_source`, `source_label_placebo`, and the `phase_pressure`
reference. `runtime_models.pkl`, `arm_manifest.json`,
`placebo_donor_mapping.json`, and `result.json` then authorize only the
separately frozen V157C single-focal-TLS same-state branch experiment.

This authorization is limited to the B100 comparison on the reused 22-seed
Jinan selector. V157B contains no matched-placebo outcome, native branch
outcome, fresh-seed result, closed-loop controller result, unseen-city result,
or evidence of superiority over PhasePressure. The source arm's absolute
PhasePressure comparison remains a separate failed gate.
