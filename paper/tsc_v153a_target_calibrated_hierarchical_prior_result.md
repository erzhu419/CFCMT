# V153A Target-Calibrated Hierarchical Prior Result

## V1 Correction Notice — 2026-09-09

The numerical v1 results and `REJECT` decision below are retained. Code review
found that the purported complete five-fold selector used cached OOF predictions
whose training sets included the current outer held-out fold. Its nested
admission statistics therefore do not measure a fully isolated selection
procedure. The final target evaluation reserve outside B100 remained excluded
from fitting and selection.

The v2 correction uses fresh 60-to-20 inner fits within each outer 80-group
training set, with the original budget, candidates, controls and thresholds.
The completed v2 correction admitted Ingolstadt (`1/7`) and retained the overall
`REJECT` decision solely because the frozen gate requires at least two admitted
cities. Its separate result is reported in
`paper/tsc_v153a_target_calibrated_hierarchical_prior_v2_result.md` and
`tsc_v153a_target_calibrated_hierarchical_prior_v2.json`; v1 is not retuned or
overwritten.

## Decision

`REJECT` (v1). All seven B100 target jobs and the recorded aggregate integrity
checks passed, but the implemented selector admitted no source prior. Every deployed
arm therefore recovered rigid target-only CFCMT exactly. V153A is a scientific
rejection, not an execution failure.

## Evidence

The forced target-calibrated source prior improved five of seven city means
relative to rigid CFCMT. Its city-macro effect was `-0.005919`, with a
city-bootstrap interval of `[-0.016912, +0.001543]`; negative values are
better. Atlanta supplied most of the apparent gain (`-0.036579`). Hangzhou
(`+0.003462`) and Salt Lake City (`+0.005116`) regressed.

Every target used exactly 100 labelled adaptation groups partitioned into five
20-group folds. Candidate predictions on each fold were fitted on the other 80
groups. The selector chose candidates using the other four cached prediction
folds; the correction notice above identifies the resulting indirect label
reuse. All target evaluation groups remained unavailable to fitting,
selection, normalization and threshold decisions. The source-null arm was
exactly equal to rigid CFCMT in all seven cities.

Several full-data candidates appeared useful, but their source-specific gains
did not survive the nested selector. Atlanta tied the zero-centred control;
Cologne tied the matched placebo in nested selection; Hangzhou and RESCO
regressed rigid under nested selection; Ingolstadt and Salt Lake City had too
few nested interventions; and New York failed the source-identity controls.

## Interpretation

V153A v1 did not identify a deployable source prior. The correction prevents
treating its selector audit as definitive evidence against the complete B100
selection procedure. V151A and V152A retain their target-label-free rejections;
the corrected B100 evidence is reported in the separately versioned v2 result.
The result does not negate the established rigid/action-contrast advantage over
the dense H2O+-style residual, nor does it erase the descriptive source
headroom. It shows that the additional source contribution is not deployably
identifiable with the tested information and representation.

No source-aware closed-loop matrix is authorized from V153A. The defensible
paper result is the structured target-adaptation method with explicit
negative-transfer rejection; a positive cross-city source claim requires a
materially new representation or new deployment-observable information and an
untouched-city confirmation, not weaker admission thresholds.

## Runtime And Artifacts

The Atlanta preflight completed in `87.88 s`. The other six cities completed in
parallel on `node001`--`node006`, each in approximately two to three minutes,
with observed peak memory between `1.1` and `1.6 GB`.

- Aggregate: `cf_h2o/results/paper_artifacts/tsc_v153a_target_calibrated_hierarchical_prior.json`
- Aggregate SHA-256: `b27fa323f6314dea3967c6a9ab85bdf976349701c38c4429af676f7e96f8494e`
- Seven target results: `cf_h2o/results/cluster/tsc_v153_target_calibrated_hierarchical_prior_20260908/`
- Frozen protocol: `paper/tsc_v153a_target_calibrated_hierarchical_prior_protocol.md`
