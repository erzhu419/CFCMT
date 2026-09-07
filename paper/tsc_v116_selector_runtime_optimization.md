# TSC v116: exact selector runtime optimization

Amended: 2026-09-01 (Asia/Shanghai), while the frozen V116 selector was still
running and before its result JSON or any V117 rollout outcome was available.

## Trigger

The selector had already replaced its own repeated group scans, but the shared
`_group_adjusted_scores` routine still evaluated every group with
`flatnonzero(groups == group)`. On the frozen selector matrix this meant about
44,000 full scans of 158,808 action rows for each model prediction. The same
pattern was also present in the phase-pressure relative-gap helper.

## Exact change

The shared implementation now performs one stable sort of action-group IDs,
validates that every group has exactly one reference, and constructs the
row-to-reference mapping from contiguous group slices. Score, uncertainty,
trust, source weights, selection thresholds, information budgets, seeds, and
all outcome gates are unchanged.

An unsorted 9,091-row, 2,000-group equivalence check produced an elementwise
identical reference map to the previous implementation. At the frozen V116
shape, the optimized mapping processed 158,808 rows and 44,000 groups in
0.0271 s and verified that every row mapped to the reference of its own group.
The regression test also checks the resulting relative scores and uncertainty
values.

The in-progress selector remains bound to snapshot v20 and is not replaced.
V117 execution is bound to snapshot v23 so the exact optimization affects only
runtime, not the frozen selector decision.
