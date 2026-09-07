# TSC V130 Intervention Model Screen Result

## Frozen execution

- Scheduler task: `t88708`
- Execution node: `node005`
- Snapshot: `d64a24724e1eb467a3b5`
- Snapshot SHA-256:
  `d64a24724e1eb467a3b584bf67e6a07e36ba0f0f8c337330fb567e61a7cc6ce1`
- Runtime: 727.10 s
- Result:
  `cf_h2o/results/cluster/tsc_v130_intervention_model_screen_20260901/development_v1/result.json`
- Result SHA-256:
  `f7f15c35ab48018f5caf2e384c434c0b5a013cdfa109a5844c564426204dce4e`

The experiment used 22 outer selector-seed folds. Each fold used 11 seeds for
target-model fitting, 10 disjoint seeds for intervention-retention calibration
and one held-out seed for scoring. No source prediction or target-prior
prediction artifact was read.

## Results

| Target-side model | Enabled folds | Interventions | Mean delta | 95% interval | Decision |
|---|---:|---:|---:|---:|---|
| Causal improvement classifier | 2/22 | 18 | -0.0001545 | [-0.0004198, 0] | Reject |
| Group-normalized advantage | 5/22 | 133 | -0.0000184 | [-0.0008052, +0.0008617] | Reject |

The frozen gate required a mean normalized waiting delta no greater than
`-0.0005` and a strictly negative upper 95% bound. Neither model passed.

The classifier's two enabled held-out folds both improved, but its training
balanced accuracy was only 0.5347--0.5517 and it generalized too rarely to
support deployment. The advantage regressor enabled more folds, but two of its
five enabled held-out folds were harmful; one harmful fold had a delta of
`+0.0067375`.

## Consequence

V130 closes source integration for these two intervention models. The failure
is target-side action identification across seeds, not evidence that aligned
source information lacks relative value. A successor must change the
intervention representation or supervision and pass the same target-only
absolute gate before source-veto and source-placebo comparisons are allowed.
