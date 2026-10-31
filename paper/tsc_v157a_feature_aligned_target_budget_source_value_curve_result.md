# V157A Feature-Aligned V123 Source-Value Replay Result

## Historical replay and superseding amendment

The correction-only replay completed and all frozen input, split, and fit
invariants passed. Resolving `PairwiseActionAdvantageRegressor` inputs by stored
feature names left every reported B25--B1000 model-path difference unchanged
from V123; corrected minus legacy difference was exactly `0.0` at all six
budgets.

V157B subsequently showed that this replay preserved a Jinan domain-handling
confound: the target-only fit retained three scenario labels, while the source
components relabelled the same selected target rows to the common `jinan`
domain. V157A therefore verifies the historical arrays but does not isolate a
source-row contribution.

At B100, the historical comparison selected uniform source weight 1 in all 22
selector folds. Its source-minus-target mean was `-0.0130941914`, paired 95%
interval `[-0.0165322629, -0.0096411054]`, with 19/22 seeds improving. These
numbers remain provenance for the confounded path and no longer authorize the
branch experiment.

## PhasePressure boundary

The B100 source policy costs `+0.0253374874` relative to PhasePressure, while
the historical target-only path costs `+0.0384316788`. PhasePressure (PP) is the
untrained heuristic baseline. The original V123 absolute gate remains rejected.

## V157B adjudication

V157B exactly reproduced every V157A B100 target/source score, uncertainty, and
context-trust array with `numpy.array_equal`, then corrected the target domain
handling. On the reused 22-seed Jinan selector, uniform-source minus the
domain-aligned target is `-0.0135734779`, paired 95% interval
`[-0.0178352335, -0.0093450927]`, with 20/22 seeds improving. That corrected
B100 gate passes and supersedes V157A's branch authorization.

It authorized only the frozen V157C same-native-state, single-focal-TLS,
one-action 450-second Jinan branch experiment. V157C was subsequently attempted:
two seed units completed, one frozen seed was invalid at a fixed checkpoint,
and the valid units did not confirm a source advantage. V157A itself provides
no matched-placebo outcome, branch outcome, fresh-seed result, full closed-loop
result, cross-city transfer result, or superiority over PP.

## Execution and provenance

Scheduler task `t92484` ran on `node005` for `1203.412674` seconds. It reused
the V123 counterfactual costs and ran zero new SUMO simulations. The derived
snapshot SHA-256 is
`5a4b8d5793b359e2d16dabff24c9cc606fa13f5ee7e8d288d83706587e0a63d4`;
the corrected result SHA-256 is
`c8e518a47070fd5ba23cc3b047314d8dddc255710d6ac789f62240e0b86f8821`.
The 209,690,586-byte prediction artifact remains on the server with SHA-256
`3f199276f08ab67e855d9598900d4023ec1f7c4c3529b4361e92b0e851bf983a`;
only the 457,870-byte result JSON was retrieved.

[V157A aggregate](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v157a_feature_aligned_target_budget_source_value_curve.json)
[Superseding V157B result](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v157b_feature_aligned_b100_runtime_freeze_result_v2.json)
