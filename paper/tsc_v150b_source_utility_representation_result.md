# V150B Source-Utility Representation Result

## Status

V150B completed on 7 September 2026 under protocol
`tsc-v150b-source-utility-representation-diagnostic-v1`.  The frozen
development gate **rejected** the right-of-way-enriched city representation.
The result is a seven-city development diagnostic, not closed-loop evidence or
an independent-city confirmation.

The label-free signature was extracted from 1,008 V114 cache shards.  Of
these, 112 were legal empty shards.  Seventeen admitted scenarios were
observed.  `ingolstadt21`, which is declared by the broader source manifest but
absent from the admitted cache, was reported as missing and was not
regenerated or imputed.

## Result

The estimand is the frozen V144 source-arm cost minus its same-architecture
target-only cost for all 42 directed source-target pairs.  Lower values are
better.

| Representation | Pair MAE | Pearson | Sign accuracy | Mean source-rank Spearman | Forced selected effect | Null-aware selected effect |
|---|---:|---:|---:|---:|---:|---:|
| Existing deploy-observable covariates | 0.02050 | -0.433 | 0.310 | 0.053 | +0.00782 (3/7 improve) | +0.01311 (1/7 improve) |
| Existing + right-of-way/neighbor execution | 0.02249 | -0.200 | 0.333 | 0.273 | +0.00178 (4/7 improve) | +0.00707 (2/7 improve) |

The enriched representation raised mean held-out source-rank Spearman by
0.220 and exceeded the cyclic-placebo median of 0.0449 by more than 0.10.
However, pair MAE became worse, and both selected-source efficacy gates
failed.  Its null-aware selected source increased cost on average and improved
only two of seven cities.

## Decision

The ranking signal is useful diagnostic evidence that execution topology is
not irrelevant, but it is insufficient for source admission.  V150B therefore
does not authorize a city-level right-of-way utility gate or promotion of this
aggregate representation into the controller.

The next source-transfer candidate must operate on state/action-conditioned
mechanism parameters.  It must use one architecture in which zero source
shrinkage is exactly the target-only model, select all hyperparameters without
evaluation-city outcomes, and defeat a matched source-prior placebo before any
closed-loop run.
