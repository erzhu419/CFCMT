# V123 Target-Budget Source-Value Curve Result

## Status and superseding amendment

V123 completed under immutable snapshot `8281502f843b7886a4fe` in 1,438.77 s.
The local and remote `result.json` SHA-256 is
`cc59ab48768901a0e73d91cde878b0a764b6a8e16f8a929df4a2406efd80d944`.
The 209,690,586-byte prediction artifact remains on the server and was not
copied to the local workspace.

V157B later found that the V123/V157A target-only and source-component paths
handled the Jinan domain labels differently. The target-only fit retained three
scenario-level labels, while the source-component fits relabelled the same
target-adaptation rows to `jinan`. The original differences below are therefore
historical model-path contrasts, not isolated source-row effects. V157B
re-adjudicates B100 only; B25, B50, B250, B500, and B1000 remain unrecomputed.

V123 is development evidence on the fixed 22-seed Jinan selector. It is not a
fresh-city confirmation.

## Historical results

Positive values below are worse than exact PhasePressure. The final two columns
retain the original arithmetic and uncertainty summaries; negative legacy-path
differences cannot be attributed to source rows because domain handling was not
held fixed.

| Target groups | Target-only vs PhasePressure | Source-augmented vs PhasePressure | Legacy-path difference | Paired 95% CI | Nested source choice |
|---:|---:|---:|---:|---:|---|
| 0 | n/a | +0.055% | n/a | n/a | Atlanta, weight 1.0 |
| 25 | +7.576% | +4.477% | -3.172% | [-3.607%, -2.768%] | Ingolstadt, weight 1.0 |
| 50 | +3.620% | +2.483% | -1.148% | [-1.574%, -0.721%] | Hangzhou, weight 0.75/1.0 |
| 100 | +3.817% | +2.534% | -1.309% | [-1.653%, -0.964%] | uniform sources, weight 1.0 |
| 250 | +1.589% | +1.563% | -0.043% | [-0.296%, +0.220%] | Atlanta, weight 0.25--0.75 |
| 500 | +1.944% | +1.122% | -0.853% | [-1.160%, -0.550%] | Atlanta, weight 0.75 |
| 1000 | +2.275% | +0.775% | -1.491% | [-2.041%, -1.004%] | Atlanta, weight 1.0 |

The historical paired differences exclude zero at B25, B50, B100, B500, and
B1000; at B25 all 22 held-out selector seeds favoured the source-component
path. Those facts remain numerically correct, but they do not isolate source
information from the Jinan domain-label change.

## Corrected B100 decision

V157B exactly reproduced the V157A B100 arrays for audit, then fitted the
domain-aligned target comparator. On the same 22 selector seeds,
uniform-source minus domain-aligned target-only is `-0.0135734779`, with paired
95% interval `[-0.0178352335, -0.0093450927]` and 20/22 seeds improving. The
corrected B100 relative gate passes.

The uniform-source arm remains `+0.0253374874` worse than PhasePressure. This
result authorizes only the V157C one-action Jinan branch experiment. It does not
establish a placebo-separated source effect, fresh-seed benefit, closed-loop
benefit, cross-city transfer, or superiority over PhasePressure.

[Superseding V157B result](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v157b_feature_aligned_b100_runtime_freeze_result_v2.json)
