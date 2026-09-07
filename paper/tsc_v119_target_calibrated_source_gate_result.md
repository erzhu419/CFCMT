# V119 Target-Calibrated Source Gate Result

## Status

V119 completed on the frozen pure-waiting estimand and failed its preregistered
development gate. The retained decision is PhasePressure fallback. The result
JSON has local and remote SHA-256
`65f8f4ce5a0c12c37c1213530d2239163b88013a40bee3b87787beaa2d30f503`.

## What was tested

V119 used the V115 B0 source components, 100 target calibration groups, and the
22 label-disjoint V116 selector seeds. Target-only, aligned-source and
group-permuted-placebo arms shared the same ridge family, selected penalty,
feature dimension, B100 groups and cross-fit folds. Selector labels were used
only for nested profile evaluation.

## Result

The shared ridge penalty selected by grouped OOF validation was 1000. The OOF
group-mean MSEs were:

| Arm | Group-mean MSE | MAE |
|---|---:|---:|
| target-only | 0.255423 | 0.419572 |
| source-aligned | 0.255212 | 0.419805 |
| source-placebo | 0.255640 | 0.420002 |

The aligned source arm reduced MSE by only approximately 0.08% relative to the
target-only arm and did not improve MAE. Its median one-sided group conformal
margin was 0.493941, far larger than the predicted action gains. Consequently,
no source, target-only or placebo profile passed the absolute safety gate; all
22 nested folds selected PhasePressure and the source-selection fraction was
zero.

## Interpretation

This result does not reverse V98. V98 showed that source-selected information
improved a strict target-only comparator. V119 asks whether B0 source models
contain enough target-specific information to calibrate safe few-shot action
overrides against exact PhasePressure. They do not.

The corrective experiment is V120/V121. V120 generates group-disjoint OOF
predictions from source-augmented and target-only models that each receive the
same 80-of-100 target groups per fold. V121 then trains the gate on those OOF
predictions and deploys the corresponding full-B100 models on the unchanged
22-seed selector. This directly tests the few-shot source benefit suggested by
V98 without using in-sample B100 predictions.
