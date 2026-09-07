# V122 B100 Source-Reliability Diagnostic Result

## Status

V122 completed under task `t87698` and rejected B100 source reliability
weighting. The retained result JSON has SHA-256
`b459b939c2d815b50fbad2eb3e5abdf0adc25cce009392c1dbe5287e83d03ca5`.

## Result

The architecture-matched target-only OOF policy delta relative to PhasePressure
was -0.024851. Every individual source-augmented model was worse: Atlanta was
closest at +0.001888, followed by Cologne at +0.033469; the other five source
models ranged from +0.040432 to +0.085308.

The best cross-fitted reliability ensemble was inverse-MSE weighting with 25%
source contribution. Its policy delta was -0.022369, substantially better than
its matched source-permutation placebo (+0.011479), but worse than target-only
by 0.002481. The source contribution gate therefore failed.

## Interpretation

V121 did not fail merely because symmetric averaging hid one strong source.
There is no individual B100 source model that outperforms the matched target-
only model under this OOF diagnostic. The result does not invalidate V98,
because V98 used an earlier selector and a differently structured target-only
comparator. It does show that V98 cannot be promoted as architecture-matched
causal source efficacy without a new budget-dependent test. V123 performs that
test on one fixed 22-seed development selector.
