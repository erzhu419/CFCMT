# V126 Source-Conditioned Ranker Feasibility Result

## Status

V126 completed under immutable snapshot `b28577a871ba043c59b5` in 140.70 s.
The result SHA-256 is
`368801bf6207d0bc77081e6894921f6051ffeb7c8a374d0acacba26b66d028cf`.
It used 21 target selector seed banks to fit each outer fold and held one seed
out from both fitting and ridge-penalty selection.

## Result

The preregistered gate failed. All three linear arms harmed PhasePressure on
average:

| Arm | Mean normalized delta | 95% CI | Improving seeds | Interventions |
|---|---:|---:|---:|---:|
| Target-only | +0.001513 | [-0.001122, +0.004134] | 8/22 | 2,513 |
| Source-aligned | +0.001907 | [-0.000773, +0.004611] | 9/22 | 2,733 |
| Source-placebo | +0.001351 | [-0.001247, +0.003890] | 8/22 | 2,513 |

The aligned source channels were also worse than target-only by `+0.000394`
(95% CI `[-0.000381, +0.001183]`) and worse than the within-seed whole-group
placebo by `+0.000555` (95% CI `[-0.000536, +0.001592]`). This is not evidence
of usable source contribution.

## Interpretation

V126 rules out the proposed 175-dimensional linear action ranker even under an
abundant target-label budget. It does not contradict V98's paired improvement
or V123's relative source-over-target prediction result: those tests ask whether
source models contain useful information, whereas V126 asks whether a linear
ranker can convert the frozen channels into an absolute gain over
PhasePressure. It cannot.

Combined with V125's pressure-nondegrading oracle mean of `-0.07709`, the
failure localizes the bottleneck to nonlinear matched-action ranking and safe
intervention identification, not to the action set. V126 must not be promoted
or repaired by selecting a different ridge penalty after observing the result.
