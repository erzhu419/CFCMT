# V150O Dense Pressure-Pairwise Source Utility Result

## Decision

V150O is rejected before closed-loop rollout. The experiment solved V150N's
record-support failure and admitted source use in Atlanta and New York, but the
Atlanta source arm did not beat either source-identity control on the untouched
target evaluation groups. The frozen all-admitted-city gate therefore failed.

## Support Repair

Creating one fixed PhasePressure-versus-rigid label for every eligible state
produced 1,512--3,240 final source records per city, compared with 0--20 under
V150N. Every final utility gate was enabled. Cross-fitted source interventions
also became nonzero in every city, ranging from 8 in Atlanta to 24 in
Ingolstadt. This establishes that V150N's rejection was not evidence that the
binary action pair itself was degenerate.

## Held-Out Effects

Atlanta and New York passed their nested B100 admission checks. On the target
groups excluded from B100, the selected source arm improved rigid CFCMT by
`-0.00265540` in Atlanta and `-0.00591098` in New York. Exact fallback made the
other five city effects zero. The seven-city mean effect versus rigid was
`-0.00122377`, with city bootstrap interval `[-0.00291262, 0.00000000]`.

The source-identity controls separate this apparent policy benefit:

| Target | Source minus placebo | Source minus source-blind |
|---|---:|---:|
| Atlanta | `+0.00024366` | `+0.00058923` |
| New York | `-0.00256922` | `-0.00253117` |

Negative values favor real source evidence. New York is a positive
source-specific result, whereas Atlanta's improvement over rigid is explained
at least as well by target-side state labels and the matched placebo. The
aggregate means favored source by `-0.00033222` versus placebo and
`-0.00027742` versus source-blind, but both city intervals crossed zero.

## Interpretation

V150O answers two separate questions. Dense fixed-pair labels repair the
information-support problem, but a utility model fitted only within one target
city still cannot reliably identify whether source-derived evidence adds value
beyond a target-only state predictor. Tightening or weakening the observed
thresholds would be post-hoc and is not used.

The next method change must therefore alter the learning population: train the
utility relation across pseudo-target source cities while excluding the
evaluated city from every label, candidate, normalization and preprocessing
role. A target-label-free arm and a predeclared few-shot target update curve can
then test whether source mechanism evidence generalizes across cities. This is
scientifically distinct from another target-local gate.

Canonical aggregate:
`cf_h2o/results/paper_artifacts/tsc_v150o_dense_pressure_pairwise_source_utility.json`.
