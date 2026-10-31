# TSC V150I Sample-Coherent Source-Prior Protocol

## Defect Under Test

V150H increased the target-labelled budget but kept source-prior strength at
`0.10`. Because the target group weights are normalized to total mass one, this
does not let B100 evidence dominate the prior more strongly than B25 evidence.
That contradicts the intended shrinkage interpretation and is a concrete model
defect rather than a new selector hypothesis.

## Single Change

V150I sets

\[
\lambda(B)=0.10\frac{25}{B},
\]

giving strengths `0.10`, `0.05` and `0.025` at B25, B50 and B100. Everything
else is inherited unchanged from V150H: nested target groups, common evaluation
outside B100, target-only architecture and information budget, 36 candidates,
five-fold complete-selector crossfit, matched placebo, admission margins and
exact rigid fallback.

The B100 gate is identical to V150H. A pass authorizes source-aware closed-loop
development at B100 only. This target-offline experiment is neither zero-shot
nor fresh-city confirmation.
