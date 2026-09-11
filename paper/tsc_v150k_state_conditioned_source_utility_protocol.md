# TSC V150K State-Conditioned Source Utility Protocol

## Question

V150J rejected city-level source selection after every apparently useful
full-data candidate failed complete cross-fitting. V150K asks a different
question: can target-offline labels identify the decision states in which one
source-mechanism proposal improves the rigid CFCMT action?

## Fixed Inputs

- Seven development cities and the fixed groups outside the B100 reserve.
- Rigid CFCMT is the target-only baseline and exact source-null fallback.
- Six source cities times six mechanism blocks give 36 proposals per target.
- The V150I sample-coherent source-prior strength remains `0.025`.
- Source mechanisms use the V150I single-conditioner representation. The
  V150J conditional representation is not retained after its rejection.

## Utility Supervision

For a state where a proposal and rigid CFCMT select different actions, the
label is proposal cost minus rigid cost under the frozen 450 s counterfactual
estimand. The linear gate sees only pre-action state, normalized disagreement
margins, source-induced score corrections and mechanism-block identity. Source
city identity is deliberately absent. A proposal is eligible only when the
cross-fitted upper bound on its cost difference is below `-0.0005`.

## Complete Nesting

Each of five outer folds is evaluated by a pipeline fitted on the other 80
groups. Within those 80 groups, four inner candidate fits produce out-of-fold
utility records. The utility ridge and its one-sided 0.75 error margin are then
fit without the outer fold. Candidate choice is per state and may return rigid.
The matched placebo receives a separately trained pipeline with the same
capacity, folds, target labels and gate.

The city-level pipeline is admitted only if outer-fold mean gains versus both
rigid and matched placebo are at least `0.0005`, at least four of five folds are
non-degrading against each comparator, and at least five outer-fold states use
a source proposal. Rejection returns the untouched evaluation set to rigid
CFCMT exactly.

## Development Gate

At least two cities must admit the nested pipeline. No city may regress rigid
CFCMT, every admitted city must improve rigid and matched placebo on the fixed
evaluation set, and every rejected city must be an exact fallback. Passing
authorizes closed-loop development only. It does not establish a fresh-city or
real-world claim.
