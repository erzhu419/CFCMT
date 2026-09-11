# TSC V150D Rigid Mechanism-Prior Integration Protocol

## Question

V150C showed that a target-adapted mechanism model with one source coefficient
prior can outperform the same mechanism architecture without a source prior.
Its matched-placebo separation was weak. V150D asks the narrower deployment
question: can the source-induced mechanism difference improve rigid CFCMT while
remaining distinguishable from a matched placebo?

## Frozen estimator

For each target city, 25 target action groups are divided into five folds. A
group-normalized rigid CFCMT model and a target-only linear mechanism model are
fit on four folds. For every source city and mechanism block, the candidate
score on the held-out fold is

```text
rigid_score + mechanism_source_prior_score - mechanism_target_score
```

The subtraction prevents the target mechanism architecture from being counted
as source value. With zero source-prior strength the two mechanism scores are
identical, so the method returns the rigid score exactly. The source and
mechanism block are selected only from the 25 adaptation groups; all remaining
target groups are evaluation-only.

Source priors use the V150C fixed ridge and prior strengths. The placebo
permutes complete source action groups and re-centers each PhasePressure
reference. V150D reports both an independently selected placebo and the placebo
for the exact source/block chosen by the real selector.

## Development gate

City, rather than source-target pair or seed, is the inferential unit. Closed-loop
development is authorized only if all checks below pass:

1. The city-macro selected-source effect improves rigid target-only.
2. Its city-bootstrap 95% upper bound is below zero.
3. At least five of seven cities improve rigid target-only.
4. The selected source beats the independently selected placebo on average.
5. It beats the same-source/same-block placebo on average.
6. The city-bootstrap 95% upper bound against that matched placebo is below zero.
7. At least four cities beat the same-source/same-block placebo.
8. No city regresses by more than `0.01` against rigid target-only.
9. At least five city selectors admit a non-null source prior.

The seven cities and their evaluation groups are development evidence already
used by V150C. A pass is not fresh-city confirmation and cannot by itself
support a general cross-city deployment claim.
