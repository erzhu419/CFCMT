# V146 Multicity OOF Compatibility Gate Protocol

## Status and purpose

This protocol is frozen before any V146 target result is generated. V145 showed
that a single adaptation intervention fraction cannot reliably identify a safe
source: it passed pooled H2O+ and matched-placebo comparisons but failed the
target-only comparator. V146 tests one mechanism-specific correction rather
than another intervention threshold.

V146 remains seven-city method development. It is not zero-shot evidence,
untouched-city confirmation, or real-world counterfactual validation.

## Information budget

Each target retains the frozen B25 target-adaptation budget and the disjoint
complete-city evaluation groups used by V143--V145. The B25 groups are
partitioned deterministically into five equal folds. For each fold, target-only
and each one-source CFCMT model are fitted with the other B20 groups and scored
on the held-out B5 features. The B5 outcome labels are not read by the selector.
No evaluation feature or label is used for fitting, source selection, trust
estimation, or source-mass estimation.

The final target-only and one-source models are fitted with all B25 adaptation
groups. Cross-fitting is used only to estimate transfer trust.

## Compatibility and source mass

For each source, let `a` be its OOF action agreement with the target-only model
over the 25 held-out predictions, and let `a0` be the mean random agreement
`1 / number_of_actions` over those groups. The source trust mass is fixed as:

```text
w = clip((a - a0) / (1 - a0), 0, 1).
```

The deployed source score is the target-only score plus `w` times the
one-source residual. The matched placebo is shrunk by exactly the same `w`.
This makes source-null (`w = 0`) part of the method rather than a post-hoc arm.
OOF target-model regret and pairwise rank agreement are recorded as diagnostics
but do not introduce tunable weights.

## Leave-one-city source gate

For evaluated target city `t`, city `t` is absent from every meta label both as
a target and as a source. A source is eligible only when its leave-one-city mean
effect is at most `-0.0005` against both target-only CFCMT and its matched
whole-action-group placebo. Eligible sources are ranked by:

1. maximum chance-corrected OOF compatibility;
2. minimum OOF target-model regret;
3. minimum leave-one-city target-only mean effect;
4. source name as a deterministic final tie break.

No eligible source, or a selected source with zero trust, returns exact
source-null.

## Frozen development gate

The selected compatibility-shrunk method must independently pass against:

- target-only causal CFCMT;
- pooled H2O+;
- the selected source's compatibility-shrunk matched placebo.

For each comparator, the city-equal mean effect must be at most `-0.0005`, the
one-sided 95% upper bound must be below zero, at least five of seven cities must
improve, and maximum city regression must not exceed `0.01`.

Uniform CFCMT and PhasePressure remain descriptive. Failure against any primary
comparator closes V146 without threshold, fold, or trust-formula retuning. A
pass only authorizes a separately frozen untouched-city protocol.
