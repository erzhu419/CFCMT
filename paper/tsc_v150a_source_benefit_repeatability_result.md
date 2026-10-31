# V150A Source-Benefit Repeatability Diagnostic

## Decision

The V144 source signal is repeatable enough to justify a new development stage,
but not stable enough to support a deployable source selector or a cross-city
source-contribution claim.

This result does not reopen V149, authorize Boston efficacy evaluation, or alter
the V148 conclusion that its deployed gate selected exact target-only fallback
in all seven cities.

## Evidence

V150A reuses the three existing evaluation seeds for each of the seven V144
targets. For each held-out seed, it selects the best source using only the other
two seeds and then compares the selected source with the same-architecture
target-only model. This is a diagnostic cross-validation of source effects, not
a deployable procedure: outcomes from evaluation seeds would not be available
to a B25 target at deployment.

- Mean within-target source-rank Spearman correlation: `0.4667`; median: `0.6000`.
- Mean ordered-pair seed-sign agreement: `0.6667`.
- Ordered source-target pairs with the same strict sign on all seeds: `0.5000`.
- Forced-source leave-one-seed-out improved target-only in `16/21` held-out
  units and all `7/7` city means. Mean effect was `-0.01990`; the worst held-out
  regression was `+0.01407`.
- A source-null-aware variant improved target-only in `15/21` units and `6/7`
  city means. Cologne regressed by `+0.00432` on its city mean.
- Forced-source selection improved the matched placebo in `18/21` units and all
  `7/7` city means, with mean effect `-0.05111`.

Negative values denote lower waiting-aligned error.

## Interpretation

Source identity contains a real, partially repeatable signal. The failure of
V148 therefore cannot be reduced to "no useful source data exist." Instead, the
current deploy-observable city-level representation and utility gate do not
identify that signal reliably. The nonzero held-out regressions also rule out
claiming that city-level source selection is already safe.

The next authorized method work is limited to fresh development data and a
representation diagnostic for movement priority, shared receiving lanes, and
neighboring signal execution. Any successor must preserve exact
same-architecture target-only behavior when all source weights are zero and
must be evaluated without using target evaluation outcomes for selection.

## Artifact

Canonical result:
`cf_h2o/results/paper_artifacts/tsc_v150a_source_benefit_repeatability.json`.
