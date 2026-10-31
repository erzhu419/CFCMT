# V156A Feature-Aligned Dense Source-Utility Recalculation

## Question

Does the V150O source-specific New York result survive the confirmed V154
rigid-model feature-binding correction?

## Frozen change

V156A repeats V150O on the same seven development city groups, B100 target
budget, B100 reserve, folds, source bank, cached target/source counterfactual
costs, candidate pool, placebo, source-blind control, utility model and
thresholds. The sole method change is that the already fitted rigid ranker's
inputs are resolved by its stored feature names in each evaluation dataset.
No new counterfactual trajectory is collected.

Here, the frozen labels are the cached target and source counterfactual costs.
The dense PhasePressure-versus-rigid utility records are recomputed because the
corrected rigid action can differ from the positionally bound action. No SUMO
counterfactual is rerun.

The replay is executed from the original V150O snapshot with only
`cf_h2o/traffic_signal/action_ranker.py` replaced. The snapshot, source-cache
root, conversion root, source manifest, fit protocol and five prerequisite
result hashes are frozen in the V156A configuration. The corrected and legacy
results must also contain identical source-bank audits and target/reserve
splits.

The old positional binding read nine of the 29 rigid inputs from shifted
columns after right-of-way fields had been inserted. Because V150O builds its
dense PhasePressure-versus-rigid labels and utility features around that rigid
score, its pre-correction New York result is not usable without this replay.

## Decision

The original V150O aggregate gate is unchanged: it requires at least two
admitted cities, no city-level regression and exact target-only fallback for
rejected cities. This remains the global source-selector decision.

A separate, predeclared city-specific follow-up gate requires that the nested
cross-fitted selector admits that city and that its selected policy has lower
reserve cost than target-only rigid, the matched placebo and the source-blind
control. A city satisfying all four conditions may enter a separately frozen,
untouched-seed closed-loop confirmation even if the global gate fails. This is
a development decision for that city; it neither overrides the global result
nor makes the city an independent external test.

If no city passes after the correction, this dense source-selector family is
stopped. If a city later passes the untouched-seed closed-loop confirmation,
the next independent-city experiment uses Bologna's three road-only scenarios
(`acosta`, `joined`, `pasubio`) with collision counts retained as diagnostics.

## Boundary

V156A is a correction-only replay on reused development data. It cannot support
a fresh-city, closed-loop, safety or universal-superiority claim.
