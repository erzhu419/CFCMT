# TSC v40/r36 Cross-Fitted Anchored Selector

Date frozen: 2026-08-09. Written after the integrity-valid v39b anchored-shrinkage development failed and before generating any v40 out-of-fold result. The v39b audit SHA-256 is `91058c229f1bbbaae1de7639af07598a95913d4ac2594d5f2e6f3cb0f3089796`.

## Fixed diagnosis

The v39b pairwise correction materially improves five city groups, is slightly useful after shrinkage in Hangzhou, and regresses Salt Lake City. A single global alpha therefore fails because the sign and safe magnitude of the correction are target dependent. The 60 target simulator groups already available to both base models must supply target-specific selection evidence without using any held-out evaluation group.

## Data and base candidates

- Frozen 18-network counterfactual cache SHA-256: `14c6dfad87999e55fa86fa942ac0ca2334129f4d1662b3c31a6e4fc7aa74c242`.
- Frozen expanded source-rule baseline SHA-256: `f9223a7001078d9a6df0e1a2ad4bfe1d2bc59fc2547621f092e8be702596dbfa`.
- Target networks: the 14 capacity-admitted v39b networks. The four insufficient Hangzhou networks remain source-only.
- Target information budget: exactly 60 safe complete matched action groups.
- Base candidate scores: the unchanged 31 v39 candidates, including 11 constant alphas and 20 local disagreement-confidence shrinkage rules.
- Final evaluation metrics: the integrity-valid v39b candidate metrics outside B60; no candidate is refit or redefined.

## Five-fold target OOF protocol

The same deterministic seed-stratified, time/TLS-balanced split assigns B60 to five folds of 12 groups. For each fold, both base models are fitted on the other 48 groups using identical source data and target groups. All 31 candidate regrets are evaluated on the 12 omitted groups. Thus each B60 group has one out-of-fold prediction.

For target `n`, candidate `c`, and fold `f`, let `r_ncf` be mean normalized action regret and let `a` denote the exact anchor. Define

- `d_ncf = r_ncf - r_naf`;
- `D_nc = mean_f(d_ncf)`;
- `SE_nc = std_f(d_ncf, ddof=1) / sqrt(5)`;
- `U_nc(lambda) = D_nc + lambda * SE_nc`;
- `I_nc = -D_nc / mean_f(r_naf)`;
- `R_nc = max_f(max(d_ncf, 0))`.

## Frozen target selector grid

A selector is indexed by candidate scope, minimum OOF relative improvement `tau`, maximum fold regression `rho`, and standard-error multiplier `lambda`:

- scope in `{constant, all}`;
- `tau` in `{0.00, 0.01, 0.02, 0.05}`;
- `rho` in `{0.00, 0.02, 0.05, 0.10}`;
- `lambda` in `{0.0, 0.5, 1.0, 2.0}`.

A non-anchor target candidate is admissible only if `I_nc >= tau`, `R_nc <= rho`, and `U_nc(lambda) <= 0`. Among admissible candidates, select minimum `U_nc(lambda)`, then minimum OOF mean regret, lower mean correction weight, and lexicographic key. If none is admissible, select the anchor. This decision reads only B60 OOF evidence.

## Nested city-level selection

Each selector is applied independently to every target network. Network evaluation regrets are averaged within city and cities are equally weighted. In every leave-one-city-out fold, a selector is admissible on the other six cities only if it improves macro regret by at least 2%, improves at least four cities, and has maximum city regression at most `0.01`. Select minimum macro regret, then minimum worst-city regret, lower mean selected correction weight, and lexicographic selector key. If none is admissible, use always-anchor.

## Success rule

The seven held-out LOCO decisions must jointly improve city-macro regret by at least 2%, improve at least five of seven held-out cities, and have maximum city regression at most `0.01`. The selector selected once on all seven development cities must independently satisfy the same 2%, five-city, and `0.01` conditions. Both gates must pass before the selector is frozen.

If v40 fails, Los Angeles and Nanchang remain sealed and a new development protocol is required. No external-city CFCMT prediction or outcome may be inspected during v40 development.
