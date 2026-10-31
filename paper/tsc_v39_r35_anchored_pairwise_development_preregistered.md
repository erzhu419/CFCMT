# TSC v39/r35 Anchored Pairwise Development

Date frozen: 2026-08-09. Written after the valid v38/r34 Salt Lake confirmation failed and before generating any v39 anchored-blend result. Salt Lake is now explicitly an expanded development domain and cannot be reused as an independent confirmation domain.

## Fixed diagnosis

The v38 integrity audit passed, but the globally deployed pairwise model increased Salt Lake macro normalized action regret by 12.3754% relative to the equally informed group-normalized fallback. Both Salt networks regressed. The failure is retained at audit SHA-256 `831b8890d595dbfd7c0572fbb836d161d0e543ff03f859561406a4c112b5883b`.

Pairwise optimal-action accuracy did not uniformly decrease; the material failure was tail cost from a smaller number of wrong rankings. v39 therefore treats antisymmetric pairwise prediction as a high-variance correction to a group-normalized causal anchor, not as a standalone replacement.

## Expanded development set

- Counterfactual bank: the immutable 18-network source-plus-Salt cache, aggregate SHA-256 `14c6dfad87999e55fa86fa942ac0ca2334129f4d1662b3c31a6e4fc7aa74c242`.
- Development city groups: RESCO synthetic, Cologne, Ingolstadt, Atlanta, Hangzhou, New York, and Salt Lake City.
- Every network is evaluated as a target using the same-city holdout rule: all networks from its city group are excluded from source fitting.
- Target information: exactly 60 deterministic complete simulator counterfactual action groups.
- Evaluation: all safe complete target groups outside those 60 groups.
- Both base models receive identical source data and identical 60 target groups.

Salt Lake generalized-pressure source costs must be collected and audited under the existing v25 source-rule protocol before Salt can enter source fits for the other development cities. The original 16-network source-rule baseline remains immutable; an expanded baseline is a new union artifact.

## Fixed base models

- Anchor: `causal_group_normalized_rigid_advantage`.
- Correction: `causal_antisymmetric_pairwise_advantage`.

For action row `a` in matched group `g`, let `s0(a)` be the anchor score, `s1(a)` the pairwise score, `u0(a)` and `u1(a)` their uncertainties, and `delta(a)=s1(a)-s0(a)`.

The constant shrinkage family is

`s_alpha(a) = s0(a) + alpha * delta(a)`,

with `alpha` in `{0.0, 0.1, ..., 1.0}`.

The local anchored family first computes

- `scale_g = max(ptp(s0[g]), median(u0[g]), 1e-6)`;
- `d_g = ptp(delta[g]) / scale_g`;
- anchor action `a0 = argmin s0[g]`;
- pairwise action `a1 = argmin s1[g]`;
- `m_g = max(s1(a0) - s1(a1), 0) / scale_g`;
- `q_g = (u1(a0) + u1(a1)) / scale_g`.

For disagreement cap `c` and confidence multiplier `z`,

`alpha_g = min(1, c / max(d_g, 1e-12)) * I[m_g >= z*q_g]`,

and `s_g(a)=s0(a)+alpha_g*delta(a)`. This gate uses only model predictions available at deployment; it never reads target evaluation costs.

Frozen grid:

- `c` in `{0.25, 0.5, 1.0, 2.0, 4.0}`;
- `z` in `{0.0, 0.25, 0.5, 1.0}`.

No additional feature, threshold, or target-specific selector may be added after v39 results are generated.

## City-level selection

Networks are averaged within city group, then city groups are weighted equally. Every candidate is compared with the exact anchor (`alpha=0`).

For each leave-one-city-out fold, select on the other six city groups. A non-anchor candidate is admissible only if, on those six groups:

1. macro regret improves by at least 2%;
2. at least four city groups improve strictly;
3. maximum city-level absolute regression is at most `0.01`.

Among admissible candidates, choose minimum macro regret, then minimum worst-city regret, then lower mean correction weight, then lexicographic candidate key. If none is admissible, select the anchor. The selected candidate is evaluated once on the omitted city group.

## Development success rule

The anchored family advances only if the seven held-out LOCO decisions jointly achieve:

1. at least 2% city-macro regret improvement over always using the anchor;
2. at least five of seven held-out city groups improve strictly;
3. maximum held-out city absolute regression is at most `0.01`.

The candidate selected once on all seven development city groups must independently satisfy the same 2% macro, five-of-seven breadth, and `0.01` regression conditions. Passing freezes that single global candidate for external confirmation. Failing keeps LA and Nanchang sealed and triggers a new development protocol rather than post-hoc v39 tuning.

## Untouched external confirmation candidates

Two local LibSignal/CityFlow bundles are reserved before v39 model selection:

- Los Angeles 1x4 flow SHA-256 `60d00b06111d63530302f2b401cd753686d73f8c44e0c30434bfc14295520180`, roadnet SHA-256 `7ec7b5b8a1855c454d56a13112b0435a5a178194b3029525f331402a607f8c24`;
- Nanchang flow SHA-256 `3c5b952ae79e3717ad8b85cfa68086d3dd5a64794e800889f1da07224ca4722e`, roadnet SHA-256 `b0e455f3ee894b33bd54b5af8d461df6f4380b611a11941e1ab8bb1cc04f92f2`.

Conversion and safety admission may inspect topology, demand counts, collisions, teleports, and rule-controller feasibility. No CFCMT counterfactual action-ranking or model result from either city may be generated or read until a v39 method passes and a separate confirmation protocol is frozen. A city may be replaced only for documented conversion or safety-admission failure, never because of CFCMT efficacy.
