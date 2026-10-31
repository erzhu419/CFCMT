# TSC V150K State-Conditioned Source-Utility Result

## Decision

`PASS` for closed-loop development. The nested state-conditioned utility gate
admitted source-mechanism interventions in three of seven development cities:
Cologne, New York and RESCO synthetic. The other four cities recovered rigid
CFCMT exactly.

## Evidence

- Aggregate artifact:
  `cf_h2o/results/paper_artifacts/tsc_v150k_state_conditioned_source_utility.json`
- SHA-256:
  `752b4eddde0b85f826f3db6e5108f0c451099a6e6ac3d60aa81fa453df912b01`
- Target information budget: 100 labelled action groups per city.
- Source-prior strength: `0.025`.
- Utility model: ridge regression over deployment-observable state, model
  disagreement and mechanism identity, with a held-out 0.75 upper error bound.
- Source admissions: `3/7` cities.
- City-unit mean selected effect versus rigid target-only CFCMT:
  `-0.00051819`, 95% interval `[-0.00135816, -0.00004686]`.
- City-unit mean selected effect versus the nested same-capacity placebo:
  `-0.00085633`, 95% interval `[-0.00183129, -0.00005597]`.
- No city regressed because rejection invokes the exact rigid fallback.

## Interpretation

V150K is the first stage in the V150 chain to identify incremental source-city
value against both the same-architecture target-only model and a nested matched
placebo. It succeeds by changing the supervision unit from one source choice per
city to conditional source-mechanism interventions at observable decision
states. This supports implementing the frozen selector in a closed-loop SUMO
controller.

The result is not a fresh-city or policy-effect confirmation. The city-level
intervals exclude zero, but the seed-level intervals cross zero for both
comparisons. The seven cities remain development units, and the measured
offline gain is small. Closed-loop development must therefore compare the
source-aware controller with rigid target-only CFCMT, the same-capacity placebo
and PhasePressure before any method freeze or external-city claim.
