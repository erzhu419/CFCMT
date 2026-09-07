# v97 LuST unseen-city safety candidate outcome

## Decision

The complete LuST DUE-static variant is rejected before controller efficacy
evaluation. The fixed safety gate required zero collisions and zero teleports
under the source-native fixed-time signal plans. The first admission seed was
terminated after the collision gate had failed irreversibly; running the
remaining horizon or additional seeds could not restore eligibility.

## Frozen candidate

- Source: `lcodeca/LuSTScenario`
- Source commit: `5edb7ecb9ad196c39172b6eb95d19ed789f4b1a6`
- Variant: complete `due.static.sumocfg` demand, including all three local DUE
  route files, bus lines, and transit traffic
- Demand: 288,250 vehicles over the full day
- Signals: 201 source-native static traffic-light programs
- Backend: libsumo 1.22.0
- Seed: 5057
- Conversion manifest SHA-256:
  `9d31634fe09dbc128e0128742dd11850f847c544d4b9d51ef54025fa64d26764`

## Observed rejection evidence

At the last complete summary step before termination (`t = 32,016 s`), SUMO
reported 81,704 loaded vehicles, 75,088 inserted vehicles, 60,607 arrivals,
1,830 collisions, and zero teleports. The collision count is strictly above
the preregistered limit of zero. This partial run is evidence of rejection,
not a completed full-day performance estimate.

The compact machine-readable record is stored at
`cf_h2o/results/cluster/tsc_v97_lust_unseen_20260831/safety_candidate_v1/seed5057/early_rejection.json`.

## Claim boundary

This outcome says that the legacy LuST native signal plan is not an eligible
closed-loop confirmation target under the paper's strict baseline-safety
protocol when executed with SUMO 1.22. It does not measure CFCMT efficacy and
does not alter the positive v93 source-transfer result on the frozen
Los Angeles and Jinan evaluation targets.
