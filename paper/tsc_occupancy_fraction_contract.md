# SUMO Lane-Occupancy Fraction Equations v1

## Unit Contract

`sumo-lane-occupancy-fraction-equations-v1` uses occupied vehicle length divided
by lane length for all lane occupancy inputs and predicted occupancy outputs.
For example, the retained Cologne observation `0.28158186943028773` represents
28.158 percent occupancy. SUMO 1.22 libsumo and TraCI expose the same fraction:
the [lane implementation](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/libsumo/Lane.cpp)
returns `MSLane::getNettoOccupancy`, which divides occupied length by lane length.

The pressure receiving factor is `max(0, 1 - occupancy_fraction / 1.2)`.
The spillback-pressure baseline retains its 0.20 floor. These equations equal
the corresponding percentage equations with divisor 120. At 80 percent
occupancy, both receiving factors are one third.

The analytic prior uses `1 - downstream_fraction / 1.35` for capacity.
Its predicted downstream occupancy is
`clip(downstream_fraction + 0.0045*service - 0.001*mean_speed - 0.025*max(1-mean_fraction, 0), 0, 1)`.
The downstream contribution to the speed equation is `-1.2*downstream_fraction`,
and the predicted occupancy contribution to cost is `+4.0*next_downstream_fraction`.
All four changes follow from converting the original percentage equation's
inputs and occupancy output together. Queue and speed units retain their
existing definitions. The empirical queue proxy remains
`halting + 0.35*vehicle_count + 0.025*occupancy_fraction`.

## Cache And Model Boundary

The retained v19/v20 caches contain the old pressure features and analytic
priors under the same feature names. Loading them by their own stored identity
can pass existing schema checks. A formula change alone would therefore mix
old training features with new deployment features.

The correction uses cache v21 for one-step mechanisms and v22 for multihorizon
prefix costs, records the occupancy-equation
protocol in cache identity and dataset metadata, and validates it on cache
load and dataset merge. V150K runtime payloads and V150L deployment inputs carry
the same protocol; the V150K runtime-model protocol advances to v2. Source-rule
rollout cache identity also carries the equation protocol so the corrected
spillback baseline cannot return a previous cached rollout. Old caches and
runtime payloads are rejected by the new
operational path. Frozen snapshots and their numerical results remain the
reproduction path for previous experiments.

The affected products are pressure feature columns, their action-reference
contrasts, analytic-prior arrays, fitted rankers, target/source utility models
and their runtime payloads. Retained raw SUMO observations and measured
occupancy targets already use fractions. The existing aggregate feature bank
does not retain every individual movement's queue and occupancy, so it cannot
reconstruct movement pressure exactly by updating a few aggregate columns.
The new operational bank therefore requires fresh collection and model fitting.

## Bounded Operational Pilot

The pilot collected fresh Cologne counterfactual labels using a 30-second
control interval and a one-interval, 30-second prediction horizon. It wrote
and reloaded the new cache, fitted a new action ranker, and checked prediction
under the same unit contract.

The fixed collection settings were duration 300 seconds, warmup 120 seconds,
one focal TLS and seed 41242.

```bash
python3 -m cf_h2o.eval.traffic_signal_occupancy_fraction_pilot \
  --sumocfg /server/path/to/cologne1.sumocfg \
  --cache-root /server/new_occupancy_pilot/cache \
  --out /server/new_occupancy_pilot/result.json
```

## Observed Result

Task `t90917`, using frozen snapshot `484f9fe12a951597a2bb`, returned `PASS`
for `operational_cache_and_ranker_check_only`. It retained 48 newly labelled
rows: six action groups with eight candidates each. Actual sample times were
25320, 25350, 25380, 25410, 25440 and 25470 seconds.

The v21 cache round trip preserved the arrays and metadata exactly. The maximum
absolute difference between stored service pressure and the fraction equation
was zero. The newly fitted `HistGradientBoostingRegressor` used a nonconstant
predictor, and its predictions were identical after model serialization and
reload. Reported runtime was `2.1428121489007026` seconds.

The related local regression checks passed 107 tests. These included full
percentage/fraction equation equivalence, production pressure and spillback
consumers, rejection of a previous cache even when validated against its own
identity, mixed-contract merge rejection, source-rule cache separation, and
the actual V150L constructor's rejection of previous runtime payloads.

The retrieved result is
`cf_h2o/results/cluster/tsc_occupancy_fraction_pilot_20260909/fresh_v1/result.json`
and contains 6,869 bytes. The bank and fitted model remain on the server under
`/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_occupancy_fraction_pilot_20260909/fresh_v1/`:
`cache/cologne1__e077c27abc7520df8f69.npz` and `fresh_ranker.pkl`. Neither was
downloaded to the local workspace.

## Limitations

This is an execution and unit-contract result. The maximum observed downstream
occupancy was `0.06380239098291836`, or about 6.38 percent. The pilot therefore
does not validate congested receiving-lane or spillback behavior; higher
occupancy was covered by equation regression tests rather than this SUMO run.

Predictions were checked on the fresh training bank. Their finiteness,
nonconstant fitted estimator and exact persistence do not establish action
quality or generalization. The pilot did not test B100 source selection, a
complete V150K utility payload, closed-loop improvement or source efficacy.
The complete V150K utility requires its source-city inventory and nested fits;
only its old-payload rejection was checked locally through V150L. Reopening
the full V150K/L experiment requires a separate result protocol and scheduler
signature for the new bank/model contract.
