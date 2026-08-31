# TSC v105: Toronto PTC independent unseen-city construction protocol

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This protocol is frozen before the Toronto raw resources are acquired, before
the city-wide SUMO network or demand is built, and before any Toronto safety,
source-selection, controller, or efficacy outcome is inspected. Toronto has not
been used to develop or select the v93/v98 causal source components, strict
target-only anchor, B100 offline selector, pressure guard, controller
parameters, or efficacy thresholds.

Toronto is a constructed independent target, not a prepackaged SUMO benchmark.
The construction uses official city-wide road and signal inventories and all
available 2025 Toronto-to-Toronto Private Transportation Company (PTC) trip
aggregates. It does not claim to represent all Toronto road traffic. PTC demand
is a complete retained subset of the published passenger-transport activity,
and ward/community-council endpoints remain spatially aggregated.

## Frozen sources

The reproducible importer reference is `Jahandad-Baloch/TorontoSUMONetworks` at
commit `7975f1aa01eeac62ba395e9ae66f59e5b0b1a5a9`. Its default single-ward,
arterial-only, sampled-demand settings are not used. The v105 package instead
uses the complete city extent and the union of all passenger-drivable feature
classes defined by that fixed commit.

The following City of Toronto Open Data resources are fixed by resource ID,
name, size, and source modification time. Acquisition must record the observed
SHA-256 of each exact payload before any parsing result is inspected.

| Role | Resource ID | Frozen resource | Size (bytes) | Last modified |
|---|---|---|---:|---|
| Road centreline | `7bc94ccf-7bcf-4a7d-88b1-bdfc8ec5aaf1` | `Centreline - Version 2 - 4326.geojson` | 93,264,749 | 2026-08-28 18:15:15 UTC |
| Ward polygons | `737b29e0-8329-4260-b6af-21555ab24f28` | `City Wards Data - 4326.geojson` | 1,148,140 | 2026-02-20 19:02:44 UTC |
| Community-council polygons | `cc935c56-dbcd-4035-b156-a7f8f8eae68b` | `Community Council Boundaries Data - 4326.geojson` | 749,891 | 2026-02-20 21:23:10 UTC |
| Signal locations | `e331c953-7e49-418b-9eab-594881c76f33` | `Traffic Signal - 4326.geojson` | 2,646,676 | 2026-08-29 00:09:26 UTC |
| Signal timing inventory | `02c90a3a-d754-4023-a283-ed5687e87f1f` | `Traffic Signal Timing` | 365,673 | 2026-08-30 13:25:05 UTC |
| PTC documentation | `6c7e183b-9df4-4179-b2d0-4929afd8e8d4` | `trips_and_summary_readme` | 374,725 | 2026-07-31 19:44:37 UTC |
| PTC daily cross-check | `c2f8c4c8-c120-480b-8713-89186488f5f5` | `summary_stats.csv` | 497,524 | 2026-08-01 04:35:59 UTC |
| PTC trip aggregates | `3aef7686-488c-4430-8bfe-1e4d44625fde` | `trips_2025.zip` | 55,410,662 | 2026-03-24 14:34:10 UTC |
| Midblock dictionary | `bed17b1a-a425-4130-a49c-67174dcf0e50` | `svc_data_dictionary.xlsx` | 56,579 | 2025-12-02 14:17:28 UTC |
| Midblock site summary | `af1ccce4-d978-4a6d-9dc3-3fb33b1cc349` | `svc_summary_data.csv` | 8,286,214 | 2026-08-30 04:36:36 UTC |
| Midblock speed observations | `25a7459e-b3ac-46f1-9b76-393be209b02c` | `svc_raw_data_speed_2025_2029.csv` | 116,677,560 | 2026-08-30 06:00:46 UTC |

The PTC technical documentation identifies three missing trip-data dates:
2025-05-26, 2025-05-27, and 2025-05-28. These absences are recorded in
advance. The expected observed-date set is therefore the other 362 dates in
calendar year 2025; any additional missing or unexpected date fails input
admission.

## Full-city network construction

The network retains the complete Toronto centreline extent and all fixed-commit
passenger-drivable feature codes: `201100`, `201200`, `201300`, `201301`,
`201400`, `201401`, `201500`, `201600`, `201601`, `201700`, `201800`, `201801`,
and `201803`. Trails and walkways that do not permit passenger vehicles are not
part of the road-control estimand. No ward, corridor, route, junction, signal,
or time window may be selected for convenience.

Direction, intersection identity, and centreline geometry come from the fixed
city source. Lane and free-flow attributes use the deterministic feature-code
mapping in the fixed importer reference; observed midblock speeds are not used
to calibrate them. Every official signal node that matches a retained road
junction is supplied to `netconvert`; generated SUMO programs establish the
uncontrolled native condition. The timing archive is retained for inventory
and paper provenance, not used to tune the controller. Unmatched source signals
and unmatched network junctions are reported before simulation and are not
repaired after a safety result.

The complete passenger-drivable network remains in the package. Demand
endpoints are restricted to non-internal edges in its largest strongly
connected passenger component so every origin-destination pair is routable.
This endpoint rule is fixed before topology inspection and does not remove
network edges.

## Seven full-day demand scenarios

Eligible source rows satisfy all of the following:

1. `dt` is in 2025 and is not one of the three documented missing dates.
2. `pickup_municipality` and `dropoff_municipality` both equal `Toronto`.
3. `pickup_hr` resolves to the same Toronto local date as `dt` and to one of
   the 24 local clock hours.
4. `trips_total` is a positive integer.

All eligible rows are retained. For each endpoint, a valid ward label is used
when present. If privacy aggregation replaces the ward with `Not included
elsewhere`, the published community-council label is used. If neither level is
resolvable against its frozen polygon source, input admission fails; the row is
not discarded.

Seven Monday-through-Sunday climatological full days are constructed from all
observed dates of the corresponding weekday, rather than selecting a favorable
date. Missing strata on an observed date count as zero. For weekday `d`, let
`n_d` be its number of observed dates and let `S_g` be the annual sum for
stratum `g = (local hour, origin level, origin name, destination level,
destination name)`. The initial integer count is `floor(S_g / n_d)`. The
rounded mean daily total is computed with decimal half-up rounding; remaining
vehicles are assigned by descending fractional remainder, with the stratum
tuple as the deterministic tie break. Thus each day exactly preserves the
rounded mean of all observed source trips for that weekday.

Within each hour, generated vehicles are ordered by stratum and ordinal and
placed uniformly at interval midpoints over the full 3,600 seconds. Candidate
origin and destination edges are sorted within each polygon and assigned by
deterministic round robin. Intra-zone trips use distinct endpoint offsets when
at least two eligible edges exist. `duarouter` must route every generated
vehicle with route errors fatal. No vehicle, OD pair, departure hour, or day
type may be sampled, scaled, repaired, or removed.

Fare, wait-time, reported distance, and duration fields are retained as passive
validation summaries but never used to tune SUMO vehicle or controller
parameters. Midblock 2025 speed observations are likewise reserved for passive
state/context validation; they do not calibrate edge speed, capacity, route
choice, or demand.

## Static and safety admission

Static admission requires:

1. exact source sizes and observed SHA-256 identities for every frozen payload;
2. exactly the documented 362 PTC dates and all seven weekday classes;
3. conservation from eligible source counts through climatology integerization
   and generated vehicle files;
4. complete 24-hour coverage for every day type;
5. valid ward/community-council resolution for every eligible row;
6. valid route-edge references for every generated vehicle;
7. a nonempty city-wide network, all fixed passenger-drivable feature classes
   observed in the source or explicitly reported absent, and at least one
   controllable traffic light; and
8. a frozen package manifest before microscopic execution.

Safety admission uses libsumo 1.22 only. Each of the seven native-signal day
types is run with its fixed seed `5100 + weekday_index`, where Monday has index
zero. Junction collision and teleport events are checked every simulation step.
Time-based teleportation is disabled, route errors are fatal, demand scale is
one, and the completion cap is 108,000 seconds, six hours after the final
departure window. Each day must load, depart, and complete its exact generated
vehicle count with zero collisions, zero teleports, zero pending vehicles, and
zero running vehicles at completion.

The seven day types form one joint gate. A failure in any source identity,
schema, date, geography, conservation, route, signal, collision, teleport,
insertion, or completion condition rejects Toronto before controller efficacy
evaluation. No network, endpoint, demand, vehicle, signal, simulator, seed, or
threshold amendment follows the first joint safety result.

## Frozen transfer evaluation

Only after joint admission may Toronto enter the unchanged causal transfer
ladder:

1. source-only zero-shot causal components;
2. B100 target-offline source selection with target-only fallback;
3. strict target-only CFCMT;
4. the selected causal source mixture;
5. dense/H2O+-style residual transfer;
6. simulator-only MPC;
7. fixed-time/native signal control; and
8. phase-pressure and MaxPressure-style rule baselines.

The B100 selector uses seed-blocked target-offline folds only. Closed-loop
confirmation uses disjoint seeds and all seven day types. No Toronto closed-loop
outcome may influence source inclusion, source weight, fallback, trust, guard,
or controller parameters. Selecting target-only is an accepted negative-transfer
decision and is reported as such.

No Toronto construction or result is yet available under this protocol.
