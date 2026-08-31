# TSC v107: Chicago full-week for-hire unseen-city protocol

Date frozen: 2026-08-31 (Asia/Shanghai)

## Confirmatory claim boundary

This protocol is frozen after source-metadata and aggregate-coverage screening,
but before any row-level or grouped trip record is acquired, before an OSM
network is built, and before any SUMO safety, passive-validation,
source-selection, controller, or efficacy outcome is inspected. Chicago was not
used to develop the v93 causal source components, v98 offline source selector,
strict target-only anchor, pressure guard, controller parameters, or efficacy
thresholds.

The target is a full-week, full-network **public for-hire demand benchmark**. It
contains every publicly released Transportation Network Provider (TNP) and taxi
trip in the frozen interval whose two endpoints can be located from the fields
released by the City of Chicago. It is not a reconstruction of all private-car
traffic in Chicago. Consequently, any accepted result supports transfer across
a new metropolitan road network and a complete declared for-hire demand cohort;
it does not establish field effectiveness or performance under the city's total
traffic demand.

The historical segment-speed archive is held out from network and demand
construction. It is used only for passive external validation after the SUMO
package and safety outcome are frozen. It cannot calibrate demand, edge speeds,
signals, source weights, residuals, guards, or policies.

## Frozen public sources

The calendar interval is `[2023-08-21 00:00:00,
2023-08-28 00:00:00)`, local Chicago timestamps: five consecutive weekdays and
the following two weekend days.

- TNP trips: City of Chicago dataset `n26f-ihde`, *Transportation Network
  Providers - Trips (2023-2024)*, metadata `rowsUpdatedAt=1740694101`
  (`2025-02-27T22:08:21Z`).
- Taxi trips: City of Chicago dataset `e55j-2ewb`, *Taxi Trips - 2023*, metadata
  `rowsUpdatedAt=1707338412` (`2024-02-07T20:40:12Z`).
- Passive traffic speed: City of Chicago dataset `sxs8-h27x`, *Chicago Traffic
  Tracker - Historical Congestion Estimates by Segment - 2018-2023*, metadata
  `rowsUpdatedAt=1747348090` (`2025-05-15T22:28:10Z`).
- Spatial boundary: City of Chicago dataset `igwz-8jzy`, *Boundaries - Community
  Areas*, metadata `rowsUpdatedAt=1745363197` (`2025-04-22T23:06:37Z`), all 77
  community areas.
- Road topology: BBBike Chicago OpenStreetMap XML snapshot
  `Chicago.osm.gz`, `Last-Modified=Sun, 30 Aug 2026 03:39:55 GMT`,
  `Content-Length=231147462`, publisher MD5
  `1ae6e180e1c996c5def4cbfb03684808`. Acquisition additionally records the
  streamed SHA-256 and rejects a publisher-size or publisher-MD5 mismatch.

Source metadata are checked before and after acquisition. A changed Socrata
metadata identity, incomplete page, repeated/missing aggregate group, changed
OSM object, or incomplete community-area set rejects the acquisition.

## Coverage observed before freezing

Only aggregate counts, timestamp extrema, and field non-null counts were queried
to determine whether the sources could support the protocol. No grouped trip
rows, routes, simulator states, actions, rewards, or outcomes were inspected.

| Date | TNP rows | Taxi rows | speed rows | positive-speed rows | archived segments |
|---|---:|---:|---:|---:|---:|
| 2023-08-21 | 181,800 | 17,217 | 135,063 | 76,983 | 1,047 |
| 2023-08-22 | 189,822 | 17,870 | 148,674 | 87,830 | 1,047 |
| 2023-08-23 | 215,703 | 19,423 | 150,768 | 88,168 | 1,047 |
| 2023-08-24 | 240,691 | 20,508 | 129,828 | 76,506 | 1,047 |
| 2023-08-25 | 260,008 | 18,151 | 149,721 | 87,121 | 1,047 |
| 2023-08-26 | 290,310 | 13,107 | 148,674 | 78,350 | 1,047 |
| 2023-08-27 | 221,223 | 12,480 | 141,345 | 70,556 | 1,047 |
| **Total** | **1,599,557** | **118,756** | **1,004,073** | **565,514** | -- |

TNP and taxi timestamps span every declared day from the `00:00` privacy bin
through the `23:45` bin. Speed records span every day from approximately
`00:01`/`00:10` through `23:50`. Published `speed<=0` values are missing
observations, not zero-speed traffic.

## Complete-demand acquisition and inclusion rule

Acquisition transfers grouped records directly to shared cluster storage; no
raw trip table or raw speed table is stored on the local workstation. For each
source and day, every published row is represented exactly once in a group
defined by the released 15-minute start timestamp, pickup/drop-off community
areas, and pickup/drop-off centroid coordinates. Each group records its row
count and available trip-duration and trip-distance sums/counts. The sum of
group counts must equal the frozen Socrata daily total above for each source.

An endpoint is locatable if either:

1. its released finite centroid lies in one of the 77 published community-area
   polygons; or
2. its released community-area value is an integer from 1 through 77.

The centroid takes precedence; the community area is the deterministic fallback.
A trip enters microscopic demand exactly when both endpoints are locatable.
Every other trip remains in the manifest as `public_location_suppressed_or_outside`
and is excluded because no released city-network endpoint exists. Counts are
reported by day, source, and missing endpoint; the packager may not impute,
sample, scale, or silently drop them. Every included group count must be
expanded exactly once into microscopic vehicle demand.

## Frozen network and route construction

SUMO 1.22 `netconvert` consumes the complete fixed Chicago OSM XML using the
documented OSM-import options `--geometry.remove`, `--ramps.guess`,
`--junctions.join`, `--tls.guess-signals`, `--tls.discard-simple`, `--tls.join`,
and `--tls.default-type actuated`. No edge is removed after import. All generated
traffic-light systems and programs remain unchanged. The exact executable,
version, command, stdout/stderr, network counts, and input identities are stored
in the build manifest.

Demand endpoints use only non-internal passenger edges in the largest strongly
connected passenger-edge component. For every community area, 32 anchors are
selected before demand expansion by deterministic capacity-first farthest-point
sampling over eligible edge midpoints. A released centroid receives its eight
nearest same-area anchors. An area-only endpoint receives all 32 anchors.
Within each sorted aggregate group, vehicle ranks cycle deterministically over
feasible origin/destination anchor pairs; same-edge pairs are skipped. Departure
times are evenly stratified inside the released 15-minute bin. These operations
distribute privacy-coarsened demand but do not change its count or time bin.

All trips are routed with SUMO 1.22 `duarouter` under fatal route errors. Route
repair, ignored errors, fallback edges, demand scaling, route sampling, and
vehicle deletion are forbidden. Every used area must have 32 eligible anchors,
every included origin-destination group must have at least one feasible pair,
and routed vehicle IDs/counts must exactly match expanded input IDs/counts.
Failure of any condition rejects Chicago before microscopic execution.

## Seven-day strict safety admission

Each day is a separate full-network 24-hour condition with demand scale one,
one-second steps, and fixed seeds `5201` through `5207` in calendar order. The
strict headless configuration may set `max-depart-delay=-1`, junction collision
checks, `collision.action=warn`, `time-to-teleport=-1`, and a completion cap of
108,000 seconds. It may remove only passive GUI/output definitions. libsumo
1.22 is the only simulation interface.

Collision and teleport events are checked at every simulation step. Loaded,
departed, arrived, running, pending, expected, and final populations are counted
exactly. For **each of all seven days**, admission requires:

- exact included-demand loading and identity conservation;
- zero unique collision incidents, including junction collisions;
- zero teleport starts/ends;
- zero discarded departures and route errors;
- no remaining running, pending, or expected vehicles at completion; and
- at least one traffic-light system with unchanged generated programs.

One failed day rejects the Chicago candidate. No post-result network repair,
signal change, demand deletion, day deletion, seed replacement, safety-threshold
change, or rerun enters this confirmatory protocol.

## Passive validation and efficacy firewall

After the package and seven-day safety outcome are frozen, positive historical
speed observations are mapped once to nearest compatible passenger edges using
their published segment endpoints. Coverage, time-bin speed quantiles, and
simulation-versus-archive errors are reported for every day. Missing or
unmapped observations remain denominators in the coverage report. Passive speed
agreement is descriptive and cannot select or modify the target simulator or
method.

Only an admitted Chicago package enters the unchanged transfer ladder:

1. source-only single-source zero-shot components;
2. source-only multi-source zero-shot and a no-residual/no-source fallback;
3. strict B100 target-only CFCMT;
4. B100 target-offline source selection and selected causal mixture;
5. dense/H2O+-style residual and simulator-only MPC;
6. generated native signal control, phase pressure, and MaxPressure-style
   baselines; and
7. the preregistered source-weight/source-exclusion ablations.

Zero-shot uses the fixed topology and unlabeled demand/speed summaries permitted
by the existing information budget, but no Chicago next-state, reward, or
closed-loop label. B100 uses exactly 100 target-offline transitions in
seed-blocked folds. Closed-loop confirmation seeds are disjoint from all B100
rows. Target-only fallback is an accepted negative-transfer outcome. No Chicago
efficacy result is yet available under this protocol.
