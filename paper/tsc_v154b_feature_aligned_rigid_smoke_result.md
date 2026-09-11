# V154B Four-City Feature-Aligned Rigid Smoke Correction Result

## Finding

Resolving the existing rigid model's inputs by feature name substantially
improved Cologne and RESCO synthetic in the retained smoke scenarios. Atlanta
and New York improved more modestly and still had substantial unmet demand.
The comparison used the original fitted model, scenario, seed and runtime
parameters in each city. No source correction was applied.

The corrected controller still has higher waiting and system load than the
retained PhasePressure in Cologne, New York and RESCO. Atlanta is lower on both
measures, but both controllers complete less than 7% of due demand. The repair
therefore establishes substantial gains in two fixed scenarios while leaving
city-specific service limitations and gaps to PhasePressure.

The three new runs completed: Atlanta `t90907`, New York `t90908` and RESCO
`t90909`. Cologne reuses the completed correction `t90756`. All four use source
snapshot `589fd20266b7265b6f2f`, seed `41242`, 3600 seconds, 60-second warmup,
ten-second control and a 450-second prediction horizon. The original rigid and
PhasePressure results are retained comparators; neither was rerun.

## Waiting And Whole-System Service

Waiting below is `mean_tripinfo_waiting_time`: the mean over **all departed
vehicles, including unfinished tripinfo records at the horizon**. It excludes
vehicles still pending insertion. Arrival completion is therefore reported
against due demand separately. System vehicle-hours include active inserted
vehicles and vehicles pending insertion after warmup.

| City | Original rigid waiting (s) | Corrected rigid (s) | Original PhasePressure (s) | Corrected vs original rigid |
|---|---:|---:|---:|---:|
| Atlanta | 2248.066 | 2194.333 | 2276.759 | −2.39% |
| Cologne | 1484.364 | 19.713 | 14.991 | −98.67% |
| New York | 1331.852 | 1281.185 | 1085.870 | −3.80% |
| RESCO synthetic | 122.289 | 59.018 | 31.166 | −51.74% |

Due demand is the native horizon count of cumulative departed vehicles plus
vehicles pending insertion. It is unchanged across the three arms in each
city.

| City / controller | Arrived / due demand | Arrival completion | Pending insertion | System vehicle-hours |
|---|---:|---:|---:|---:|
| Atlanta / original rigid | 140 / 2171 | 6.45% | 1747 | 1762.282 |
| Atlanta / corrected rigid | 150 / 2171 | 6.91% | 1729 | 1750.530 |
| Atlanta / original PhasePressure | 86 / 2171 | 3.96% | 1914 | 1809.041 |
| Cologne / original rigid | 210 / 2015 | 10.42% | 1606 | 886.761 |
| Cologne / corrected rigid | 1997 / 2015 | 99.11% | 0 | 41.080 |
| Cologne / original PhasePressure | 1999 / 2015 | 99.21% | 0 | 34.028 |
| New York / original rigid | 2790 / 15841 | 17.61% | 4385 | 6396.234 |
| New York / corrected rigid | 3096 / 15841 | 19.54% | 4244 | 6265.708 |
| New York / original PhasePressure | 3676 / 15841 | 23.21% | 3309 | 5778.219 |
| RESCO / original rigid | 1445 / 1473 | 98.10% | 0 | 106.914 |
| RESCO / corrected rigid | 1442 / 1473 | 97.90% | 0 | 80.482 |
| RESCO / original PhasePressure | 1457 / 1473 | 98.91% | 0 | 67.737 |

Cologne's original service collapse disappears: pending insertion falls from
1606 to zero, and system vehicle-hours fall by 95.37%. RESCO shows a clear
fixed-scenario improvement: waiting falls by 51.74% and system vehicle-hours
by 24.72%, with nearly unchanged completion. The three fewer arrivals are
0.20% of due demand, a small tradeoff relative to the waiting and load gains.

Atlanta's system vehicle-hours fall only 0.67%, and New York's only 2.04%.
Both retain large pending populations; their preserved PhasePressure runs
also serve only a limited fraction of due demand. These outcomes show that
repairing the column error alone does not resolve their service limitations.

The completed-only mean is a different metric and population. In Atlanta it
increases from `11.507` to `47.753` seconds while arrivals increase from 140
to 150. The all-departed waiting means use 424 and 442 vehicles respectively,
including 284 and 292 unfinished records. This illustrates why the waiting
mean alone cannot describe whole-network service or a fixed vehicle cohort.
All completed-only means and population counts are retained in the summary
artifact.

## Collision Reporting

Every comparison below uses the native `collision_incidents` field. Raw
`collision_events` are shown separately and are not converted by dividing by
a fixed factor. The execution and counting protocol is identical across the
four cities and their retained comparators. All starting and ending teleport
counts are zero.

| City | Original rigid incidents | Corrected rigid incidents | Original PhasePressure incidents | Raw events: original / corrected / PhasePressure |
|---|---:|---:|---:|---|
| Atlanta | 0 | 0 | 0 | 0 / 0 / 0 |
| Cologne | 0 | 18 | 20 | 0 / 36 / 40 |
| New York | 0 | 0 | 0 | 0 / 0 / 0 |
| RESCO synthetic | 0 | 0 | 2 | 0 / 0 / 2 |

## Scope And Limitations

This is a correction on four existing scenarios with one seed per city.
It measures the effect of reading the original model's intended input columns;
model parameters, occupancy formulas, signal execution and collision handling
remain unchanged. All four corrected runs have zero source-induced action
changes and zero utility-gate selections. The unrecomputed V150K utility
payload was not used to add source, and no source-admission threshold changed.
Cologne retains 18 collision incidents, so this correction does not establish
safe-controller admission. Its collision diagnosis and the separate feature
unit repair are outside this batch.

## Execution And Artifacts

The first scheduler records `t90904`--`t90906` failed before simulation because
the inline Python command exceeded the SSH argument limit after shell
quoting. Their `started_at` values remained null and no result directories
were created. The exact same frozen runtime code was moved to small server
scripts and submitted as `t90907`--`t90909`; the original launch and the
transport-correction mapping are both retained. Scientific protocol V154B v1
was unchanged.

The new runs took 17.38 seconds (Atlanta), 736.80 seconds (New York) and
43.95 seconds (RESCO) with one CPU and 8192 MB RAM requested per task. Retrieval
contained only three result JSON files, totalling 1,628,510 bytes. The existing
Cologne result was reused locally; no model or checkpoint was downloaded.
The comparison script verified original model and manifest identities,
runtime arguments, metric populations and collision-reporting protocols before
writing the four-city artifact.

- Summary: `cf_h2o/results/paper_artifacts/tsc_v154b_feature_aligned_rigid_smoke_v1.json`
- Frozen protocol: `paper/tsc_v154b_feature_aligned_rigid_smoke_protocol.md`
- Frozen configuration: `cf_h2o/config/traffic_signal_tsc_v154b_feature_aligned_rigid_smoke.json`
- New results: `cf_h2o/results/cluster/tsc_v154b_feature_aligned_rigid_smoke_20260909/rigid_v1/`
- Successful launch: `cf_h2o/results/cluster/tsc_v154b_feature_aligned_rigid_smoke_20260909/launch_transport_v2.json`
- Original launch record: `cf_h2o/results/cluster/tsc_v154b_feature_aligned_rigid_smoke_20260909/launch_v1.json`
- Reproducible comparison: `cf_h2o/results/cluster/tsc_v154b_feature_aligned_rigid_smoke_20260909/aggregate_v1.py`
- Reused Cologne correction: `cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/rigid_feature_alignment_v1/result.json`
