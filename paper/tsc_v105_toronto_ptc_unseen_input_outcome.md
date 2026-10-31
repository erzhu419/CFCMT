# TSC v105: Toronto PTC independent-target input outcome

Date evaluated: 2026-08-31 (Asia/Shanghai)

## Decision

**REJECT before network construction.** Toronto does not advance to routing,
SUMO safety admission, source selection, or controller evaluation.

The frozen acquisition passed for all 11 resources (279,478,393 bytes) and the
fixed TorontoSUMONetworks commit. The corrected eight-worker input audit passed
seven of eight components. Its only remaining gate failure is nevertheless
decisive: 20 published Toronto-to-Toronto aggregate rows, representing 21
trips, cannot be resolved to either a ward or community-council polygon.

These rows report both municipalities as `Toronto` while the affected endpoint
has ward `Not included elsewhere` and community council `Not in Toronto`. The
pre-registered protocol requires every Toronto-to-Toronto row to resolve through
the ward-first, community-council-fallback rule. It explicitly forbids dropping
an unresolved row. Mapping these 21 trips to an arbitrary city-wide endpoint or
reclassifying them as external after seeing the data would be a post hoc source
repair, so neither is permitted.

## Passed evidence

- Exact PTC archive inventory: 12 monthly files, 3,203,093 rows, and the
  documented 362 dates; there are no additional missing or unexpected dates.
- Resolvable Toronto-to-Toronto demand: 2,053,498 aggregate rows and 70,946,123
  trips across all seven weekdays. The unresolved remainder is 20 rows and 21
  trips.
- PTC daily summary: the same 362 dates and 87,067,979 reported trips started.
- Network/geography inventory: 64,341 centreline features, 49,428 features in
  all 14 amended passenger-drivable classes, 25 wards, and four community
  councils.
- Signal inventory: 2,551 source signals; 2,313 match centreline intersections,
  222 lack a source `NODE_ID`, and 16 valid node IDs do not match. These are
  reported unmatched records allowed by the protocol, not silently repaired.
- Passive road evidence: 414,336 valid 2025 speed rows from 676 count sites over
  153 dates, representing 12,518,381 speed-bin vehicle observations.
- Signal timing inventory: 66,912 rows for 851 TCS identifiers.

The first audit attempt is retained separately. It incorrectly treated the
documented three missing summary dates as an error and did not handle source
signals with null `NODE_ID`; both audit assumptions were corrected without
changing any data, demand, network, safety, or efficacy threshold. The corrected
audit still fails on the pre-existing geography gate above.

## Claim boundary

This outcome is an input-admission rejection, not a negative-transfer or
controller result. It neither weakens nor strengthens the prior v93/v98 efficacy
evidence. Toronto contributes only a transparent data-quality exclusion to the
candidate-city flow diagram and cannot be used in a performance table.

No Toronto package, SUMO run, source weight, or controller result was produced.
