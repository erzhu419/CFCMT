# TSC v42/r38 Pre-efficacy Nanchang Connection Correction

## Discovery

Before any external efficacy outcome was computed, source inspection showed
that the earlier Nanchang converter preserved edges and routes but did not use
two topology fields in the frozen CBEngine roadnet:

1. each lane's three permissible-movement flags;
2. each signalized intersection's four outgoing approach road IDs.

It instead allowed `netconvert` to infer turns and signal connectivity. That
does not preserve the benchmark's lane-level road network.

The official CBEngine roadnet specification defines each lane's three digits
as left, straight, and right permissions. It defines signal approaches as the
northern outgoing road followed clockwise by east, south, and west, with `-1`
for a missing leg:
https://cbengine-documentation.readthedocs.io/en/latest/content/cbengine/cbengine.html

## Frozen correction

The v4 converter now:

- reverses CBEngine inner-to-outer lane order into SUMO's lane numbering;
- creates one explicit connection per permitted lane movement;
- uses the frozen signal approach order at controlled intersections;
- geometrically classifies left, straight, and right only at uncontrolled
  intersections;
- excludes U-turns;
- fails conversion if any frozen flow route lacks a generated edge pair.

On the full Nanchang input this yields 13,390 unique lane-level connections.
All 186,350 consecutive transitions across all 9,786 flow definitions are
connected; the missing-pair count is zero.

## Execution correction

The attempted v3 job also exposed a tuple-unpack regression in the LA path and
failed before atomic output installation. A full-path conversion regression
test now runs both cities with a stubbed `netconvert`, checks the LA duplicate
inventory, checks the Nanchang connection inventory, verifies both SUMO config
files, and rejects staging paths.

## Provenance

- v3 config SHA-256:
  `2f52c617165545a4729a90cebaed5efd36a03e13e2cfd3e8fe00e7a5b3fb3fec`
- v3 converter SHA-256:
  `672b79e92d1b9c00a806387702144f6744c35f32982abc02e32158f9296b946b`
- v4 converter SHA-256:
  `4d35b9d775ea5c75586382abba1bbf5917063fbdf12092f4363f989fd4a13db8`
- canonical v4 candidate root:
  `cf_h2o/results/cluster/tsc_v42r38_external_la_nanchang_20260809/conversion/full_networks_v4`

No network admission, target cache, CFCMT fit, selector decision, or external
action-regret result was opened before this correction was frozen.
