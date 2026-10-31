# TSC v105: Toronto pre-build centreline schema amendment

Date frozen: 2026-08-31 (Asia/Shanghai)

## Timing and scope

This amendment is frozen after source-identity acquisition and the first
schema-only inspection, but before network conversion, demand aggregation,
routing, SUMO execution, source selection, or controller evaluation. No safety
or efficacy outcome exists. The original v105 protocol remains unchanged as an
audit record; this document supersedes only its centreline feature-code list
and lane/free-flow mapping.

## Observed incompatibility

The fixed `TorontoSUMONetworks` importer and the frozen 2026 centreline payload
do not use the same feature-code semantics. In particular:

- the importer labels `201800` as Expressway Ramp, while the frozen City source
  labels all 646 such records `Pending`;
- the City source identifies `201101` as Expressway Ramp (1,134 records), but
  the importer does not include that code;
- the importer labels `201600` as Major Arterial Ramp, while the City source
  labels 2,102 such records `Other`; and
- the City source identifies `201201` as Major Arterial Ramp (171 records), but
  the importer does not include that code.

Using the stale mapping would misstate road semantics and would omit current
expressway and major-arterial ramps. This is a source-schema defect in the
reference importer, not a simulator or controller result.

## Frozen corrected mapping

The build uses the current official `FEATURE_CODE_DESC` as the authoritative
class identity. The complete passenger-drivable set and deterministic SUMO
attributes are frozen as follows:

| Code | City class | Lanes | Speed (m/s) | Direction default |
|---:|---|---:|---:|---|
| 201100 | Expressway | 4 | 27.78 | source direction |
| 201101 | Expressway Ramp | 1 | 25.00 | source direction |
| 201200 | Major Arterial | 3 | 12.50 | source direction |
| 201201 | Major Arterial Ramp | 1 | 12.50 | source direction |
| 201300 | Minor Arterial | 2 | 12.50 | source direction |
| 201301 | Minor Arterial Ramp | 1 | 12.50 | source direction |
| 201400 | Collector | 2 | 12.50 | source direction |
| 201401 | Collector Ramp | 1 | 11.11 | source direction |
| 201500 | Local | 2 | 12.50 | source direction |
| 201600 | Other | 1 | 11.11 | source direction |
| 201601 | Other Ramp | 1 | 12.50 | source direction |
| 201700 | Laneway | 1 | 4.17 | source direction |
| 201801 | Busway | 1 | 11.11 | source direction |
| 201803 | Access Road | 1 | 5.56 | source direction |

All classes permit passenger vehicles. Busway additionally preserves bus use;
laneway and access-road permissions follow the reference importer. Code
`201800` is excluded because the source marks it `Pending`, not as a defined
passenger-road class. Railway, water, trail, walkway, hydro, shoreline,
geostatistical, and ferry features remain outside the road-control estimand.

The frozen source has exactly one `201301` segment and six `201601` segments;
rarity is not a reason to remove them. All 14 corrected classes must be retained
in the pre-conversion inventory, and their total retained feature count must be
49,428. Any future mismatch rejects the build rather than triggering another
mapping change.

All other v105 source, demand, routing, safety, transfer, and no-post-result-
amendment conditions remain in force.
