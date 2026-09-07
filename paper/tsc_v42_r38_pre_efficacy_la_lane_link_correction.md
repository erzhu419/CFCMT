# TSC v42/r38 Pre-efficacy LA Lane-link Correction

## Discovery

After the relocation-safe conversion completed, but before network admission,
counterfactual collection, or external efficacy evaluation, its `netconvert`
audit reported:

`Unused state in tlLogic '269390046', program '0' at tl-index 31`.

Inspection of the frozen CityFlow input showed 708 lane-link records but only
638 unique SUMO connection identities. Seventy records are byte-equivalent in
their endpoint lane indices and point geometry. Seven of these duplicates are
at controlled intersection `269390046`: the input contains 198 controlled
lane-link records but 191 unique controlled connections.

The v2 converter emitted every record and assigned all records separate TLS
indices. SUMO dropped duplicate connection identities during `netconvert`, so
the generated TLS state dimension could diverge from the realized controlled
links. Consequently the v2 conversion is not admissible.

## Frozen correction

The v3 converter now keys each LA connection by source road, destination road,
source lane, and destination lane. An exact duplicate maps back to the existing
connection/TLS index. If duplicate endpoint identities have different point
geometry, conversion fails instead of silently merging them. The manifest
records source, unique, and duplicate counts.

Nanchang conversion, routes, demand, coordinates, seeds, target budget,
selector, model, and efficacy gates are unchanged. External efficacy remained
sealed throughout diagnosis and correction.

## Provenance and exclusions

- v2 config SHA-256:
  `6dcd1107566e3a29fff3990901967e8dd84b68a50c8b207a84eb51ec6d2897ad`
- v2 converter SHA-256:
  `704f30e4813015800c765f88cd56c0451bd30d978db17acc95ae61020d8d9373`
- v3 converter SHA-256:
  `672b79e92d1b9c00a806387702144f6744c35f32982abc02e32158f9296b946b`
- excluded v2 conversion root:
  `cf_h2o/results/cluster/tsc_v42r38_external_la_nanchang_20260809/conversion/full_networks_corrected`
- canonical v3 candidate root:
  `cf_h2o/results/cluster/tsc_v42r38_external_la_nanchang_20260809/conversion/full_networks_v3`

Both earlier roots are engineering evidence only and are prohibited from
network admission, target-cache construction, and efficacy evaluation.
