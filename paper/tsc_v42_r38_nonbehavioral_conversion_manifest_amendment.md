# TSC v42/r38 Non-behavioral Conversion-Manifest Amendment

## Scope

The first full-network conversion completed with process exit code `0`, but
its manifest recorded generated artifacts using the temporary atomic-write
directory `.full_networks.staging-42602`. The directory was then correctly
renamed to `full_networks`, so the SUMO files themselves were intact while the
absolute paths embedded in `conversion_manifest.json` became stale.

The scheduler also classified the short successful process as failed because
the converter did not emit a recognized completion marker. Its automatic
retry correctly refused to overwrite the existing immutable output root.

## Amendment

Before route-load admission or any external efficacy outcome was computed, the
converter was changed only to:

1. record generated files and generated-file arguments relative to the
   conversion root;
2. retain the external `netconvert` executable as an absolute runtime path;
3. emit `Results saved to:` after the output root has been atomically installed;
4. label the conversion-manifest schema as
   `cfcmt-libsignal-full-external-sumo-conversion-v2`.

No topology, coordinates, lanes, routes, flows, vehicle parameters, signal
programs, simulation horizon, random seed, target budget, selector, model, or
efficacy gate changed.

## Frozen provenance

- pre-amendment config SHA-256:
  `33ba97410d8ae9e1a5b3ce932dcfedfbbe78295f87e278a0204b6c330130e81c`
- pre-amendment converter SHA-256:
  `f4dbfcc878264de27320db86ef4f3c15f880c07fe0b117d228903f1875ee4e7c`
- amended converter SHA-256:
  `704f30e4813015800c765f88cd56c0451bd30d978db17acc95ae61020d8d9373`
- excluded first conversion root:
  `cf_h2o/results/cluster/tsc_v42r38_external_la_nanchang_20260809/conversion/full_networks`
- canonical corrected conversion root:
  `cf_h2o/results/cluster/tsc_v42r38_external_la_nanchang_20260809/conversion/full_networks_corrected`

The excluded root may be retained only as engineering evidence. It cannot be
used for v42 network admission, cache collection, or efficacy evaluation.
