# V107 Chicago full-week safety admission outcome

## Status

The Chicago route package passed the route-construction contract, but the
strict seven-day libsumo network admission failed. Chicago is not admitted as
an unseen confirmatory target and no policy result may be reported for it.

## Route-package result

The immutable v3 package contains all 1,422,405 requested vehicles over
2023-08-21 through 2023-08-27. Per-day demand counts are exact, no route was
dropped or repaired after construction, and the maximum departure-time
perturbation is 0.005455 seconds, below the frozen 0.006-second limit.

## Full-week admission result

Each day was launched once with one libsumo process on a separate CPU node
(day 6 followed day 5 on node006). Every day encountered a junction collision
early in the run, between simulation seconds 501 and 1939. The strict monitor
terminated immediately, as required. Teleports remained zero before each
termination, but exact full-day demand completion and controllable-TLS checks
were necessarily incomplete.

This is a network/demand compatibility failure, not an infrastructure failure
and not a policy comparison. Re-running the unchanged package cannot change
the admission decision. The next admissible step is to repair and independently
validate the shared network/conflict model, then rebuild a new versioned route
package and repeat all seven days from the start.

## Evidence

- Route manifest: `cf_h2o/results/cluster/tsc_v107_chicago_unseen_20260831/route_v3_manifest.json`
- Snapshot manifest: `cf_h2o/results/cluster/tsc_v107_chicago_unseen_20260831/admission_snapshot_v2.json`
- Admission launch: `cf_h2o/results/cluster/tsc_v107_chicago_unseen_20260831/admission_v1/launch_manifest.json`
