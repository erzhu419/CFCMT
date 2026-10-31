# TSC v118r: ETH Boston destination-TAZ schema remediation

## Trigger

The frozen V118 Boston operational preflight stopped at simulation time
7,696 s with `FatalTraCIError: Vehicle '54747' has no valid route.` No Boston
controller outcome, passive adaptation trajectory, source choice, or efficacy
metric had been observed.

The published trip carries `fromTaz="981100"` and
`trip_toTaz="120400"`. SUMO 1.22 recognizes `fromTaz` and `toTaz`; it does not
interpret `trip_toTaz` as the trip destination zone. Consequently, it attempted
to route to the published edge anchor `-8647106`, which has no microscopic
incoming connection. A diagnostic duarouter probe on this first failing trip
failed with the published custom field and succeeded when the same value was
written as `toTaz="120400"`.

## Frozen remediation

Before any repaired simulation, V118r authorizes one uniform transformation:
stream every published trip through an XML parser and rename `trip_toTaz` to
the canonical SUMO attribute `toTaz`. Trip ID, departure, origin and destination
edge anchors, origin TAZ, destination TAZ value, and all other attributes remain
unchanged. No trip may be removed, no edge may be substituted, route repair
remains disabled, and `ignore-route-errors` remains false.

This is schema canonicalization of the dataset's declared TAZ semantics, not a
route repair selected from controller outcomes. The same operation is applied
to all 3,806,510 trips before the next run. Passive tripinfo, vehroute, summary,
and edge-output declarations inherited from the mesoscopic configuration are
removed from the derived package; the admission runner writes only its compact
summary output.

## Required validation

The rebuilt package must prove equal input/output trip and unique-ID counts,
all custom destination fields converted, all canonical destination fields
present, zero changed edge anchors, and zero removed trips. The complete demand
must then pass duarouter with TAZ routing and errors enabled before microscopic
admission is retried.

The original V118 seeds, arms, information budgets, primary metric, and joint
confirmation gates remain unchanged. The failed v1 package and preflight are
retained as provenance rather than overwritten.
