# TSC Collision-Monitoring Incident Record

## Finding

On 2026-08-08, a controlled A/B run exposed that the shared SUMO launch did not pass `--collision.check-junctions true`. The generated Salt Lake native action-catalogue program produced zero collision records under the old launch and five junction collision records in 600 s when junction checking was enabled. The route, seed, horizon, and signal program were otherwise unchanged.

## Scope

This defect affects the interpretation of every pre-r15 `collision_events = 0` result. Performance, queue, demand, teleport, and paired-seed measurements remain historical evidence under their recorded protocol, but zero junction collisions were not actually tested. Pre-r15 outputs and caches are retained for provenance and may not support a junction-safety claim.

## Correction

The canonical execution protocol is upgraded to `no-teleport-write-unfinished-junction-collision-audit-v2`, with collision action `warn` and explicit junction checking. Cache identities are invalidated. All source collection, rule selection, calibration, and evaluation needed for the submission will be rerun under the corrected protocol. Salt Lake admission and any 18-network extension remain blocked until a deployable safe controller passes the corrected detector.

## Second finding: benchmark collisions are not equivalent to controller failures

The corrected detector subsequently found junction collisions in the unmodified Cologne benchmark under its native `fixed_program`. For Cologne1, seed 2027, and a 600 s horizon, the evaluator observed six collision-event seconds corresponding to three deduplicated incidents. A separate SUMO 1.27.1 command-line replay reported seven raw collision records. Thus, a blanket zero-raw-collision requirement cannot be used as a common admission rule for these benchmark inputs: it rejects the dataset-provided controller as well as learned controllers.

The event was not eliminated consistently by reducing simulation step length. Otherwise matched native fixed-program runs reported 7, 3, 11, and 7 collision records at step lengths 1.0, 0.5, 0.2, and 0.1 s, respectively. The 0.5 s run also changed the trajectory distribution and produced 14 emergency stops and 41 emergency-braking events. Step-size reduction is therefore not adopted as a safety fix.

The original r15 gate is rejected before any r15 cache is admitted. It remains in the record below its original source-tree history rather than being silently reinterpreted.

## Benchmark-aware correction

The successor r16/v20 protocol keeps junction detection enabled and separates observation from attribution:

1. Starting or ending teleports are hard failures in source collection, rule rollout, calibration, and deployment evaluation.
2. Raw junction collision events and deduplicated incidents are always retained in the audit output. They are not automatically attributed to the focal signal intervention.
3. If any counterfactual branch collides, every candidate in that matched action group is excluded from the supervised dataset. Previously completed branches from that group are buffered and discarded as well. This defines a common safe-support action-contrast estimand and prevents action-dependent cherry-picking.
4. Repeated branch observations are deduplicated by a stable signature over scenario, seed, time, vehicle pair, collision type, and lane. Branch failures and physical incident signatures are reported separately.
5. The deployable primary policy must have no more paired unique collision incidents than `selected_source_prior`; raw benchmark collisions remain visible. Synthetic Salt Lake admission retains the stronger zero-collision requirement because its audited rule policies can satisfy it.

An exact local replay of the previously failing Manhattan seed 2027, shard 7/16 completed under the new collector. One non-focal physical incident at time 295 s affected four focal groups at the same snapshot; all four groups were symmetrically excluded, while eight groups and 72 branch rows were retained. The deterministic replay audit passed with zero numerical difference and no teleports. This is implementation validation, not a production SUMO 1.22 result; all production caches are still required to be rebuilt from a frozen source snapshot.
