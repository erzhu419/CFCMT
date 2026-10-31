# V154E Cologne bilateral geometry: original-duration validation

Frozen before the full-duration bilateral run. The user authorized continuing
with the unchanged V154D geometry for the original 3600-second duration to
evaluate full-run collisions and traffic service. V154D's short-window FAIL
remains a separate result.

## Fixed package and controller

Reuse the existing V154D server-side package, including its original demand
reference. Do not rebuild or shift either waiting point. The internal lengths
remain 13/24 = 8.48/20.05 m and 3/20 = 8.37/19.83 m. Its net-file semantic
delta against the original must equal the retained 12-attribute report before
submission; a discrepancy requires correcting the supplied path or package
reference before execution.

Package configuration:
`/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_v154d_cologne_bilateral_waiting_geometry_20260909/package_v1/cologne1.sumocfg`.

Use SUMO 1.22.0 and controller source snapshot `85b2b43cd12832c5a1dd`.
The original rigid model, stored-feature-name binding, original occupancy
equations, seed 41242, 60-second warmup, 10-second control interval,
450-second prediction horizon, direct execution and zero cooldown remain
unchanged. Use native collision warnings and the existing no-teleport
execution protocol, phase clearance and vehicle parameters. The standalone
validation tool receives a separate immutable snapshot; its recorded
controller module imports must resolve to the old source.

Tooling snapshot: `95293dee78371b53157d`. All 16 focused validation tests
passed before staging. They cover exact original metrics, uncapped inventory
and incident grouping, fixed passage/horizon conditions, the retained V154D
prefix and explicit unavailable collision participant poses.

Both arms begin at 25200 and end at **28800**, exactly **3600 seconds**.
This is the original full-run horizon, not an extension of the V154D artifact.
One task requests one CPU and 8192 MB and runs the arms sequentially.

## Original-arm prerequisite and observation completeness

Run the original network with the new read-only recorder. Require exact
equality of the complete deterministic metrics dictionary against corrected
original run `t90756`, including its 175 retained intervention records,
20 retained collision reports, traffic metrics and execution audits. The
deterministic originator diagnostics must also match. The reference's
top-level host, paths and wall-clock timing are not scientific
comparison values. Require all 3600 one-second steps and zero teleports.
A failed prerequisite stops before the bilateral simulation.

The existing evaluator counts every native collision but retains only its
first 20 debug reports. The new observer additionally reads native collisions
immediately after each simulation step, before executor advancement. Keep
every event without a sample cap. Use the evaluator's existing incident-key
definition to group reports, and require the new inventory's event, distinct
incident and collision-step totals to agree with the evaluator's native
totals. Preserve collision-time vehicle IDs, lanes, speeds, positions and
available actual passenger polygons, with missing poses explicitly recorded.
Capture static lane/connection context once. Do not collect a full vehicle
trace or download tripinfo XML.

The original evaluator's retained debug rows are taken after executor
advancement. The extra observer's collision-time state is taken before it;
these stage labels remain explicit. Baseline reproduction compares the
evaluator's own unchanged retained rows to its reference.

## Bilateral evaluation

Run the unchanged V154D package for the same 3600 seconds. Within this run,
compare the retained early action/observation/passage prefixes against V154D
to verify that the longer horizon reproduces its prior trajectory. This
comparison reuses the 108 original detailed samples in 25630–25665 and the
23 early intervention records; it does not trigger another short simulation.

Overall full-run PASS requires:

- Successful completion of all 3600 steps and consistent complete collision
  inventory totals.
- Exact reproduction of the retained V154D early trajectory.
- Zero native collision events/incidents and zero teleports across the entire
  3600 seconds.
- Actual incoming → prescribed internal lane(s) → outgoing passage of all
  four previously specified focal vehicles by 28800, on their original routes:
  `102219_396_0`, `129962_409_0`, `121463_406_0`, `168358_425_0`.

The full-duration conditions are frozen independently. A completed focal
passage after 25666 contributes to this evaluation and does not retroactively
change V154D's unmet short-window criterion. Preserve any remaining collision
and the overall FAIL without adjusting geometry, physics or thresholds.

## Traffic comparison and next decision

Report arrivals, departures, due demand, remaining active vehicles, pending
insertion, mean tripinfo waiting, system vehicle hours, queue and execution
metrics for original and bilateral arms. Waiting includes departed vehicles
with unfinished tripinfo at the horizon and excludes pending vehicles. System
vehicle hours include active plus pending vehicles after warmup; this code's
one-second accounting has 3541 post-warmup samples.

Use the retained original-network PhasePressure result as descriptive context
only: it is not a repaired-network paired comparator. There is no newly tuned
service acceptance threshold. Interpret service changes alongside collision
outcomes; a safety failure remains a failure even if some service metrics
improve, and reduced throughput is not automatically evidence of gridlock.

Classify the uncapped incident inventory by actual movement pairs. Distinguish
the two repaired waiting pairs from the previously observed moving 16/22 pair
and any other actual collisions. Remaining incidents determine whether the
next useful action is another localized mechanism diagnosis. Choose no new
geometry change within this run.

## Artifacts and limitations

Protocol: `tsc-v154e-cologne-bilateral-full-validation-v1`.
Artifact directory: `tsc_v154e_cologne_bilateral_full_validation_20260909`.
Check existing task signatures before submitting one paired task. Keep the
network, demand, model and scratch tripinfo on the server; retrieve the compact
result JSON and use a small log tail for execution failures. Retain complete
event/incident summaries, comparison outcomes and reference paths rather than
duplicating the already retained short-window raw trace.

This is one seed and one original-duration scenario. It establishes the
observed full-run geometry comparison and its traffic tradeoffs. It does not
establish multi-seed robustness, superiority over PhasePressure on a repaired
network, a benefit from the newer occupancy equations or source-transfer
efficacy.
