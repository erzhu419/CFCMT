# V154F Cologne: fixed-network rigid versus PhasePressure

Frozen before the new comparison runs. Keep the successful V154E network
geometry and compare the original rigid controller with PhasePressure on the
same scenario, demand, simulation settings and prespecified seeds.

## Roster and reuse

Use the existing V150L full-run seed roster, in its original order:
**52282, 41242, 63792**. Its source is
`cf_h2o/config/traffic_signal_tsc_v150l_state_conditioned_source_closed_loop.json`,
`full.seeds`. Do not choose or replace seeds based on this comparison.

| Seed | Rigid target-only | PhasePressure |
|---|---|---|
| 52282 | New run | New run |
| 41242 | Reuse V154E `t90988`, repair arm | New run |
| 63792 | New run | New run |

The matrix has six cells and **five new simulations**. Reuse the complete,
validated V154E rigid/41242 metrics and collision inventory directly, with
its source artifact identified in the aggregate. Do not rerun that cell or
the original-network baseline merely to reproduce a settled result.

Seed 41242 was used to diagnose and construct the geometry and is reported
as the development seed. Seeds 52282 and 63792 provide additional simulations
of the frozen geometry. This does not label them as independent training
holdouts or untouched-city evidence.

## Fixed environment and fair controller inputs

Use the existing V154D package configuration:
`/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_v154d_cologne_bilateral_waiting_geometry_20260909/package_v1/cologne1.sumocfg`.

The waiting splits remain 13/24 = 8.48/20.05 m and 3/20 = 8.37/19.83 m. Each
new run verifies its parsed static network against V154E and uses that same
configuration and original 2015-trip demand. No geometry, routes or vehicle
parameters are changed.

Both controllers use SUMO 1.22.0 and controller source snapshot
`85b2b43cd12832c5a1dd`, with the same native collision warnings and no-teleport
execution, phase clearance, 60-second warmup and 10-second control interval.
All cells run from **25200 to 28800**, exactly **3600 seconds**.

Rigid uses the original fitted model and stored-feature-name binding, a
450-second prediction horizon, direct execution and zero cooldown. Its model
is fixed across evaluation seeds. PhasePressure uses the original evaluator's
`policy="phase_pressure", models=None` path. It has no prediction model or
training requirement. The original occupancy equations remain in use for both
arms; the newer unit corrections are not mixed into this comparison.

## Recording and per-cell outcomes

Reuse V154E's read-only, uncapped native collision recorder and original
incident-key definition. Record collision participant poses and movements,
explicit unavailable poses, every event and every distinct incident. Do not
collect the old 108-sample diagnostic window again or a full vehicle trace.

A valid cell must complete all 3600 one-second steps with successful evaluator
output, have matching recorder/native event, incident and collision-step
totals, zero teleports, and preserve the population identities:

- Due demand = departed + pending insertion = 2015.
- Departed = arrived + active vehicles at the endpoint.

Report the zero-collision diagnostic separately. A completed cell with native
collisions remains in every comparison and retains its safety FAIL. A runtime
or population-accounting failure is reported as invalid and prevents claiming
a complete paired matrix. Do not drop failing seeds or substitute new ones.

Track the four earlier focal vehicles as auxiliary passage evidence. Read
their routes from the actual demand and accept every legal internal path for
the corresponding from/to movement. In particular, the straight vehicles may
use lanes 1_0 or 1_1 and 11_0 or 11_1, respectively. Waiting/continuation paths
remain 13_0→24_0 and 3_0→20_0. Different legal lane choices must not be
reported as failed passage. Focal timing is diagnostic in this controller
matrix, not a new service acceptance threshold.

## Comparison and interpretation

Keep three conclusions distinct: matrix completeness, each controller's
zero-collision outcome across the roster, and service differences. Report
native incidents and zero-collision seeds for each arm, as well as paired
incident differences. Overall zero-collision robustness over this roster
requires zero incidents in all three seeds for the relevant controller.

The existing primary service metric is mean tripinfo waiting. Report all
three paired differences as **rigid minus PhasePressure**, then their equally
weighted mean. Also report the equally weighted arm means and their relative
difference. System vehicle hours are the companion service metric, with
arrivals, active vehicles, pending insertion and departure delay shown to
explain any tradeoff. Waiting includes departed but unfinished tripinfo and
excludes pending insertion; system hours include active and pending vehicles
under the original 3541-sample post-warmup accounting.

Use all three seeds in the main comparison. Also show the prespecified
two-additional-seed summary (52282 and 63792), with 41242 identified as the
geometry-development result. Do not change the main roster after seeing
outcomes. This small comparison reports observed paired effects; it does not
introduce a new significance test, tuned performance threshold, source gate
or claim of universal controller superiority.

## Execution and artifacts

Protocol: `tsc-v154f-cologne-repaired-controller-comparison-v1`.
Artifact directory: `tsc_v154f_cologne_repaired_controller_comparison_20260909`.
Check existing task signatures and reuse records before submitting five
single-cell tasks. Each task requests one CPU and 8192 MB. The standalone
runner has an immutable tooling snapshot while importing the frozen old
controller source. Retain task IDs, parameters, per-cell results and the
six-cell aggregation with explicit reuse provenance.

The pre-run tooling snapshot is `8c7842518a4d53225905`. Seventeen focused
runner tests and five aggregation tests passed before submission. They cover
the fixed roster and reuse exclusion, legal passage alternatives, controller
inputs, complete population/event accounting, and retention of colliding
valid cells in all prespecified service summaries.

Keep network, demand, model and scratch tripinfo on the server. Retrieve only
small result JSON files, or log tails for execution errors. Report five new
simulation costs separately from the reused V154E result. Preserve V154D's
short-window FAIL and V154E's one-seed full-window PASS.

## Limitations

This is one city, one demand file, three fixed simulator seeds and one fitted
rigid model. It tests the frozen repaired geometry across these trajectories
and compares the two controllers on the same network. It does not establish
other-city robustness, repeated-training-seed performance, the benefit of
new occupancy formulas or source-transfer efficacy. If collisions recur,
retain their complete evidence and diagnose the actual movement pair before
proposing another repair.
