# V154F Cologne repaired-network controller comparison

The six-cell comparison is **COMPLETE**. All three rigid runs have zero
native collisions; PhasePressure has zero collisions in two seeds and one
incident in seed 52282. Rigid remains slower on both service metrics in every
seed: its three-seed mean waiting is **50.70% higher** and system vehicle
hours are **24.98% higher** than PhasePressure on the same repaired network.
The geometry result and the remaining controller-performance deficit are
therefore distinct findings.

## Fixed comparison and provenance

Both controllers use the retained V154D bilateral network, with internal
lengths 13/24 = 8.48/20.05 m and 3/20 = 8.37/19.83 m. The original 2015-trip
demand, SUMO 1.22.0, controller source `85b2b43cd12832c5a1dd`, 3600-second
window from 25200 to 28800, 60-second warmup, 10-second control interval,
phase clearance and native collision/no-teleport settings are identical.

Rigid uses the original fitted model, stored-feature-name binding,
450-second prediction horizon, direct execution and zero cooldown.
PhasePressure uses the original evaluator with `models=None`. Both retain
the original occupancy equations. No network, model or seed was adjusted
after observing these runs.

| Seed | Rigid target-only | PhasePressure |
|---|---|---|
| 52282 | New run, `t91006` | New run, `t91007` |
| 41242 | Reused V154E repair arm, `t90988` | New run, `t91008` |
| 63792 | New run, `t91009` | New run, `t91010` |

The roster is the existing V150L `full.seeds`, in its original order. The
matrix contains **five new simulations and one reused cell**. Seed 41242 is
the geometry-development seed; 52282 and 63792 provide the prespecified
additional-simulation summary.

## Runtime completeness and collision outcomes

Every cell is valid: all 3600 steps completed, uncapped event/incident/step
counts match the evaluator, no teleports occurred, and population and
tripinfo identities hold. Each cell has the original 3541 post-warmup metric
samples. Due demand is 2015 throughout.

| Seed | Rigid incidents / reports | PP incidents / reports | Rigid arrivals | PP arrivals | Pending at end, both arms |
|---|---:|---:|---:|---:|---:|
| 52282 | 0 / 0 | 1 / 2 | 1995 | 1997 | 0 |
| 41242 | 0 / 0 | 0 / 0 | 1993 | 1996 | 1 |
| 63792 | 0 / 0 | 0 / 0 | 1996 | 1996 | 0 |

The rigid zero-collision diagnostic is **PASS, 3/3 seeds**. PhasePressure's
is **FAIL, 2/3 seeds**, with one incident and two repeated native reports in
52282. The paired incident differences, rigid minus PhasePressure, are
−1, 0 and 0. The colliding PhasePressure cell is a completed valid run and
remains included in every service mean.

The remaining incident is reported at 27481 and 27482. In its first frame,
`200684_437_0` is moving at 9.606 m/s on straight internal lane 16_1
(link 17); `201382_438_0` is moving at 11.989 m/s with its front already on
outgoing lane `32324544#0_1`. Its observed route and unique static connection
identify a U-turn via link 9, internal lanes 9_0→23_0 and the same outgoing
lane. The preceding 23_0 passage is inferred from that route/connection,
not directly observed in the first collision frame. This is a different
movement pair from the repaired 13/1_1 and 3/11_1 waiting points.

The observed signal is yellow. At 27482 the native report retains the first
incident's speeds, while actual participant speeds have changed to 0.606
and 13.407 m/s; the retained analysis distinguishes these values. The next
localized diagnostic target is the straight/U-turn merge into the shared
outlet, with no further geometry change selected by this comparison.

## Paired service results

All differences below are **rigid minus PhasePressure**. Positive waiting
and system-hour differences indicate worse service for rigid.

| Seed | Rigid waiting (s) | PP waiting (s) | Difference (s) | Rigid system hours | PP system hours | Difference (veh·h) |
|---|---:|---:|---:|---:|---:|---:|
| 52282 | 19.660546 | 12.472953 | +7.187593 | 38.963333 | 31.328333 | +7.635000 |
| 41242 | 20.962264 | 14.357498 | +6.604767 | 41.150833 | 33.409444 | +7.741389 |
| 63792 | 19.752854 | 13.231762 | +6.521092 | 41.457778 | 32.535556 | +8.922222 |

Equal weighting of all three seeds gives:

| Metric | Rigid mean | PP mean | Paired mean difference | Relative difference of means |
|---|---:|---:|---:|---:|
| Mean tripinfo waiting (s), primary | 20.125221 | 13.354071 | +6.771150 | +50.70% |
| System vehicle hours | 40.523981 | 32.424444 | +8.099537 | +24.98% |
| Active vehicle hours | 33.667870 | 28.490000 | +5.177870 | +18.17% |
| Pending vehicle hours | 6.856111 | 3.934444 | +2.921667 | +74.26% |
| Mean departure delay (s) | 12.268662 | 7.048073 | +5.220589 | +74.07% |
| Mean queue (vehicles) | 10.376259 | 6.894098 | +3.482161 | +50.51% |
| Arrivals | 1994.666667 | 1996.333333 | −1.666667 | −0.0835% |
| Departures | 2014.666667 | 2014.666667 | 0 | 0% |

Relative differences use the ratio of the two equally weighted arm means
minus one. They are not averages of the per-seed percentages.

The arrival difference is small: two fewer vehicles, three fewer, and a tie.
It does not explain away the waiting and system-hour deficits, which occur
in every seed. Rigid accumulates both more active vehicle time and more
pending-insertion time. Endpoint pending counts happen to match, while their
integrals differ over the hour. The current controller's main shortfall is
therefore delay and queue service, rather than a material loss of the demand
population or widespread failure to finish trips.

Waiting covers all departed vehicles, including unfinished tripinfo, and
excludes pending insertion. System vehicle hours integrate active plus
pending vehicles over the original post-warmup samples. Mean departure delay
is a separate statistic. These populations are unchanged across arms.

## Additional seeds

The prespecified 52282/63792 summary excludes the geometry-development seed
without replacing the main three-seed comparison:

| Metric | Rigid mean | PP mean | Paired mean difference | Relative difference of means |
|---|---:|---:|---:|---:|
| Mean tripinfo waiting (s) | 19.706700 | 12.852357 | +6.854342 | +53.33% |
| System vehicle hours | 40.210556 | 31.931944 | +8.278611 | +25.93% |
| Arrivals | 1995.500000 | 1996.500000 | −1.000000 | −0.0501% |

Rigid has zero collisions in both additional seeds; PhasePressure has zero
in 63792 and the single retained incident in 52282. The service deficit
persists in the additional simulations. It is not driven only by the seed
used to construct the geometry.

## Execution and artifacts

The tooling snapshot is `8c7842518a4d53225905`; the controller snapshot remains
`85b2b43cd12832c5a1dd`. The runner and summarizer have **22 passing tests**.
Independent comparison against all six source branches reproduced every
aggregate service value, both seed-group means, collision totals, elapsed
time sum and retrieval size without discrepancy.

The five new runs consumed **21.938090 seconds** of summed runner wall time.
This is execution time, excluding scheduler queueing and staging; the reused
V154E cell is not charged again. Each task requested one CPU and 8192 MB.
Only **697,623 bytes** of new result JSON were retrieved. Network, demand,
model and scratch tripinfo stayed on the server.

- [Frozen protocol](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154f_cologne_repaired_controller_comparison_protocol.md)
- [Six-cell aggregate](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154f_cologne_repaired_controller_comparison_v1.json)
- [Launch record](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v154f_cologne_repaired_controller_comparison_20260909/launch_v1.json)
- [Reused V154E source, repair branch](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v154e_cologne_bilateral_full_validation_20260909/paired_full_v1/result.json)
- [PP/52282 collision evidence](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v154f_cologne_repaired_controller_comparison_20260909/seed_52282/phase_pressure/result.json)
- [Remaining-collision analysis, 8,217-byte local summary](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154f_phase_pressure_remaining_collision_v1.json)

Every new cell's exact result path and task ID are recorded in the aggregate.

## Limitations

This is one city, one demand file, three fixed simulator seeds and one
fitted rigid model. The geometry-development seed is included explicitly.
The additional seeds are simulations of the frozen geometry, not independent
training holdouts. No new significance test, tuned service threshold,
source-transfer claim or benefit of the newer occupancy equations is
introduced. V154D's short-window FAIL and V154E's original-duration PASS
remain separate results. This comparison observes fewer collisions for rigid
and lower delay for PhasePressure; it does not establish universal superiority
of either controller.

The remaining PP incident's full victim body is unavailable in its first
frame because the rear is still on the preceding lane. The evidence does
not yet establish overlap geometry or distinguish yielding, merge geometry
and signal-transition causes; observing yellow does not establish the cause.

The auxiliary strict lane-sequence diagnostic for `168358_425_0` in
PP/63792 is false because one-second observations omit its short waiting
lane 3_0. Its unchanged route is observed on continuation 20_0 at 25591 and
the actual outgoing edge at 25593. This is incomplete intermediate sampling,
not a vehicle that failed to pass the junction, and the prespecified diagnostic
is not a validity or service gate.
