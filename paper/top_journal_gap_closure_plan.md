# Top-Journal Gap Closure Plan

This note records the remaining work needed to move the CFCMT manuscript toward
a top transportation journal submission, with emphasis on Part C / IEEE T-ITS
experimental standards.

## Current Strongest Evidence

The strongest current result is the extended RESCO traffic-signal-control stress
test.  It uses eight RESCO scenarios, three matched seeds, 600-second closed-loop
rollouts, original SUMO `tlLogic` green phases as the action set, a generated
native SUMO actuated baseline, and official SUMO `tripinfo` metrics.

| method | mean queue | p90 queue | trip time | trip wait | time loss | throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MaxPressure | 12.9068 | 19.5279 | 87.5891 | 8.0605 | 24.5171 | 0.8267 |
| CFCMT + pressure guard | 12.9180 | 19.5973 | 87.4863 | 8.0114 | 24.4127 | 0.8256 |
| Phase pressure | 13.2446 | 19.3911 | 86.5371 | 7.4544 | 23.4628 | 0.8199 |
| H2O+-style dense phase MPC | 17.2279 | 28.9325 | 86.6585 | 8.5768 | 24.5488 | 0.7653 |
| Target-static simulator MPC | 20.4023 | 32.5538 | 97.9633 | 19.9032 | 35.3928 | 0.7429 |
| Native actuated SUMO | 30.1825 | 47.6873 | 119.7813 | 33.0017 | 57.0805 | 0.7580 |
| Fixed RESCO signal programs | 42.6732 | 69.6852 | 142.3116 | 54.6333 | 79.8759 | 0.7127 |

The paired seed-target bootstrap favors pressure-guarded CFCMT against
H2O+-style dense residual transfer, simulator-only MPC, native actuated SUMO,
spillback pressure, and fixed programs.  Against MaxPressure, the mean queue delta is slightly
unfavorable but the 95% interval crosses zero.  Against phase pressure, the mean
delta is favorable but also crosses zero.  The correct claim is
MaxPressure/phase-pressure-level parity, not significant dominance over
classical pressure control.

The 3600-second long-horizon check is a stricter boundary condition:

| method | mean queue | p90 queue | trip time | trip wait | time loss | throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase pressure | 32.2343 | 62.1800 | 104.2291 | 20.8528 | 38.5434 | 0.9492 |
| MaxPressure | 42.7872 | 71.6277 | 106.9182 | 20.5521 | 40.7331 | 0.9448 |
| CFCMT + pressure guard | 43.5106 | 74.2496 | 108.6326 | 22.0400 | 42.4485 | 0.9448 |
| Target-static simulator MPC | 44.8973 | 65.4094 | 258.2547 | 169.8900 | 188.9974 | 0.8754 |
| H2O+-style dense phase MPC | 49.5058 | 74.9880 | 221.4173 | 131.1706 | 150.3275 | 0.8827 |
| Native actuated SUMO | 58.3468 | 99.4055 | 164.5754 | 66.1326 | 98.1604 | 0.9714 |
| Fixed RESCO signal programs | 84.6088 | 133.0312 | 221.2093 | 113.3988 | 154.3776 | 0.9385 |

At 3600 seconds, phase pressure is clearly better than guarded CFCMT, while
guarded CFCMT remains better than dense residual transfer, native actuated
SUMO, and fixed programs.
This makes the honest claim sharper: CFCMT is a safe residual/transfer layer
around physical priors, not a long-horizon replacement for the best pressure
rule.

## Gap-Closure Status

### 1. Fix the Main Story

The paper currently has two related but different storylines:

- bus-holding cross-city causal mechanism transfer;
- traffic-signal-control RESCO stress testing.

The bus-holding setting has a strong classical-rule ceiling.  The safer
top-journal positioning is:

> CFCMT is a calibration-light causal mechanism transfer framework, validated
> first on cross-city bus residual/passive dynamics and then stress-tested in
> traffic signal control where counterfactual control is richer.

An even cleaner future paper could make TSC the main environment and leave bus
holding as motivation plus passive validation.

Status update: the current manuscript remains bus-first, with RESCO traffic
signal control positioned as a counterfactual stress test rather than as the
primary problem definition.  The abstract, introduction, results, discussion,
and conclusion now use the bounded claim: CFCMT is a calibration-light
mechanism-transfer layer that improves dense residual/simulator-only transfer
and can reach pressure-rule-level safety when coupled with physical guards, but
it does not universally dominate classical pressure control.

### 2. Upgrade RESCO to Official-Scale Validation

The earlier RESCO result used five scenarios and one seed.  The closure run
upgrades this to multi-scenario, multi-seed evidence.

Completed additions:

- included `grid4x4`, `arterial4x4`, `cologne1`, `cologne3`, `cologne8`,
  `ingolstadt1`, `ingolstadt7`, and `ingolstadt21`;
- added matched multi-seed evaluation (`131,337,911`);
- reported mean queue, seed standard deviation, p90 queue, throughput, paired
  seed-target bootstrap intervals, and per-target winners;
- added a 3600-second long-horizon check;
- excluded Salt Lake from the reported set because the discovered `.flo.xml`
  files contain route definitions but no usable vehicle/flow demand.  Salt Lake
  needs separate CSV-to-flow generation before it can be included honestly.

### 3. Strengthen Baselines

The current baselines are now fixed program, generated native SUMO actuated
signals, phase pressure, MaxPressure, spillback pressure, simulator-MPC, and
H2O+-style dense residual.  The remaining harder comparators are optional but
useful:

- same-information dense residual / ensemble baselines;
- selected full-budget RESCO/LibSignal RL baselines if the paper pivots to a
  leaderboard-style TSC submission;
- disciplined naming: use "H2O+-style dense residual" unless the original H2O+
  training stack is faithfully reproduced.

Status update: the official RESCO RL adapter is implemented in
`cf_h2o/eval/traffic_signal_resco_official_rl_baseline.py`.  IDQN/MPLight pilots
validate execution/log parsing, and a five-episode short-budget check now covers
the extended RESCO set after explicit route overrides for `grid4x4` and
`arterial4x4`.  `IDQN` runs on all eight scenarios; `MPLight` still hits an
upstream PFRL shared-DQN indexing error on `cologne8` and `ingolstadt21`.
A 20-episode/5-test-episode, seed-131 mid-budget diagnostic confirms the same
pattern: `IDQN` completes all eight scenarios, `MPLight` completes six and fails
only on those two large irregular scenarios.  The official adapter now has
per-run cache/resume, which was validated by resuming an interrupted mid-budget
run.  It also supports `--skip-runs`, so full-budget RESCO comparisons can
report the supported official algorithm-scenario set without treating known
upstream `MPLight` crashes as experiment failures.
The supported official RESCO full-budget run is now complete: 42 supported
runs finished with zero failures under 100 training episodes, five testing
episodes, and seeds `131,337,911`.  The final table covers `IDQN` on all eight
extended scenarios and `MPLight` on the six upstream-supported scenarios; the
known `MPLight:cologne8` and `MPLight:ingolstadt21` failures remain explicitly
skipped.  The full result is recorded in
`cf_h2o/results/traffic_signal_resco_official_rl_full.md`.

LibSignal is now also connected through
`cf_h2o/eval/traffic_signal_libsignal_baseline.py`.  SUMO smoke runs complete
for `DQN`, `PressLight`, `FRAP`, and `MPLight` on `sumo1x1` after local
SUMO-only compatibility patches to the LibSignal checkout; 5-episode,
600-step diagnostics now complete for those four agents on `sumo1x1`,
`sumo1x3`, and `sumo4x4`.  `sumo4x4` needed missing SUMO XML assets copied from
the RESCO checkout, now automated by the adapter.  The LibSignal full-budget
run is now complete after cProfile-driven optimization: 36 runs finished with
zero failures for `DQN`, `PressLight`, `FRAP`, and `MPLight` on `sumo1x1`,
`sumo1x3`, and `sumo4x4`, with seeds `131,337,911`, 100 training episodes, and
3600-step final tests.  The optimized protocol disables per-episode train-time
tests, disables checkpoint writes, and runs CPU-only with one native/PyTorch
thread per subprocess under `--workers 8`.  The final result is recorded in
`cf_h2o/results/traffic_signal_libsignal_full_optimized.md`.

### 4. Explain and Bound Failures

CFCMT is not uniformly dominant.  Some RESCO targets still favor pressure or
spillback pressure.  The paper must explain:

- which network regimes favor CFCMT;
- which regimes favor classical pressure rules;
- whether a source-only or passive-target guarded selector can detect this;
- what information the selector is allowed to use.

The passive policy selector is currently a negative result and should not be
overstated.

### 5. Add Statistical Support

Completed:

- paired seed uncertainty;
- target-level win counts;
- confidence intervals for CFCMT vs H2O+-style dense, simulator MPC, phase
  pressure, MaxPressure, generated native actuated SUMO, and fixed programs;
- completed-trip travel time and completed waiting time in the RESCO benchmark
  output;
- official SUMO `tripinfo` duration, waiting time, time-loss, and depart-delay
  metrics in the RESCO benchmark output and paper tables;
- 3600-second long-horizon check.

Still useful:

- If this becomes a leaderboard-first TSC paper, run one additional sensitivity
  pass with `workers=12` or `workers=14` to reduce wall-clock time and confirm
  that the one-thread-per-process protocol remains stable.  The completed
  `workers=8` run is already valid and should not be rerun unless more speed is
  needed.

### 6. Thicken the Theory

The manuscript needs a formal framework rather than only engineering narrative:

- source/target domain definitions;
- zero-shot, target-static, and few-shot target offline adaptation information
  budgets;
- mechanism parent sets and invariance assumptions;
- why dense residuals can mix domain-specific nuisance variation;
- residual/trust error decomposition;
- counterfactual data source separation: passive real logs validate prediction
  and OPE-like quantities; SUMO or intervention logs are needed for action
  counterfactuals.

Status update: the problem and method sections now define the target-information
budgets, parent-set invariance assumption, passive-log versus counterfactual
claim boundary, and the pressure-guard control analogue.  This is enough to
make the current experimental story defensible, although a future TSC-first
paper could still expand the theory around MPC regret and pressure-guard error
decomposition.

### 7. Reposition Claims

Safe claim:

> CFCMT provides a calibration-light mechanism-factored transfer model that
> improves dense residual transfer and simulator-only MPC under cross-network
> validation, with stronger control evidence in RESCO TSC than in bus-holding
> closed-loop control.

Unsafe claims until more evidence is added:

- CFCMT universally beats classical rules;
- CFCMT is field-ready without calibration;
- CFCMT has proven real-world counterfactual policy superiority from passive
  AVL/APC logs alone.

## Execution Order

Completed in this pass:

1. Extended RESCO multi-seed validation.
2. MaxPressure baseline and pressure-guarded CFCMT controller.
3. Paired seed-target uncertainty and failure-mode reporting.
4. RESCO paper figure plus CSV/Markdown/LaTeX table generation.
5. Generated native SUMO actuated baseline and official `tripinfo` metrics.
6. Official RESCO RL adapter plus short IDQN/MPLight pilot, extended
   short-budget route-fix validation, and 20-episode mid-budget diagnostic.
7. LibSignal SUMO adapter plus DQN/PressLight/FRAP/MPLight smoke,
   multi-network short-budget validation, cProfile-driven optimization, and
   full-budget 36-run leaderboard cross-check.
8. Problem, method, results, and discussion updates around information budgets,
   pressure guarding, and claim scope.

Remaining before submission:

1. Decide manuscript positioning: bus-first with TSC stress test, or a future
   TSC-first paper with bus as passive transfer evidence.
2. Do journal-style copyediting and visual QA on the 36-page compiled PDF,
   especially table float ordering and appendix density.
3. Keep the strongest claim bounded: CFCMT reaches MaxPressure/phase-pressure
   level safety under a pressure guard on the 600-second stress test, but the
   3600-second check favors phase pressure; it does not prove universal
   dominance over classical control.

Completed manuscript-placement decision:

- The completed RESCO and LibSignal full-budget RL tables are included in the
  appendix as native-protocol cross-checks.
- They are referenced in the results and discussion, but not mixed into the
  same-protocol CFCMT phase-MPC comparison table.
