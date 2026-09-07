# Traffic Signal CFCMT Stress-Test Progress

This note records the traffic-signal-control branch that was added after the
bus-holding experiments hit a strong rule-controller ceiling.  The purpose is
not to replace the bus paper immediately, but to test whether CFCMT has a
cleaner transfer setting with richer counterfactual control structure.

## Completed Stages

| stage | file | setting | key result |
| --- | --- | --- | --- |
| Phase 0 analytic feasibility | `cf_h2o/eval/traffic_signal_transfer_feasibility.py` | queueing model, leave-one-network-out | CFCMT few-shot/global is near oracle and ahead of H2O+ dense and pressure rules. |
| Phase 1 SUMO | `cf_h2o/eval/traffic_signal_sumo_phase1.py` | generated 3x3 grid, one controlled TLS | `sim_mpc` is best; CFCMT global/trust beat H2O+ dense and pressure rules on aggregate. |
| Phase 2 SUMO | `cf_h2o/eval/traffic_signal_sumo_phase2.py` | generated 4x4 grid, four controlled TLS, OD routes | target-static `sim_mpc` is best; CFCMT trust/few-shot is close to generic simulator MPC and better than H2O+ dense and pressure rules. |
| RESCO probe | `cf_h2o/eval/traffic_signal_resco_probe.py` | real RESCO SUMO scenarios | `cologne1`, `cologne3`, `cologne8`, `ingolstadt1`, and `ingolstadt7` load under libsumo. |
| RESCO phase baseline | `cf_h2o/eval/traffic_signal_resco_phase_benchmark.py` | real `tlLogic` green phases as actions | phase pressure beats fixed programs by a large margin; arbitrary real phase actions are valid. |
| RESCO CFCMT transfer | `cf_h2o/eval/traffic_signal_resco_cfcmt_benchmark.py` | real RESCO networks, leave-one-scenario-out phase residual transfer | Core set: CFCMT global is best. Extended set: CFCMT + pressure guard roughly matches MaxPressure, beats dense/simulator/native-actuated/fixed baselines, but loses to phase pressure on 3600-second horizon. |
| Official RESCO RL adapter | `cf_h2o/eval/traffic_signal_resco_official_rl_baseline.py` | official RESCO state/reward/action/logging loop | IDQN/MPLight pilots complete; IDQN covers all eight extended scenarios in short- and mid-budget checks; MPLight covers six and still has upstream failures on two large irregular scenarios. |
| LibSignal adapter | `cf_h2o/eval/traffic_signal_libsignal_baseline.py` | LibSignal SUMO state/reward/action/logging loop | DQN, PressLight, FRAP, and MPLight 5x600 diagnostics complete on `sumo1x1`, `sumo1x3`, and `sumo4x4`; these validate integration but are not trained baselines. |

## Current RESCO CFCMT Result

Core result file: `cf_h2o/results/traffic_signal_resco_cfcmt_benchmark.md`.

| policy | mean queue | p90 queue | throughput ratio |
| --- | --- | --- | --- |
| CFCMT global phase MPC | 7.1265 | 13.0394 | 0.9096 |
| CFCMT few-shot bias phase MPC | 7.1265 | 13.0394 | 0.9096 |
| CFCMT trust phase MPC | 7.1955 | 13.0380 | 0.9071 |
| Passive policy selector phase MPC | 7.2132 | 12.7504 | 0.9116 |
| CFCMT few-shot selector phase MPC | 7.2715 | 13.0218 | 0.9101 |
| Phase pressure | 7.3115 | 12.9773 | 0.9035 |
| Target static simulator MPC | 7.3127 | 12.9074 | 0.9092 |
| H2O+-style dense phase MPC | 7.6559 | 13.4804 | 0.9048 |
| Phase spillback pressure | 8.9068 | 15.9597 | 0.8958 |
| Fixed RESCO programs | 28.4898 | 47.9729 | 0.8527 |

Extended eight-scenario, three-seed validation file:
`cf_h2o/results/traffic_signal_resco_cfcmt_extended_multiseed_validation.md`.

| policy | mean queue | seed std | p90 queue | trip time | trip wait | time loss | throughput ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MaxPressure | 12.9068 | 0.1546 | 19.5279 | 87.5891 | 8.0605 | 24.5171 | 0.8267 |
| CFCMT + pressure guard | 12.9180 | 0.1919 | 19.5973 | 87.4863 | 8.0114 | 24.4127 | 0.8256 |
| Phase pressure | 13.2446 | 0.1816 | 19.3911 | 86.5371 | 7.4544 | 23.4628 | 0.8199 |
| H2O+-style dense phase MPC | 17.2279 | 3.6084 | 28.9325 | 86.6585 | 8.5768 | 24.5488 | 0.7653 |
| Target static simulator MPC | 20.4023 | 0.1758 | 32.5538 | 97.9633 | 19.9032 | 35.3928 | 0.7429 |
| Native actuated SUMO | 30.1825 | 0.3536 | 47.6873 | 119.7813 | 33.0017 | 57.0805 | 0.7580 |
| Fixed RESCO programs | 42.6732 | 0.2709 | 69.6852 | 142.3116 | 54.6333 | 79.8759 | 0.7127 |

Extended paired seed-target bootstrap uses
`cfcmt_phase_pressure_guard_mpc - baseline`; negative values favor CFCMT.

| baseline | mean delta | 95% CI | wins |
| --- | --- | --- | --- |
| MaxPressure | 0.0112 | [-0.0457, 0.0787] | 3/24 |
| Phase pressure | -0.3266 | [-0.7588, 0.0227] | 12/24 |
| Native actuated SUMO | -17.2645 | [-21.8067, -12.9979] | 24/24 |
| Target static simulator MPC | -7.4843 | [-14.3783, -1.7428] | 18/24 |
| H2O+-style dense phase MPC | -4.3099 | [-9.6002, -0.9210] | 18/24 |
| Fixed RESCO programs | -29.7552 | [-36.5483, -23.3073] | 24/24 |

Extended 3600-second long-horizon file:
`cf_h2o/results/traffic_signal_resco_cfcmt_extended_multiseed_3600s_validation.md`.

| policy | mean queue | seed std | p90 queue | trip time | trip wait | time loss | throughput ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase pressure | 32.2343 | 0.9901 | 62.1800 | 104.2291 | 20.8528 | 38.5434 | 0.9492 |
| MaxPressure | 42.7872 | 0.7314 | 71.6277 | 106.9182 | 20.5521 | 40.7331 | 0.9448 |
| CFCMT + pressure guard | 43.5106 | 0.5699 | 74.2496 | 108.6326 | 22.0400 | 42.4485 | 0.9448 |
| Target static simulator MPC | 44.8973 | 1.2335 | 65.4094 | 258.2547 | 169.8900 | 188.9974 | 0.8754 |
| H2O+-style dense phase MPC | 49.5058 | 3.9510 | 74.9880 | 221.4173 | 131.1706 | 150.3275 | 0.8827 |
| Native actuated SUMO | 58.3468 | 1.1370 | 99.4055 | 164.5754 | 66.1326 | 98.1604 | 0.9714 |
| Fixed RESCO programs | 84.6088 | 0.6458 | 133.0312 | 221.2093 | 113.3988 | 154.3776 | 0.9385 |

At 3600 seconds, phase pressure is clearly best.  Guarded CFCMT is close to
MaxPressure, remains better than dense H2O+-style residual transfer,
native-actuated SUMO, and fixed programs, but is significantly worse than phase
pressure.

Earlier core-only paired target bootstrap, kept as a historical smoke
diagnostic, used `cfcmt_phase_global_mpc - baseline` on target mean queue;
negative values favor CFCMT.

| baseline | mean delta | 95% CI | target wins |
| --- | --- | --- | --- |
| Phase pressure | -0.1850 | [-0.8988, 0.3295] | 3/5 |
| Target static simulator MPC | -0.1862 | [-0.2823, -0.0491] | 4/5 |
| H2O+-style dense phase MPC | -0.5293 | [-1.2475, -0.0607] | 4/5 |
| Phase spillback pressure | -1.7802 | [-5.0996, 0.3135] | 4/5 |
| Fixed RESCO programs | -21.3632 | [-27.5960, -16.2052] | 5/5 |

Per-target winners are mixed:

| target | winner |
| --- | --- |
| `cologne1` | CFCMT global |
| `cologne3` | CFCMT trust |
| `cologne8` | phase pressure |
| `ingolstadt1` | H2O+ dense / CFCMT tied by action choice |
| `ingolstadt7` | phase spillback pressure |

## Interpretation

The TSC branch now has a stronger story than the bus-holding rule-ceiling
problem:

1. CFCMT is not merely beating a weak fixed controller.  On the 600-second
   extended RESCO set, the pressure-guarded CFCMT variant beats dense residual
   baselines, simulator MPC, spillback pressure, and fixed programs.  Against
   MaxPressure it is effectively tied with a slightly worse mean; against phase
   pressure it has a better mean but a CI that still crosses zero.  On the
   3600-second check, phase pressure is clearly better than guarded CFCMT.
2. The dense H2O+-style residual is weaker than the sparse CFCMT residual in
   both synthetic multi-intersection SUMO and real RESCO phase-transfer tests.
3. The simulator prior is very strong.  The target-static simulator and
   source-average simulator rows are close to pressure rules, so the paper must
   report the information budget explicitly.
4. CFCMT is not uniformly dominant.  Several targets and the 3600-second horizon
   still favor MaxPressure, phase pressure, simulator MPC, or H2O+-style dense
   residuals.  These are boundary cases, not failures to hide.
5. The passive policy selector is currently a negative result: it selects among
   rule/sim/H2O/CFCMT policies without target rollouts, but does not fix the two
   pressure-rule-winning RESCO targets.

## Validation Guardrails

- RESCO policies use the same SUMO seed within each target scenario, so policy
  comparisons are not confounded by per-policy route-choice randomness.
- Actions are selected from the original SUMO `tlLogic` green phases; there is
  no synthetic NS/EW assumption in the RESCO scripts.
- Few-shot variants use passive target transitions only.  They do not inspect
  target closed-loop policy rollouts.
- The pressure guard is target-static and conservative: it falls back to
  MaxPressure on large regular networks, very high-demand single-junction
  regimes, and sparse low-demand targets.
- The RESCO benchmark is a transfer stress test, not a faithful RESCO RL
  leaderboard submission.  It should be described as cross-network phase-MPC
  transfer over RESCO SUMO scenarios.
- The official RESCO and LibSignal adapters are intentionally separate because
  those agents use different state, reward, action, and metric definitions from
  the CFCMT phase-MPC stress test.
- LibSignal required local SUMO-only compatibility patches to skip optional
  CityFlow/CoLight imports.  Use a pinned LibSignal environment or explicitly
  cite the patched checkout before reporting full leaderboard numbers.

## Remaining Work

Completed after this note was first drafted:

1. Added MaxPressure and a conservative CFCMT pressure guard.
2. Ran the extended eight-scenario, three-seed RESCO validation.
3. Added a compact paper figure and CSV/Markdown/LaTeX result tables.
4. Added official RESCO RL adapter with parallel subprocess execution and
   IDQN/MPLight pilot plus extended short-budget route-fix validation and
   20-episode mid-budget diagnostic, with resume/cache and `--skip-runs` support
   for known upstream official RESCO failures.
5. Added LibSignal adapter with DQN/PressLight/FRAP/MPLight smoke and
   multi-network short-budget validation.
6. Updated the paper problem, method, results, and discussion sections.

Still open:

1. Add a stronger source-only guarded selector if we want CFCMT to improve over
   MaxPressure rather than only match it under a guard.
2. Run selected official RESCO/LibSignal RL baselines at full training budget if
   we pivot to a traffic-signal-control-first leaderboard paper rather than
   using RESCO as a transfer stress test.
3. Decide whether the paper remains bus-control-first with TSC as an additional
   stress test, or whether a future paper should pivot fully to TSC.
