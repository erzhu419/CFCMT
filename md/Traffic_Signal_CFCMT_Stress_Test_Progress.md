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
| RESCO CFCMT transfer | `cf_h2o/eval/traffic_signal_resco_cfcmt_benchmark.py` | real RESCO networks, leave-one-scenario-out phase residual transfer | CFCMT global has the best aggregate mean queue: 7.1265 vs 7.3115 pressure, 7.6559 H2O+ dense, and 28.4898 fixed program. |

## Current RESCO CFCMT Result

Result file: `cf_h2o/results/traffic_signal_resco_cfcmt_benchmark.md`.

| policy | mean queue | p90 queue | throughput ratio |
| --- | --- | --- | --- |
| CFCMT global phase MPC | 7.1265 | 13.0394 | 0.9096 |
| CFCMT few-shot bias phase MPC | 7.1265 | 13.0394 | 0.9096 |
| CFCMT trust phase MPC | 7.1955 | 13.0380 | 0.9071 |
| CFCMT few-shot selector phase MPC | 7.2715 | 13.0218 | 0.9101 |
| Passive policy selector phase MPC | 7.2132 | 12.7504 | 0.9116 |
| Phase pressure | 7.3115 | 12.9773 | 0.9035 |
| Target static simulator MPC | 7.3127 | 12.9074 | 0.9092 |
| H2O+-style dense phase MPC | 7.6559 | 13.4804 | 0.9048 |
| Phase spillback pressure | 8.9068 | 15.9597 | 0.8958 |
| Fixed RESCO programs | 28.4898 | 47.9729 | 0.8527 |

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

1. CFCMT is not merely beating a weak fixed controller; on RESCO it beats a
   strong phase-pressure rule on aggregate.
2. The dense H2O+-style residual is weaker than the sparse CFCMT residual in
   both synthetic multi-intersection SUMO and real RESCO phase-transfer tests.
3. The simulator prior is very strong.  The target-static simulator and
   source-average simulator rows are close to pressure rules, so the paper must
   report the information budget explicitly.
4. CFCMT is not uniformly dominant.  `cologne8` and `ingolstadt7` still favor
   classical pressure variants, which should be reported as boundary cases.
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
- The RESCO benchmark is a transfer stress test, not a faithful RESCO RL
  leaderboard submission.  It should be described as cross-network phase-MPC
  transfer over RESCO SUMO scenarios.

## Remaining Work

1. Add paired seed/bootstrap uncertainty for the RESCO aggregate table.
2. Add a stronger source-only guarded selector, because the passive-target
   policy selector did not reliably identify pressure-rule-winning targets.
3. Add a compact figure comparing fixed, pressure, simulator MPC, H2O+ dense,
   and CFCMT across Phase 2 and RESCO.
4. Decide whether the paper remains bus-control-first with TSC as an additional
   stress test, or whether a future paper should pivot fully to TSC.
