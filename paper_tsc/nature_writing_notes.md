# Submission writing ledger for the TSC-first paper

## One-sentence argument

In target-offline cross-network signal control, matched action contrast
improves a same-information dense residual, but a frozen target-label-only
comparison shows that source-labelled counterfactuals have network-dependent
closed-loop effects: they help Jinan, harm Los Angeles and do not establish a
general source contribution, while pressure policies remain stronger.

## Terminology ledger

| Canonical term | Meaning | Variants or claims to avoid |
| --- | --- | --- |
| anchored CFCMT | Parent-restricted rigid anchor plus antisymmetric all-pair correction | full CFCMT, pressure-guarded CFCMT |
| MC-WM | Earlier mechanism-factored world-model route retained as method lineage | final controller |
| target offline adaptation | Target SUMO counterfactual labels are available before deployment | zero-shot, target-data-free |
| target-label-only comparator | Same causal action-ranking family and target labels, with target prediction weight one and zero deployed source contribution | source-free features, independently redesigned target policy |
| source contribution | Difference between frozen source-plus-target CFCMT and the target-label-only comparator | cross-network evaluation alone |
| calibration-light | Target simulator parameters are not fitted to field trajectories | calibration-free, sim-to-real |
| H2O+-style dense residual | Same-protocol dense residual inspired by simulator-plus-residual transfer | H2O+ reproduction |
| pressure reference | Coordinate origin for matched contrasts | deployment guard, fallback |
| same-protocol comparator | Shared target, seeds, actions, executor and metrics | native leaderboard number |
| native-protocol RL check | RESCO/LibSignal execution under upstream definitions | paired CFCMT baseline |
| external city | Los Angeles or Jinan city-derived benchmark geometry | calibrated digital twin, field deployment |
| post-freeze successor stress test | v86--v89 120-s latent originator plus global execution cooldown on Jinan | final anchored v43 controller, cross-network confirmation |

## Claim-evidence map

| Claim | Evidence | Status |
| --- | --- | --- |
| Action contrast improves seed-held-out development. | Regret 0.2414 to 0.2177; 9.84% city-macro gain; 6/7 groups improve. | Supported. |
| External offline point estimates improve over the rigid anchor in both cities. | LA 2.18%; Jinan 10.10%; macro 8.09%; no city regression. | Supported, but LA interval crosses zero. |
| Source labels improve offline action regret over target-label-only fitting. | Macro 0.3104 to 0.2685; LA interval excludes zero, Jinan interval crosses zero. | Supported in LA; directional in Jinan. |
| CFCMT improves the dense residual in closed loop. | 159.15 to 149.17 s all-departed waiting; 6.27%; descriptive interval [2.01, 17.19] s. | Supported on two external cities. |
| Source labels improve closed-loop control over target-label-only fitting. | V91: CFCMT 155.14 s versus 149.09 s; relative delta +4.25%, interval [-0.77%, 13.21%]; LA +7.87%, Jinan -3.04%. | Not supported; fresh joint confirmation rejected. |
| CFCMT is better than simulator-only and rigid-anchor control. | Point gains of 7.54 s and 3.75 s. | Directional; intervals cross zero. |
| CFCMT beats classical pressure control. | MaxPressure 141.74 s and phase pressure 143.62 s versus CFCMT 149.17 s. | Not supported; pressure is stronger. |
| A longer-horizon successor closes the phase-pressure gap. | V89 fresh-seed mean difference -0.043%; 95% interval [-0.449%, 0.370%]; sign-test p=0.354; joint gate failed. | Not supported. |
| CFCMT demonstrates field causal transfer. | All action counterfactuals and closed-loop outcomes are from SUMO. | Not supported. |
| CFCMT is zero-shot. | Target adaptation uses 165 LA and 911 Jinan matched action groups. | False for the final experiment. |

## Protocol boundaries

- Development uses 18 networks from seven city groups and an excluded seed 4047.
- External adaptation uses target seeds 5057 and 6067; selection leaves one complete seed out.
- Offline confirmation uses seed 8171 only after joint freeze.
- Closed-loop confirmation uses seeds 8081 and 9091 in 56 matched 3,600-s rollouts.
- V90 adds the frozen target-label-only comparator on those eight cells post hoc.
- V91 uses 64 new seeds, four scenarios and two frozen policies for 512 rollouts; no refit or seed exclusion is allowed.
- The final controller has no pressure guard, fallback or target closed-loop tuning.
- The distinct v86--v89 successor uses 54 adaptive seeds and 64 fresh Jinan seeds; it is not pooled with v43, and v85 remains sealed.
- Los Angeles and Jinan are the only independent external-city units; intervals are descriptive.
- CrossLight, X-Light, MetaLight and GESA have not been reproduced under the same executor and information budgets.
- A persistent public repository DOI is still required before submission.

## Figure contract

| Figure | Core conclusion | Status |
| --- | --- | --- |
| Figure 1 | The estimand progresses from dense absolute residuals through MC-WM to matched action ranking, followed by a frozen target-adaptation protocol. | Generated from Python in PDF, SVG, PNG and TIFF. |
| Figure 2 | Development and external targets differ in controlled-signal and lane scale; converted targets are not claimed as calibrated twins. | Generated from the admitted network inventory. |
| Figure 3 | Development and external offline regret both improve, and the closed-loop gain is strongest against dense residual transfer. | Generated only after all frozen artifact gates pass. |

## Submission blockers

1. Deposit code, converted inputs, per-rollout records, models and source data in a persistent archive.
2. Develop a source-inclusion or source-weight rule without using v91 outcomes, then confirm it on new cities; the current method does not establish source benefit.
3. Either add a same-protocol nearest-method benchmark or retain the comparison gap as an explicit limitation.
4. Replicate on additional independent external cities before making population-level transfer claims.

The canonical artifact commands are listed in `paper_tsc/BUILD.md`. The
unreferenced `build_tsc_theory_data_figures.py` and its v16-era figures are
development history, not submission inputs.
