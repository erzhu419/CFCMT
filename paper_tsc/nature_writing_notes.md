# Submission writing ledger for the TSC-first paper

## One-sentence argument

In target-offline cross-network signal control, matched action contrast
improves a same-information dense residual, while separate frozen controller
comparisons show target-network-dependent outcomes, including repeatable Jinan
gains and a Los Angeles regression; they do not isolate a general source-row
contribution, and pressure policies remain stronger.

## Terminology ledger

| Canonical term | Meaning | Variants or claims to avoid |
| --- | --- | --- |
| anchored CFCMT | Parent-restricted rigid anchor plus antisymmetric all-pair correction | full CFCMT, pressure-guarded CFCMT |
| MC-WM | Earlier mechanism-factored world-model route retained as method lineage | final controller |
| target offline adaptation | Target SUMO counterfactual labels are available before deployment | zero-shot, target-data-free |
| legacy near-target-only comparator | Frozen v43/v91 causal baseline with target prediction weight numerically near one; its retained source prior can still resolve exact action ties | strict target-only, zero-source-row comparator |
| strict zero-source-row target model | V98 target model fitted only on target rows | architecture-matched target-only |
| V98 target-offline selector | Post-v91 selector that abstains for Los Angeles and admits a Hangzhou component for Jinan before its fresh rollouts | final v43 selector, universal source gate |
| controller-pair contrast | Difference between two declared frozen controllers; V91 and V98 estimate these contrasts | source-row contribution |
| source-row contribution | Effect isolated while architecture and domain handling are held fixed and source rows are removed | V91 or V98 controller-pair contrast, cross-network evaluation alone |
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
| The source-aware controller improves offline action regret over legacy near-target-only fitting. | Macro 0.3104 to 0.2685; LA interval excludes zero, Jinan interval crosses zero. | Supported as a controller-pair contrast in LA; directional in Jinan, with exact-tie source-prior caveat. |
| CFCMT improves the dense residual in closed loop. | 159.15 to 149.17 s all-departed waiting; 6.27%; descriptive interval [2.01, 17.19] s. | Supported on two external cities. |
| The frozen v43 source-plus-target controller improves both external cities over its legacy near-target-only comparator. | V91: macro relative delta +4.25%, interval [-0.77%, 13.21%]; Los Angeles +7.87%, Jinan -3.04%. | Not supported; fresh joint confirmation rejected. |
| The V98-selected controller improves Jinan over its declared comparator. | Target-offline selection abstained for Los Angeles and admitted Hangzhou for Jinan; the selected controller improved by 4.53% on 56/56 fresh seeds. | Supported as a controller-pair contrast; the zero-source-row comparator has lower capacity, so source-row contribution is not isolated. |
| The domain-aligned B100 source arm improves over its target-only comparator on the reused Jinan selector. | V157B: source-minus-target -0.01357, paired 95% interval [-0.01784, -0.00935], 20/22 seeds improve. V157C remained a two-valid-seed diagnostic. V158 ten-seed native-prefix OOF: source-minus-target -0.00113, 95% interval [-0.00240, +0.00019] (6/10 improve); source-minus-placebo -0.00072, [-0.00219, +0.00091] (7/10); source-minus-PhasePressure +0.00014, [-0.00239, +0.00261] (5/10). | Supported only for the reused-selector V157B comparison. V123/V157A were domain-handling-confounded, V157C was incomplete, and all three V158 OOF gates failed. The reserve was withheld; no controller adoption, unseen-city claim or post-hoc retuning is authorized. |
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
- V90 adds the frozen legacy near-target-only comparator on those eight cells post hoc.
- V91 uses 64 new seeds, four scenarios and two frozen policies for 512 rollouts; no refit or seed exclusion is allowed.  Its comparator is legacy near-target-only, not a strict zero-source-row estimator.
- V98 is a separate post-V91 protocol: target-offline selection returns exact target-only for Los Angeles and admits Hangzhou for Jinan; its 56-seed Jinan confirmation is not pooled with V91.
- The frozen final-v43 controller has no pressure guard, fallback or target closed-loop tuning; v98 is a separate guarded successor.
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
2. Replicate a precommitted source-admission rule on additional external city units; the Jinan-specific V98 controller-pair result does not establish general or unseen-city source-row benefit.
3. Either add a same-protocol nearest-method benchmark or retain the comparison gap as an explicit limitation.
4. Replicate on additional independent external cities before making population-level transfer claims.

The canonical artifact commands are listed in `paper_tsc/BUILD.md`. The
unreferenced `build_tsc_theory_data_figures.py` and its v16-era figures are
development history, not submission inputs.
