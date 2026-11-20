# Partial-CV manuscript experiment records

The current records use the corrected Salt Lake City all-red startup. The shared executor preserves the native clearance duration and enters the native next green when internal lanes are clear. The Salt Lake City target labels, fitted controllers, and affected evaluation rows were rebuilt on the original seeds. The source bank and Los Angeles/Jinan evaluation rows were retained after the [startup parity check](round2_startup_parity.json).

The Los Angeles and Jinan manuscript runs use `phase2_augmented_dataset.json`. It contains the Phase 1 bank of 335 source groups and 88 target groups, plus 76 source groups collected on the repaired Hangzhou network with Gudang demand. The target groups are split as 20 per adaptation seed for Los Angeles and 8 per scenario and adaptation seed for Jinan. The fit selects 3 or 18 groups per seed before cross-fitting. Salt Lake City contributes 40 separate adaptation groups and is absent from the source bank.

The [reproduction instructions](REPRODUCTION.md) link to the bundle with the augmented label archive, trained models, and exact simulator inputs. Run this command from the repository root with the experiment's Python environment and a new output directory. In the bundle, the augmented archive is stored as `inputs/phase2_frozen_prereg/phase2_counterfactual_dataset.json`.

```bash
python -m experiments.tsc_partial_cv_pilot_20261118.phase2_residual_transfer \
  --stage fit_paper_archive \
  --archive-labels /path/to/phase2_augmented_dataset.json \
  --work-dir /path/to/paper-phase2-refit
```

This checks the archive's source scenarios, adaptation split and zero-restoration receipt, then writes `phase2_frozen_models.pkl` and `phase2_offline_fit_audit.json`. The checked-in fit audit carries the Phase 1 count of 1,908 source branches. The refit audit counts all 2,516 branches directly from the augmented archive. The separate `collect_and_fit` stage and `phase2_counterfactual_dataset.json` correspond to a different collection and source bank.

For Salt Lake City, pass the same archive to `phase3_tits_suite` with `--stage saltlake --archive-labels /path/to/phase2_augmented_dataset.json`. The stage also uses the configured scenario directories and runs the Salt Lake adaptation branches and closed-loop evaluation. The reported results are already available in `phase3_saltlake_fit_audit.json` and `phase3_saltlake_eval_fresh_result.json`. The penetration sweep uses the Phase 2 models and the original 10% results in `phase2_eval_fresh_result.json` under `--phase2-dir`.

## Corrected Double-DQN comparison

Run Phase 3 with `--stage rl` in a separate work directory. Each replay transition uses its own 10-second step, including forced continuation. The Double-DQN target selects only allowed next actions, and the final reward at 3,600 seconds has no bootstrap term. For round two, the three Salt Lake City target fits were retrained with the corrected executor and the original training seeds. The manuscript reads `phase3_rl_baselines_repaired_result.json`, which combines the corrected 60 Salt Lake City evaluations with the 120 retained evaluations. The original `phase3_rl_baselines_result.json` records the earlier implementation.

For the source-only comparison, run Phase 3 with `--stage rl_source_only`. The ten original source training episodes produce one frozen model. Round two uses those same weights with the corrected executor on the 20 Salt Lake City evaluation cells. The full source-only panel has 60 cells at 10% CV penetration.

`phase3_rl_source_only_result.json` contains the evaluation rows. `phase3_rl_source_only_comparison.json` pairs them with the archived CFCMT B=0 rows. The comparison averages scenarios within city and seed, then weights the three cities equally. Inference uses the ten shared Los Angeles–Jinan seed blocks and ten independent Salt Lake City seed blocks. The intervals use 10,000 bootstrap draws. The exact sign-flip tests for waiting time and vehicle-hours use Holm adjustment across those two outcomes.

## Manuscript revision experiments

- [Fixed-label mechanism ablations](phase4_mechanism_ablation_result.json) compare routed and full source pools, source rejection, deployment scaling, and ranker representations on the original held-out cells.
- [Congestion replays](phase4_case_trace_summary.json) give entry backlog and signal states, and [visited-state branches](phase4_case_diagnostic_result.json) compare cost horizons and continuation policies.
- [Independent target-label repeats](phase4_independent_repeat_result.json) report three nested adaptation splits on a shared held-out panel. [Truth-reward Double-DQN](phase4_truth_reward_rl_result.json) records each fit, its simulator-time budget, and evaluation on both the original and shared new seeds.
