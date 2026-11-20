# Partial-CV manuscript experiment records

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

Run Phase 3 with `--stage rl` in a separate work directory. Each replay transition now uses its own 10-second step, including forced continuation, and the Double-DQN target selects only next actions allowed by the maximum-red filter. The final reward at 3,600 seconds has no bootstrap term. The rerun used the original 70 training episodes and the same 180 evaluation cells on `jtl110gpu2`. The manuscript reads `phase3_rl_baselines_repaired_result.json`. The original `phase3_rl_baselines_result.json` records the earlier implementation. Both files contain evaluation summaries, and the trained weights remain on the server.

For the source-only comparison, run Phase 3 with `--stage rl_source_only`. The original experiment retained the target-adapted weights. This stage repeats the 10 original source episodes and evaluates one frozen source model in all 60 target scenario-and-seed cells at 10% CV penetration. The source training audit matches the repaired RL run. The new weights remain on the server.

`phase3_rl_source_only_result.json` contains the evaluation rows. `phase3_rl_source_only_comparison.json` pairs them with the archived CFCMT B=0 rows. The comparison averages scenarios within city and seed, then weights the three cities equally. Inference uses the ten shared Los Angeles–Jinan seed blocks and ten independent Salt Lake City seed blocks. The intervals use 10,000 bootstrap draws. The exact sign-flip tests for waiting time and vehicle-hours use Holm adjustment across those two outcomes.

## Manuscript revision experiments

- [Fixed-label mechanism ablations](phase4_mechanism_ablation_result.json) compare routed and full source pools, source rejection, deployment scaling, and ranker representations on the original held-out cells.
- [Congestion replays](phase4_case_trace_summary.json) give entry backlog and signal states, and [visited-state branches](phase4_case_diagnostic_result.json) compare cost horizons and continuation policies.
- [Independent target-label repeats](phase4_independent_repeat_result.json) report three nested adaptation splits on a shared held-out panel. [Truth-reward Double-DQN](phase4_truth_reward_rl_result.json) records each fit, its simulator-time budget, and evaluation on both the original and shared new seeds.
