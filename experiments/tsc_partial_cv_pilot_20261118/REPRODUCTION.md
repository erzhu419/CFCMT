# Partial-CV benchmark reproduction

The reproduction bundle contains the inputs and frozen controllers from the reviewed benchmark, its source snapshot from commit 94743f89, the repaired Double-DQN and source-only weights, and the corresponding evaluation receipts. It includes the staged road networks and demand files that the SUMO configurations actually read, with their relative paths preserved. The original simulator inputs and weights are in `inputs/`. The capture uses Linux x86-64, Python 3.10.20, SUMO/libsumo 1.22.0, NumPy 2.2.6, SciPy 1.15.3, and scikit-learn 1.7.2.

The [prepared reproduction bundle](../../paper/CFCMT_PartialCV_Reproduction_94743f89.tar.gz) contains 7,020,535 bytes. Its SHA-256 is `c435fb2ec6e561dd95b7e7b4fbf9c353b241f33f4d95823b2a3f479e52cddf83`, verified against the server archive after transfer.

The native archive contains **411 source groups and 2,516 source action branches**. The historical Phase 2 audit retains the pre-augmentation count of 1,908. The validator reads the augmented archive supplied under its original server filename and recomputes the correct count. The target archives contain 40 Los Angeles groups, 48 Jinan groups, and 40 Salt Lake City groups. The Salt Lake City labels come from native-prefix replays and reproduce the original recorded fit audit.

From the extracted bundle root, create a clean Python 3.10 environment.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python verify_archive.py --smoke-sumo
python paper/CFCMT_TITS_PartialCV_20261119/figure_source/paired_comparisons.py
```

The verifier refits the archived labels for all three cities, compares the fit audits and policy scores on the archived target groups, and loads the six held-out network and demand configurations in SUMO. The statistical script reads the archived evaluation rows.

Verification from a fresh extraction reproduced all three historical fit audits, matched 1,024 policy/group score vectors against the frozen controllers, loaded all six target configurations with libsumo 1.22.0, and reproduced all 12 Table III rows from the reviewed manuscript.

To write the refitted Los Angeles and Jinan weights, run the following.

```bash
python -m experiments.tsc_partial_cv_pilot_20261118.phase2_residual_transfer \
  --stage fit_paper_archive \
  --archive-labels inputs/phase2_frozen_prereg/phase2_counterfactual_dataset.json \
  --work-dir work/refit
```

Use `--stage eval_fresh` and the same work directory for the original held-out Los Angeles and Jinan panel, passing `--staged-scenarios-dir inputs/phase0/staged_scenarios` and `--source-scenarios-dir inputs/source_scenarios`. For Salt Lake City, use `phase3_tits_suite --stage saltlake` with those same scenario directories, `--phase2-dir inputs/phase2_frozen_prereg`, `--ext-targets-dir inputs/ext_targets`, and a fresh work directory. The `rl` and `rl_source_only` stages rebuild the two Double-DQN panels.

## Revision experiments

Run the new modules from the revised project checkout with the same environment. Set `inputs` to the absolute path of the extracted `inputs/` directory and use fresh output directories.

```bash
jobs=experiments.tsc_partial_cv_pilot_20261118
labels="$inputs/phase2_frozen_prereg/phase2_counterfactual_dataset.json"
saltlake="$inputs/phase4_mechanism_ablation_20261120/saltlake_original_labels.json"
staged="$inputs/phase0/staged_scenarios"
external="$inputs/ext_targets"

python -m "$jobs.phase4_mechanism_ablation" --archive-root "$inputs" \
  --saltlake-labels "$saltlake" --work-dir work/ablation
python -m "$jobs.phase4_case_diagnostics" --experiment-root "$inputs" \
  --output work/cases
python -m "$jobs.phase4_independent_repeat" --archive-labels "$labels" \
  --staged-scenarios-dir "$staged" --ext-targets-dir "$external" --work-dir work/repeats
python -m "$jobs.phase4_truth_reward_rl" --archive-labels "$labels" --saltlake-labels "$saltlake" \
  --staged-scenarios-dir "$staged" --source-scenarios-dir "$inputs/source_scenarios" \
  --ext-targets-dir "$external" --families la jinan saltlake --work-dir work/truth-reward
```

The repository collects the revised results as `phase4_*_result.json`. The case runner writes the compact plot data to `work/cases/case_trace_summary.json`. To regenerate the figure from that output, install Matplotlib and run:

```bash
cp work/cases/case_trace_summary.json \
  experiments/tsc_partial_cv_pilot_20261118/phase4_case_trace_summary.json
python paper/CFCMT_TITS_PartialCV_20261119/figure_source/build_case_figure.py
```

The archived package provides the baseline inputs and original weights. The revision runners reproduce the additional fits and measurements from those inputs.
