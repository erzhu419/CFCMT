# Partial-CV benchmark reproduction

The [round-two bundle](../../paper/CFCMT_PartialCV_Reproduction_Round2.tar.gz) contains the exact network and demand files, the corrected executor and experiment code, the augmented source labels, and the frozen models and compact results used for the revision. The [earlier bundle](../../paper/CFCMT_PartialCV_Reproduction_94743f89.tar.gz) records the pre-correction benchmark.

The 411 source groups contain 2,516 action branches. The 1,908 count in the legacy Phase 2 audit predates the Gudang augmentation. The original target label banks contain 40 Los Angeles groups and 48 Jinan groups. The corrected Salt Lake City bank contains 40 groups. A further 108 corrected Salt Lake City groups cover the three independent adaptation repeats.

## Verify from a clean extraction

Use Linux x86-64 and Python 3.10. The environment file pins SUMO/libsumo 1.22.0 and the scientific libraries.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python verify_archive.py --smoke-sumo
paper=paper/CFCMT_TITS_PartialCV_20261119
python "$paper/figure_source/paired_comparisons.py"
python "$paper/figure_source/paired_comparisons.py" --source-only
```

The verifier refits the main rankers and the three Salt Lake City repeats and checks score vectors for exact agreement with their frozen versions. The repeats were evaluated under Python 3.12.3 with the same scientific-library and SUMO versions. The verifier allows an absolute tolerance of \(10^{-15}\) for the smooth Boltzmann regret in the audit. In the third repeat this value differs by \(2.8\times 10^{-17}\) under Python 3.10, and the original runtime reproduces it exactly. The startup check exercises all six target scenarios and the eight source scenarios used by the label bank. It checks that the four Salt Lake City signals hold their native all-red phase for five seconds and then enter the intended green, including when the controller requests that same green during clearance. The [startup parity receipt](round2_startup_parity.json) records matching 25-second trajectories between executor versions in the other cities and the used source scenarios.

## Rebuild the corrected Salt Lake City runs

From the extracted bundle root, set the variables below and start from fresh work directories. The unchanged, archived Los Angeles and Jinan rows are joined with the corrected Salt Lake City cells by scenario, policy, and seed.

```bash
jobs=experiments.tsc_partial_cv_pilot_20261118
records=experiments/tsc_partial_cv_pilot_20261118
inputs="$PWD/inputs"
phase2="$inputs/phase2_frozen_prereg"
labels="$phase2/phase2_counterfactual_dataset.json"
staged="$inputs/phase0/staged_scenarios"
source_cfgs="$inputs/source_scenarios"
ext="$inputs/ext_targets"
pretrained="$inputs/phase3_rl_source_only_20261120/results/phase3_rl_source_only_models.pkl"
scenario_args=(--staged-scenarios-dir "$staged" --source-scenarios-dir "$source_cfgs" --ext-targets-dir "$ext")

python -m "$jobs.phase3_tits_suite" --stage saltlake --phase2-dir "$phase2" \
  "${scenario_args[@]}" --work-dir work/saltlake
python -m "$jobs.phase4_mechanism_ablation" --archive-root "$inputs" \
  --saltlake-run-dir work/saltlake --family saltlake --work-dir work/ablation
python -m "$jobs.phase4_independent_repeat" --archive-labels "$labels" \
  --staged-scenarios-dir "$staged" --ext-targets-dir "$ext" --families saltlake \
  --reuse-results "$records/phase4_independent_repeat_result.json" --work-dir work/repeats
python -m "$jobs.phase4_case_diagnostics" --experiment-root "$inputs" \
  --saltlake-dir work/saltlake --output work/cases

python -m "$jobs.phase3_tits_suite" --stage rl --phase2-dir "$phase2" \
  "${scenario_args[@]}" --rl-family saltlake --rl-source-models "$pretrained" \
  --work-dir work/rl/cv
python -m "$jobs.phase3_tits_suite" --stage rl_source_only --phase2-dir "$phase2" \
  "${scenario_args[@]}" --rl-family saltlake --rl-source-models "$pretrained" \
  --work-dir work/rl/source
python -m "$jobs.phase4_truth_reward_rl" --archive-labels "$labels" \
  --saltlake-labels work/saltlake/phase3_saltlake_labels.json \
  "${scenario_args[@]}" --families saltlake --work-dir work/rl/truth
python -m "$jobs.merge_saltlake_rl_results" --archived "$records" \
  --runs work/rl --output work/merged_rl
```

Use the bundled figure scripts with the merged result files. The case runner writes `case_trace_summary.json`, which the plotting script reads under the repository name `phase4_case_trace_summary.json`. Install Matplotlib to regenerate the figures.
