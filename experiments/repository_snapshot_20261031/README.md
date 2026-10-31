# CFCMT repository snapshot — 31 October 2026

This snapshot includes the current code, manuscript sources and PDFs, and the
locally available compact experiment records. Existing results are preserved
without retraining, rerunning simulations, or combining controller versions.

- Current submission: [`paper/GPT_TITS_paper/`](../../paper/GPT_TITS_paper/).
  The V93 PDF and source ZIP are also available as
  [`CFCMT_V93_TITS.pdf`](../../paper/GPT_TITS_paper/CFCMT_V93_TITS.pdf) and
  [`CFCMT_V93_TITS_Source.zip`](../../paper/CFCMT_V93_TITS_Source.zip).
- Previous submission text and results:
  [`paper/development_history_20261030/`](../../paper/development_history_20261030/).
  The separate `paper_tsc/` tree retains its earlier manuscript and V43 confirmation.
- Experiment reports: `paper/*_result.md`, `paper/*_outcome.md`, and the existing
  experiment directories. [`evidence_index.csv`](evidence_index.csv) lists
  directories containing the included records, with file counts and byte sizes.
- Added evidence includes protocols, launch receipts, logs, admissions, audits,
  and result summaries, covering 182 local cluster experiment directories.
  Selected text records are at most 5 MiB each. Successful and failed outcomes
  are retained. Small bus-data acquisition and conversion summaries are included.
- Large raw datasets, checkpoints, per-sample label caches, native working trees,
  and build caches remain in their existing local/server archives. Larger result
  files remain there alongside the existing compact reports of their outcomes.

V93 and V43 are different controller versions. The V93 comparison comprises
1,344 runs: 56 paired seeds, four scenarios, and six controllers. Its waiting
reduction is confirmed against MaxPressure; the PhasePressure point estimate
is favorable but its 95% interval crosses zero.
