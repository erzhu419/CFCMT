# CFCMT manuscript: second review revision

Version: **CFCMT-V93-ExactReference-20261031**. Main text and appendix are in the
single `main.tex`, including all 18 reference entries. No `.bbl` file or BibTeX step is needed.
Source packaging revision: 2026-11-01 (inline bibliography; experiment results unchanged). Build:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The main PDF has 10 pages, four figures and three tables. The two-page
`Supplementary_material.pdf` contains departure-cohort Fig. S1 and execution Table S1.
`python3 figure_source/build_real_figures.py` regenerates all figures and the supplement
using numpy, pandas and matplotlib. Figure 3 uses all 1,568 recorded runs.

The added 224 exact-reference runs preserve the V93 model, parameters, networks,
demands, executor and 56 reused seeds. New simulations ran on jtl110gpu2; node001–node007
received no new simulation tasks. The one-run technical smoke duplicates a formal
scenario/seed and is excluded from formal estimates. Raw tripinfo and model files
remain on the servers; this package contains compact per-run records and traces.

`source_data/seven_controller_*.csv` contains the current seven-controller evidence.
The six-controller `confirmation_*.csv` files remain historical. Per-run counters
in `execution_rollouts.csv` reconstruct the eight learned-policy summary rows and
six original incident rows; `seven_controller_evaluation.metadata.json` defines
the full metric scope. Missing rule-policy guard counters are not zeros.

From the extracted source archive, independently regenerate the statistics with
numpy and scipy, then regenerate the original execution/incident summaries:

```bash
python3 experiments/tsc_v93_exact_reference_20261031/analyze.py --source-data source_data --shards-root experiments/tsc_v93_exact_reference_20261031/shards/_shards --out-dir recomputed_statistics
python3 source_data/recompute_execution_summaries.py --out recomputed_execution
```

The paired GP interval crosses zero. Jinan matches GP in 167/168 runs across all 10
outcome fields; its sole non-reference request raises waiting and completion.
Positive incremental learned-control benefit and cross-city source-data benefit
are not established by this panel. The original MP comparison remains favorable.
The eight development and 56 evaluation seeds are disjoint within this round but
reuse the earlier 64-seed panel; this is not a full-history independent confirmation.

Original code/records are fixed at GitHub commit `12e7955d0350e4cac2a344d04954cfb5efea70a2`.
The versioned source ZIP supplies the manuscript-specific additions. Exact native
simulation reproduction requires the frozen runtime/assets; no weights are bundled.
Author order, correspondence, affiliations and all five e-mail addresses are retained.
