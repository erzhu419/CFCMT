# CFCMT evidence update — 2026-11-12 submission version

Main manuscript: `main.pdf` (10 pages). Supplement: `Supplementary_material.pdf`.
The main manuscript has 18 inline bibliography entries. Corresponding arXiv
records are used where available. `source_data/reference_versions.csv` records
the cited versions, and `source_data/reference_publication_updates.json` documents the
earlier publisher-version audit. The paper builds with the inline bibliography.

This update separates the historical 1,568-run / Holm18 panel from the independent
896-run / Holm9 panel, which varies only SUMO evaluation seeds under fixed network
and demand files. CFCMT retains the best observed equal-city waiting,
vehicle-hours, and completion means. No fresh equal-city waiting comparison
passes the prespecified confirmation criterion. Jinan matches GP in 167/168 runs
in each panel. Learned-intervention and source-data increments remain unestablished.
Historical waiting distributions and the development guard plot move to
Supplementary Figures S2 and S3; Figure S1 and Table S1 retain their historical data.

Target fitting uses 165 Los Angeles and 911 Jinan matched action groups from two
simulator seeds per city. Eight separate simulator seeds supply development
outcomes for configuration selection. Both evaluation panels use the archived
Jinan v8 network. The later audit found reversed right-of-way in 24 of 48
conflicting right-turn and through-movement pairs sharing an exit road under
concurrent green. The static audit and its comparison to the V93 network archive
are in `source_data/jinan_network_audit/`.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=Supplementary_material supplementary.tex
```

Recompute the fresh statistics using Python 3, numpy, and scipy:

```bash
python3 source_data/fresh_confirmation/analyze.py --records source_data/fresh_confirmation/rollouts.csv --out-dir recomputed_fresh
```

Recompute the historical statistical and execution summaries:

```bash
python3 experiments/tsc_v93_exact_reference_20261031/analyze.py --source-data source_data --shards-root experiments/tsc_v93_exact_reference_20261031/shards/_shards --out-dir recomputed_historical
python3 source_data/recompute_execution_summaries.py --out recomputed_execution
```

`source_data/seven_controller_*` is the historical seven-controller panel;
`confirmation_*` is its earlier six-controller subset. Fresh records, protocol,
seed searches, duplicate-run exclusion, guard counts, and analysis live under
`source_data/fresh_confirmation/`. Recorded absolute server paths are provenance;
the bundled statistical scripts consume local scalar records.
`figure_source/build_real_figures.py` regenerates the historical figures and
`Historical_supplement.pdf` using numpy, pandas, and matplotlib. Rebuild
`supplementary.tex` afterwards to assemble the complete supplement.

Validation: bundled fresh and historical summary/comparison tables reproduce the
reported tables exactly. Main and supplement compile without undefined references
or overfull boxes; rendered pages were inspected. Native simulation reproduction
requires the archived runtime, model files, and benchmark assets, which are not
included in this compact source package.

Supplementary Note S1 reports the separate native-label pilot: 195 source groups,
100 reused target challenge groups, 15 fits, zero new simulations. All three
paired intervals cross zero and source-arm harm increases 13.19%; the fixed
construction is closed under its development gate. Complete scalar outcomes and
protocol are in `source_data/native_transfer/`; adapter, runner, unchanged bank
helper, and focused tests are in `native_reproduction/` under their original repo
paths. Refit requires the frozen CFCMT snapshot and original server banks named
in `source_inputs.json`; they are not bundled. No additional fits are needed to
read or rebuild the manuscript.
