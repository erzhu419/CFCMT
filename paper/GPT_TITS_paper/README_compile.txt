CFCMT: Structured Action-Cost Transfer for Traffic Signal Control
IEEE Transactions on Intelligent Transportation Systems manuscript
Updated 30 October 2026

SCOPE

This submission describes the final V93 source selector and its complete
same-cohort evaluation: 56 paired seeds, four scenarios, six controllers,
1,344 one-hour simulations. All six controllers and all evaluation seeds are retained.

CONTENTS

main.tex, main.pdf         Main manuscript and compiled PDF.
sections/                 Final method, experiments, results and proof appendix.
references.bib, main.bbl   21 references; 12 verified arXiv versions.
IEEEtran.cls, IEEEtran.bst Unmodified IEEE journal class and bibliography style.
figures/                  Workflow and final comparison, previews/build scripts.
source_data/              V93 five-method records and complete MP-extended data.
experiments/              Compact MP supplement scripts, protocol and records.
cf_h2o/config/            Frozen V93 confirmation specification.
scripts/cluster/          MP evaluation worker.
EDITORIAL_NOTES_zh.txt     Current manuscript notes.

BUILD

Compile main.tex. Python is not required to compile the manuscript.

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

Or:

    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex

In Overleaf, upload this ZIP and select main.tex as the main document.
The manuscript is 9 US-letter pages, including two figures, two tables, the
component-configuration/proof appendix and references. Normal IEEE 10-point typography is retained.

REBUILD FIGURES

Python dependencies: matplotlib and numpy.

    python3 figures/build_figures.py
    python3 figures/build_source_selector_with_mp.py

No figure uses a simulation seed identifier or ordering as a plotted variable.
The final comparison displays recorded overall and city-specific intervals
and the empirical distribution of all paired relative waiting changes.

SOURCE DATA AND STATISTICS

- traffic_signal_transfer_networks.csv: network inventory, Table I.
- source_selector_confirmation_{rollouts,summary,comparisons,seed}.csv:
  the V93 five-controller evaluation (1,120 runs), retained for its recorded
  intervals and as the input to the fixed-baseline MP supplement.
- source_selector_with_mp_{rollouts,summary,comparisons,seed}.csv:
  the complete six-controller evaluation (1,344 runs), Table II and Figure 2.
- reference_versions.csv: version verification for all 21 references.

The primary registered relative effect forms a ratio within each city/seed,
then averages cities and seeds. It is not a ratio of pooled waiting means.
Against MP: -2.420%, 95% interval [-4.581%, -0.662%], 40/56 favorable,
Holm15 p=0.005772. Against PP: -1.913%, interval [-4.419%, +0.077%].
Recorded V93 intervals are preserved; the added MP intervals use 20,000
paired seed-block bootstrap replicates with RNG seeds 20261030/31/32.
Four-comparator Holm12 and expanded Holm15 are separate columns in the extended export.

Optional analysis reproduction from the unpacked ZIP (requires numpy/scipy):

    python3 experiments/tsc_v93_max_pressure_20261030/report.py --shards-root experiments/tsc_v93_max_pressure_20261030/shards/_shards --source-data source_data --out-dir /tmp/cfcmt-v93-analysis --analysis-out /tmp/cfcmt-v93-analysis/analysis.json

Simulation execution requires the original repository and archived server
runtime, network inputs and fitted models; this is not a complete native
simulation reproduction bundle. The report above uses only included compact
records. Full trajectories and trained model files remain in the archive.

CODE PROVENANCE

Repository: https://github.com/erzhu419/CFCMT
Source-component code: commit 61e8a81e10c058cf07125efe86f2a8c8e2dd1c18.
The October MP supplement scripts and compact records accompany this package.
The internal h2oplus identifier denotes the project dense linear residual
baseline, not the published H2O+ algorithm. The target anchor retains a source
prior and is not strictly source-free. The final source selector uses Atlanta
for Jinan and the target anchor for Los Angeles, with the target information
and execution rules stated in the manuscript.

AUTHOR INFORMATION

Replace the author-information placeholder with the author list, affiliations
and correspondence details. Review the claims and AI-assistance acknowledgment
before submission. Benchmark inputs retain their upstream licences.
