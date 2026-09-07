# TSC paper build contract

Run all commands from the repository root. The submission figures and source
tables are regenerated only by these commands:

```bash
python3 paper_tsc/figures/build_tsc_external_confirmation.py
python3 paper_tsc/figures/build_tsc_causal_transfer_schematic.py
python3 paper_tsc/figures/build_tsc_postfreeze_stress_test.py
python3 paper_tsc/figures/build_tsc_source_gate_development.py
```

The first builder validates the frozen v43 development, external adaptation,
held-out and closed-loop gates, the post-hoc v90 target-only ablation and the
rejected fresh-seed v91 source-contribution decision before writing artifacts.
The v91 scientific gate is required to remain rejected; only its evidence
integrity is required to pass. The third builder requires the v86-v89 evidence
audits to pass and requires the preregistered v89 confirmation decision to
remain rejected. The fourth builder validates the frozen v145, v146 and v148
seven-city development aggregates and requires all three decisions to remain
rejected before regenerating their source-data table.

Then run the paper contract tests and compile:

```bash
python3 -m pytest -q -s tests/test_tsc_paper_artifacts.py
cd paper_tsc
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The submission uses exactly these figure basenames:

- `tsc_anchored_transfer_schematic.pdf`
- `traffic_signal_transfer_networks.pdf`
- `tsc_external_confirmation.pdf`

`build_tsc_theory_data_figures.py` and the unreferenced
`cfcmt_theory_stack_schematic`, `cfcmt_benchmark_scope`,
`traffic_signal_network_footprints` and `tsc_causal_transfer_schematic`
artifacts predate the frozen v43 manuscript. They remain as development history
and are not part of the submission build.
