# TSC paper build contract

Run all commands from the repository root. The submission figures and source
tables are regenerated only by these commands:

```bash
python3 paper_tsc/figures/build_tsc_external_confirmation.py
python3 paper_tsc/figures/build_tsc_causal_transfer_schematic.py
python3 paper_tsc/figures/build_tsc_postfreeze_stress_test.py
python3 paper_tsc/figures/build_tsc_source_gate_development.py
python3 paper_tsc/figures/build_tsc_source_remedy_ladder.py
```

The first builder validates the frozen V43 development, external adaptation,
held-out and closed-loop gates, the post-hoc V90 ablation, the rejected
fresh-seed V91 source-contribution decision and the separate positive V98 Jinan
controller-pair audit. It writes the V91 and V98 controller-comparison rows
without pooling them, requires the V98 comparator to consume zero source rows
and validates the frozen audit's declared lower-capacity boundary, and checks
the metadata-only remote audit covering all 336 result identities. The V91
scientific gate is required to remain rejected; only its evidence integrity is
required to pass. The third builder requires the V86-V89 evidence
audits to pass and requires the preregistered V89 confirmation decision to
remain rejected. The fourth builder validates the frozen V145, V146 and V148
seven-city development aggregates and requires all three decisions to remain
rejected before regenerating their source-data table. The fifth builder
validates the canonical V150A, V150C, V150I--V150L, V151A--V153A and
V156A--V159 remedy artifacts, including the negative closed-loop smoke,
exact-fallback outcomes, superseded V123/V157A domain-confounded comparison,
V157B's corrected B100 branch-only authorization, and V157C's incomplete fixed
three-seed native one-action diagnostic, before writing the remedy ladder. The
V157C row reports only descriptive two-valid-seed means and preserves seed
27178 as invalid at checkpoint 480; it does not make a three-seed inference.
The V158 row reports the completed ten-seed native-prefix out-of-fold (OOF)
estimates. The
builder requires all three V158 comparison gates to remain failed and reserve
authorization to remain false; V158 therefore supports no controller adoption,
unseen-city claim or post-hoc retuning. The V159 row reuses the same B100 bank
to compare direct physical endpoints with the V158 delta representation. Its
endpoint-minus-delta effect is -0.00060648 (paired 95% interval
[-0.00223628, +0.00081223]; 5/10 seeds improve), one of the three demand
settings regresses, and endpoint minus PhasePressure is +0.00065478 (3/10
seeds improve). The builder requires the V159 OOF gate to remain failed and
the direct-endpoint hypothesis to remain closed.
The V157B result binds the local arm manifest and its server-side
runtime-bundle identity; the bundle itself is not required to regenerate the
table.

Then run the paper contract tests and compile:

```bash
PYTHONPATH=. python3 -m pytest -q -s \
  tests/test_tsc_paper_artifacts.py \
  tests/test_tsc_source_remedy_ladder_builder.py
cd paper_tsc
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The submission uses exactly these figure basenames:

- `tsc_anchored_transfer_schematic.pdf`
- `traffic_signal_transfer_networks.pdf`
- `tsc_external_confirmation.pdf`
- `tsc_source_contribution_diagnostic.pdf`

`build_tsc_theory_data_figures.py` and the unreferenced
`cfcmt_theory_stack_schematic`, `cfcmt_benchmark_scope`,
`traffic_signal_network_footprints` and `tsc_causal_transfer_schematic`
artifacts predate the frozen V43 manuscript. They remain as development history
and are not part of the submission build.
