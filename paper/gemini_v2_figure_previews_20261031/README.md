# Revised figure-layout previews

The supplied revision contains six figures: two conceptual schematics and four quantitative layouts. The five main previews comprise the architecture and the four quantitative figures; the mechanism timeline is also supplied separately.

This is a layout review using the draft's supplied arrays and random constructions. The progression lines are manually constructed synthetic trajectories (generator seed 20261031), not SUMO records. The violin distributions are 240 Gaussian draws per method/regime (seed 42), not measured vehicle samples. Weight curves and ablation means/error bars are hard-coded examples. The two schematics describe the proposed draft framework, not the implemented frozen V93 controller. All previews carry a visible status label.

Python/matplotlib generates the PDF, SVG, and 300-dpi PNG exports. The source numerical arrays, distributions, and random seeds are preserved. Labels are wrapped or shortened for legibility; the two-panel ablation is enlarged to a double-column width. The conceptual diagrams are redrawn in Python from the draft's inline vector specifications.

The current V93 manuscript, experimental records, and earlier previews remain separate. These files do not establish new performance or mechanism evidence.

Run `python3 build_all_previews.py` from this directory to reproduce all exports. `five_main_figures_preview.pdf` contains the architecture and the four quantitative layouts (draft Figs. 2–6). `all_six_figures_preview.pdf` additionally includes the timeline (draft Fig. 1). `revised_figures_preview.zip` contains both documents, individual PNG/PDF/SVG figures and their scripts.
