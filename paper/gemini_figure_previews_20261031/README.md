# Five draft figure previews

These previews reproduce the supplied Python chart design, using its supplied illustrative arrays. They are separate from the current V93 manuscript and do not change its figures, data, or conclusions.

Figure 1 is a proposed conceptual workflow, not a verified description of the frozen implementation. Figures 2–5 contain illustrative values: the sensitivity, city comparisons and eight-replicate curves are hard-coded in the supplied script. They are not measured V93 outcomes. The actual V93 evaluation has 56 paired seeds; those results are unchanged.

The figure contract is to preview five supplied layouts at their original double-column dimensions, with visible data-status labels. Python/matplotlib generates PNG previews, editable SVGs, vector PDFs and a combined five-page PDF. The alpha string literals, export routine, workflow arrow routing, and crowded label layout were repaired; the supplied numerical values are unchanged.

Run `python3 build_previews.py` from this directory. `plot_from_draft_original.py` preserves the extracted script.
