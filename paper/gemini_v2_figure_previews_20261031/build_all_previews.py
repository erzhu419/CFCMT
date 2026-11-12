"""Export the revised draft's conceptual and illustrative figures."""

import os
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from schematic_figures import draw_framework, draw_timeline
import build_experimental_previews as experimental


ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    timeline = draw_timeline()
    framework = draw_framework()
    for stem, figure in [
        ("fig1_mechanism_timeline", timeline),
        ("fig2_framework_overview", framework),
    ]:
        for extension in ("pdf", "png", "svg"):
            figure.savefig(figure_dir / f"{stem}.{extension}",
                           bbox_inches="tight", dpi=300)

    experimental.PREVIEW_FIGURES.clear()
    experimental.plot_queue_trajectories()
    experimental.plot_source_alpha_profile()
    experimental.plot_waiting_violin()
    experimental.plot_ablation_sensitivity()
    main_figures = [framework, *experimental.PREVIEW_FIGURES]
    for name, figures in [
        ("five_main_figures_preview.pdf", main_figures),
        ("all_six_figures_preview.pdf", [timeline, *main_figures]),
    ]:
        with PdfPages(ROOT / name) as document:
            for figure in figures:
                document.savefig(figure, bbox_inches="tight")

    members = [
        *ROOT.glob("*.py"), ROOT / "README.md", ROOT / "schematics_original.tex",
        ROOT / "five_main_figures_preview.pdf", ROOT / "all_six_figures_preview.pdf",
        *sorted(figure_dir.glob("*.png")), *sorted(figure_dir.glob("*.pdf")),
        *sorted(figure_dir.glob("*.svg")),
    ]
    with zipfile.ZipFile(ROOT / "revised_figures_preview.zip", "w",
                         compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(ROOT))
    plt.close("all")
    print("Saved five main previews, all six previews, and the figure ZIP.")


if __name__ == "__main__":
    main()
