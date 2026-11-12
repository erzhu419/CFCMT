"""Diagram of matched action labels, predictor fitting, and phase control."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 7.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
}
BLUE, ORANGE, TEAL = "#286587", "#AD6C2B", "#237A6D"


def _box(ax, x, y, width, height, *, fill="white", edge="#41505A",
         dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.025,rounding_size=0.08",
        facecolor=fill, edgecolor=edge, linewidth=0.85,
        linestyle=(0, (4, 2.5)) if dashed else "-", zorder=2,
    ))


def _arrow(ax, start, end, *, color="#41505A", dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=6.5, linewidth=0.8,
        color=color, linestyle="--" if dashed else "-", shrinkA=1,
        shrinkB=2, zorder=3,
    ))


def _route(ax, points, *, color="#41505A", dashed=False):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, linewidth=0.8,
            linestyle="--" if dashed else "-", zorder=1)
    _arrow(ax, points[-2], points[-1], color=color, dashed=dashed)


def _node(ax, x, y, width, height, title, lines, *, edge,
          fill="white", title_size=7.0, body_size=6.3):
    _box(ax, x-width/2, y-height/2, width, height, edge=edge, fill=fill)
    ax.text(x, y+height/2-height*0.075, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color="#243540",
            linespacing=1.2, zorder=4)
    title_rows = title.count("\n") + 1
    top = y+height/2-height*(0.37+0.15*(title_rows-1))
    bottom = y-height/2+height*0.14
    positions = [(top+bottom)/2] if len(lines)==1 else [
        top+(bottom-top)*i/(len(lines)-1) for i in range(len(lines))
    ]
    for line, yy in zip(lines, positions):
        ax.text(x, yy, line, ha="center", va="center", fontsize=body_size,
                color="#263746", zorder=4)


def draw_framework():
    """Return the matched-label protocol, fitting, and execution workflow."""
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.16, 4.3))
        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.085, top=0.98)
        ax.set(xlim=(-0.2, 18.2), ylim=(-0.55, 8.7))
        ax.axis("off")
        panels = [
            (0, 5.6, "#E8F1F5", BLUE, "I. Offline action labels"),
            (5.9, 5.8, "#FDF2E9", ORANGE, "II. Fitting and selection"),
            (12.0, 6.0, "#E8F8F5", TEAL, "III. Frozen phase control"),
        ]
        for x, width, fill, edge, title in panels:
            _box(ax, x, 0.12, width, 8.35, fill=fill, edge=edge, dashed=True)
            ax.text(x+width/2, 8.2, title, ha="center", va="top",
                    fontsize=8.0, fontweight="bold", color=edge)

        _node(ax, 2.8, 6.55, 5.0, 2.15, "Multi-city source bank",
              ["18 networks in 7 source groups",
               "Atlanta · Cologne · Hangzhou",
               "Ingolstadt · New York",
               "Synthetic · Salt Lake City"], edge=BLUE, body_size=6.1)
        _node(ax, 2.8, 4.0, 5.0, 2.1, "Target adaptation data",
              ["Seeds 5057 and 6067",
               "Los Angeles: 165 action groups",
               "Jinan: 911 action groups"], edge=BLUE, body_size=6.2)
        _node(ax, 2.8, 1.5, 5.0, 2.4, "Matched-label protocol",
              ["60-s warm-up; save simulator + executor",
               "Restore state for every candidate phase",
               "10-s request + 50-s PP continuation",
               "Other signals: PP throughout",
               r"$z_g(a)=[Y_g(a)-Y_g(r_g)]/\sigma_g$"],
              edge=BLUE, body_size=6.0)

        _node(ax, 8.8, 6.35, 5.15, 2.8, "One source-group component",
              ["One source group + target groups",
               "Reference model: 29 features",
               "Pairwise model: 23 + 40 features",
               r"$S_g=(1-\alpha_g)A_g+\alpha_gP_g$",
               r"Jinan: $\alpha_g=0.9$", "Los Angeles: local rule"],
              edge=ORANGE, body_size=6.1)
        _node(ax, 8.8, 3.75, 5.15, 1.95, "Target anchor (117 features)",
              ["Source head + target head",
               "Near-unit target-head weight",
               "Domain-scaled cost contrasts"], edge=ORANGE, body_size=6.1)
        _node(ax, 8.8, 1.45, 5.15, 2.2, "Development selection",
              ["8 target development seeds",
               "Source identity → weight → guard",
               "Freeze before evaluation"], edge=ORANGE, body_size=6.1)

        _node(ax, 15.0, 6.5, 5.35, 2.3, "Selected city configurations",
              ["Los Angeles: target anchor",
               "Source-group mass 0; no score guard",
               "Jinan: Atlanta; source-group mass 1",
               r"Score-guard multiplier $\beta_n=0.5$"], edge=TEAL, body_size=6.1)
        _node(ax, 15.0, 3.75, 5.35, 2.4, "Screen phase proposals",
              [r"$\widehat a=\arg\min_{a\in\mathcal{A}_g} F_g(a)$",
               r"Jinan: $F_g(\widehat a)+0.5u_g(\widehat a)<0$",
               "Normalized pressure deficit within cap",
               "Otherwise: pressure reference"], edge=TEAL, body_size=6.1)
        _node(ax, 15.0, 1.25, 5.35, 1.8, "Coordinate and execute",
              ["Greedy selection of eligible proposals",
               "Minimum intervention gap: 60 s",
               "10-s decisions; common executor"], edge=TEAL, body_size=6.0)

        # Label protocol is a shared collection rule for both data banks.
        _arrow(ax, (5.3, 6.55), (6.225, 6.55), color=BLUE)
        _route(ax, [(5.3, 4.2), (5.6, 4.2), (5.6, 5.15), (6.225, 5.15)], color=BLUE)
        _arrow(ax, (5.3, 3.7), (6.225, 3.7), color=BLUE)
        _route(ax, [(11.375, 6.35), (11.65, 6.35),
                    (11.65, 1.45), (11.375, 1.45)], color=ORANGE)
        _arrow(ax, (8.8, 2.775), (8.8, 2.55), color=ORANGE)
        _route(ax, [(11.375, 1.45), (11.85, 1.45),
                    (11.85, 6.5), (12.325, 6.5)], color=ORANGE)
        _arrow(ax, (15.0, 5.35), (15.0, 4.95), color=TEAL)
        _arrow(ax, (15.0, 2.55), (15.0, 2.15), color=TEAL)
        fig.text(0.5, 0.024,
                 "Matched simulator trials offline; frozen predictions every 10 s online",
                 ha="center", fontsize=6.4, color="#46535C")
        return fig


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    figure = draw_framework()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(destination / f"fig1_framework.{extension}", dpi=300,
                       bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)
