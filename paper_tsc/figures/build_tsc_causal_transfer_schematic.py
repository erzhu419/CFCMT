from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
OUT = HERE / "tsc_anchored_transfer_schematic"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "svg.hashsalt": "cfcmt-v43-paper",
        "pdf.fonttype": 42,
        "font.size": 7.0,
    }
)


COL = {
    "ink": "#252B32",
    "muted": "#6A7581",
    "line": "#B8C1CB",
    "blue": "#365F8A",
    "blue_soft": "#EEF4FA",
    "green": "#2E7D57",
    "green_soft": "#EAF5EF",
    "teal": "#347E82",
    "teal_soft": "#EAF5F5",
    "gold": "#A86F2D",
    "gold_soft": "#FFF4E4",
    "red": "#AE4D49",
    "red_soft": "#FCEFED",
    "grey": "#F3F5F7",
    "white": "#FFFFFF",
}


def wrap(value: str, width: int) -> str:
    return "\n".join(textwrap.wrap(value, width=width, break_long_words=False))


def panel_label(ax, x: float, y: float, label: str, title: str) -> None:
    ax.text(x, y, label, fontsize=10.5, fontweight="bold", color=COL["ink"], va="top")
    ax.text(
        x + 0.035,
        y,
        title,
        fontsize=10.0,
        fontweight="bold",
        color=COL["ink"],
        va="top",
    )


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    edge: str,
    face: str,
    body_width: int = 28,
    title_size: float = 7.5,
    body_size: float = 6.4,
    linewidth: float = 1.0,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            linewidth=linewidth,
            edgecolor=edge,
            facecolor=face,
        )
    )
    ax.text(
        x + 0.012,
        y + h - 0.020,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        x + 0.012,
        y + h - 0.055,
        wrap(body, body_width),
        ha="left",
        va="top",
        fontsize=body_size,
        linespacing=1.16,
        color=COL["ink"],
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COL["muted"],
    width: float = 1.15,
    dashed: bool = False,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=width,
            color=color,
            linestyle=(0, (3, 2)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def export(fig: plt.Figure) -> None:
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    fig.savefig(
        OUT.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    fig = plt.figure(figsize=(7.35, 5.55), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel_label(ax, 0.025, 0.975, "a", "The transferred object")

    # Three method families. The final row is intentionally wider because it is
    # the method deployed in the external confirmation protocol.
    box(
        ax,
        0.055,
        0.775,
        0.235,
        0.125,
        "H2O+-style dense residual",
        "absolute transition correction; unrestricted feature mixing",
        edge=COL["blue"],
        face=COL["blue_soft"],
        body_width=29,
    )
    box(
        ax,
        0.382,
        0.775,
        0.235,
        0.125,
        "MC-WM",
        "mechanism-factored dynamics; structured state prediction",
        edge=COL["teal"],
        face=COL["teal_soft"],
        body_width=29,
    )
    box(
        ax,
        0.709,
        0.775,
        0.235,
        0.125,
        "Anchored CFCMT",
        "matched action contrasts; constrained action ranking",
        edge=COL["green"],
        face=COL["green_soft"],
        body_width=29,
        linewidth=1.4,
    )
    arrow(ax, (0.292, 0.838), (0.380, 0.838), color=COL["line"])
    arrow(ax, (0.619, 0.838), (0.707, 0.838), color=COL["line"])
    ax.text(0.336, 0.858, "factor", ha="center", va="bottom", fontsize=6.2, color=COL["muted"])
    ax.text(0.663, 0.858, "rank", ha="center", va="bottom", fontsize=6.2, color=COL["muted"])

    ax.text(
        0.172,
        0.735,
        "Absolute residuals can carry\ndomain nuisance",
        fontsize=6.5,
        color=COL["red"],
        ha="center",
        va="top",
        linespacing=1.12,
    )
    ax.text(
        0.500,
        0.735,
        "Factorization alone does not fix\ntarget action ordering",
        fontsize=6.5,
        color=COL["teal"],
        ha="center",
        va="top",
        linespacing=1.12,
    )
    ax.text(
        0.827,
        0.735,
        "Matched contrasts cancel\naction-common bias",
        fontsize=6.5,
        color=COL["green"],
        ha="center",
        va="top",
        linespacing=1.12,
    )

    ax.plot([0.025, 0.975], [0.685, 0.685], color="#DDE2E8", lw=0.9)
    panel_label(ax, 0.025, 0.655, "b", "Frozen external target-adaptation and deployment protocol")

    # Inputs and counterfactual construction.
    box(
        ax,
        0.045,
        0.480,
        0.168,
        0.115,
        "Source bank",
        "18 networks, 7 city groups; frozen source cache",
        edge=COL["blue"],
        face=COL["blue_soft"],
        body_width=22,
    )
    box(
        ax,
        0.045,
        0.325,
        0.168,
        0.115,
        "Target adaptation",
        "uncalibrated SUMO; all valid groups from seeds 5057 and 6067",
        edge=COL["gold"],
        face=COL["gold_soft"],
        body_width=22,
    )
    box(
        ax,
        0.265,
        0.402,
        0.170,
        0.150,
        "Matched action groups",
        "restore one state, roll out every feasible phase, normalize interval cost by the within-group range",
        edge=COL["muted"],
        face=COL["grey"],
        body_width=23,
    )
    arrow(ax, (0.215, 0.535), (0.263, 0.500), color=COL["blue"])
    arrow(ax, (0.215, 0.382), (0.263, 0.452), color=COL["gold"])

    # Two complementary score models.
    box(
        ax,
        0.480,
        0.500,
        0.190,
        0.120,
        "Rigid anchor  A",
        "declared causal parents; candidate-only sign-balanced gradient boosting",
        edge=COL["blue"],
        face=COL["blue_soft"],
        body_width=25,
    )
    box(
        ax,
        0.480,
        0.340,
        0.190,
        0.120,
        "Pairwise correction  P",
        "all unordered phase pairs; antisymmetric augmentation by construction",
        edge=COL["green"],
        face=COL["green_soft"],
        body_width=25,
    )
    arrow(ax, (0.437, 0.485), (0.478, 0.555), color=COL["blue"])
    arrow(ax, (0.437, 0.458), (0.478, 0.400), color=COL["green"])

    # Selector and deployable score.
    box(
        ax,
        0.715,
        0.448,
        0.240,
        0.150,
        "Seed-blocked selector",
        "leave one complete adaptation seed out; select constant or local confidence-capped alpha; refit once on all adaptation groups",
        edge=COL["green"],
        face=COL["green_soft"],
        body_width=31,
        linewidth=1.3,
    )
    arrow(ax, (0.672, 0.560), (0.713, 0.548), color=COL["blue"])
    arrow(ax, (0.672, 0.402), (0.713, 0.485), color=COL["green"])

    ax.text(
        0.835,
        0.408,
        r"$S(s,a)=A(s,a)+\alpha(s)\,[P(s,a)-A(s,a)]$",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COL["ink"],
    )
    ax.text(
        0.835,
        0.376,
        "Los Angeles: local cap 1, z=0.25   |   Jinan: alpha=0.9",
        ha="center",
        va="center",
        fontsize=6.1,
        color=COL["muted"],
    )

    # Strictly separated evaluation stage.
    ax.add_patch(
        FancyBboxPatch(
            (0.045, 0.075),
            0.910,
            0.180,
            boxstyle="round,pad=0.008,rounding_size=0.010",
            linewidth=0.9,
            edgecolor=COL["line"],
            facecolor=COL["white"],
        )
    )
    ax.text(
        0.060,
        0.235,
        "Untouched confirmation",
        fontsize=7.6,
        color=COL["ink"],
        fontweight="bold",
        va="top",
    )
    box(
        ax,
        0.070,
        0.105,
        0.205,
        0.090,
        "Offline seed 8171",
        "action regret; no refit or reselection",
        edge=COL["muted"],
        face=COL["grey"],
        body_width=27,
        title_size=7.1,
        body_size=6.1,
    )
    box(
        ax,
        0.315,
        0.105,
        0.205,
        0.090,
        "Closed loop 8081/9091",
        "3,600 s; safe phase-transition executor",
        edge=COL["green"],
        face=COL["green_soft"],
        body_width=27,
        title_size=7.1,
        body_size=6.1,
    )
    box(
        ax,
        0.560,
        0.105,
        0.180,
        0.090,
        "Transfer baselines",
        "dense | simulator-only | rigid",
        edge=COL["blue"],
        face=COL["blue_soft"],
        body_width=24,
        title_size=7.1,
        body_size=5.9,
    )
    box(
        ax,
        0.780,
        0.105,
        0.150,
        0.090,
        "Rule comparators",
        "MaxPressure | phase pressure",
        edge=COL["gold"],
        face=COL["gold_soft"],
        body_width=20,
        title_size=7.1,
        body_size=5.7,
    )
    arrow(ax, (0.835, 0.340), (0.835, 0.258), color=COL["green"], width=1.4)
    arrow(ax, (0.835, 0.258), (0.418, 0.198), color=COL["green"], rad=-0.10)
    arrow(ax, (0.350, 0.400), (0.175, 0.198), color=COL["muted"], dashed=True, rad=0.16)

    ax.text(
        0.500,
        0.025,
        "Target transition labels are used only in offline adaptation; this is not zero-shot or field causal identification.",
        ha="center",
        va="center",
        fontsize=6.4,
        color=COL["red"],
        fontweight="bold",
    )

    export(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
