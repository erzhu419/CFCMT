from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def add_box(ax, x, y, w, h, text, fc, ec, fs=7.2, lw=1.15):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#1F2937",
        linespacing=1.18,
    )
    return patch


def add_arrow(ax, x1, y1, x2, y2, label=None, rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.15,
        color="#4B5563",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=5,
        shrinkB=5,
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            label,
            ha="center",
            va="center",
            fontsize=6.4,
            color="#4B5563",
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.12", alpha=0.9),
        )


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue = ("#EAF4FB", "#2F6F9F")
    green = ("#EAF7F1", "#2D7D59")
    grey = ("#F2F4F7", "#667085")
    orange = ("#FFF5E6", "#B26A00")
    red = ("#FFF1F3", "#B42318")

    ax.text(
        0.5,
        0.965,
        "H2O/H2O+ framework and the bus-holding instantiation",
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color="#111827",
    )

    # Panel labels.
    ax.text(0.04, 0.845, "Original H2O/H2O+ template", fontsize=8.4, fontweight="bold", color="#344054")
    ax.text(0.04, 0.432, "This paper: transit holding under a SUMO/sim-core fidelity gap", fontsize=8.4, fontweight="bold", color="#344054")
    ax.plot([0.035, 0.965], [0.505, 0.505], color="#D0D5DD", lw=0.9)

    # Top row: original H2O/H2O+.
    add_box(ax, 0.045, 0.640, 0.145, 0.135, "Target-domain\noffline data\n$D_{real}$", *blue)
    add_box(ax, 0.045, 0.515, 0.145, 0.095, "Imperfect\nsimulator\n$M_{sim}$", *blue, fs=6.9)
    add_box(ax, 0.255, 0.585, 0.165, 0.145, "Offline warm start\nactor-critic /\noffline RL", *green)
    add_box(ax, 0.485, 0.585, 0.175, 0.145, "Dynamics-gap\nestimator\n(discriminator,\npenalty/IS)", *green, fs=6.7)
    add_box(ax, 0.725, 0.585, 0.205, 0.145, "Hybrid policy update\nmix offline + sim data\ntrust simulator where\ngap is small", *green, fs=6.55)

    add_arrow(ax, 0.190, 0.705, 0.255, 0.670, "pretrain")
    add_arrow(ax, 0.190, 0.560, 0.255, 0.640, "rollout")
    add_arrow(ax, 0.420, 0.660, 0.485, 0.660, "compare")
    add_arrow(ax, 0.660, 0.660, 0.725, 0.660, "weight")

    # Bottom row: this paper.
    add_box(ax, 0.045, 0.270, 0.145, 0.105, "SUMO target\n$M_{real}$\n12 lines, 389 buses", *blue, fs=6.7)
    add_box(ax, 0.045, 0.145, 0.145, 0.095, "$D_{off}$: 675K\nSAC + heuristic\n+ random + ep39", *blue, fs=6.55)
    add_box(ax, 0.255, 0.265, 0.165, 0.125, "Stage 1\nE=5 Q ensemble\nSAC actor\nLCB pretrain", *green, fs=6.55)
    add_box(ax, 0.255, 0.115, 0.165, 0.115, "Stage 2\nsim-core or\nSUMO oracle\nhybrid replay", *green, fs=6.55)
    add_box(ax, 0.485, 0.255, 0.175, 0.135, "H2O+ variants\nTransDisc /\nDynamicsDisc /\nContrastive IS\n+ optional Q-floor", *green, fs=6.25)
    add_box(ax, 0.485, 0.115, 0.175, 0.105, "Baselines\nRLPD, WSRL,\nTD3+BC, IQL,\nAWAC, BC,\nDaganzo", *grey, fs=6.1)
    add_box(ax, 0.725, 0.275, 0.205, 0.115, "Paired evaluation\n1 default scenario\n+ 3 x 3 stress grid", *orange, fs=6.55)
    add_box(ax, 0.725, 0.100, 0.205, 0.125, "Claim boundary\nrecovers most ep39;\nbeats unlearned;\nnot dominant vs\nWSRL/RLPD/ep39", *red, fs=6.1)

    add_arrow(ax, 0.118, 0.270, 0.118, 0.240, "collect")
    add_arrow(ax, 0.190, 0.190, 0.255, 0.325, "offline")
    add_arrow(ax, 0.190, 0.190, 0.255, 0.175, "reset")
    add_arrow(ax, 0.420, 0.330, 0.485, 0.325, "ablate")
    add_arrow(ax, 0.420, 0.170, 0.485, 0.165, "compare")
    add_arrow(ax, 0.660, 0.325, 0.725, 0.335, "evaluate")
    add_arrow(ax, 0.660, 0.165, 0.725, 0.165, "rank")
    add_arrow(ax, 0.828, 0.275, 0.828, 0.225, "metrics")

    # Adaptation cues between rows.
    add_arrow(ax, 0.565, 0.585, 0.565, 0.390, "adapted correction")
    add_arrow(ax, 0.828, 0.585, 0.828, 0.390, "SUMO eval")

    ax.text(
        0.5,
        0.045,
        "Top row follows the H2O/H2O+ abstraction; bottom row shows the paper's implementation and empirical comparison.",
        ha="center",
        va="center",
        fontsize=6.6,
        color="#475467",
    )

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT_DIR / "framework_overview.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "framework_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
