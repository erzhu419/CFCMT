"""Generate publication-grade IEEE T-ITS vector PDF + PNG figures from exact JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import ScalarFormatter
import numpy as np

from paired_comparisons import EXP_DIR, RL_RESULT_FILE, summarize

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Colorblind-safe Okabe-Ito / IEEE palette
C_CFCMT = "#0072B2"
C_ZS = "#56B4E9"
C_CONS = "#009E73"
C_MAXRED = "#2B8CBE"
C_TO = "#E69F00"
C_PLAC = "#CC79A7"
C_RL = "#D55E00"
C_FIXED = "#7F7F7F"
C_OBS = "#B22222"
C_ORACLE = "#222222"


def save_both(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_fig1_framework(out_dir: Path) -> None:
    """Fig. 1: Architectural overview of CFCMT under sparse CV partial observability."""
    fig, ax = plt.subplots(figsize=(12.4, 3.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc, ec, title_c="#111111"):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.35,rounding_size=0.9",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.25,
        )
        ax.add_patch(p)
        ax.text(
            x + w / 2,
            y + h - 1.7,
            title,
            ha="center",
            va="top",
            fontsize=7.7,
            fontweight="bold",
            color=title_c,
        )
        body = "\n".join(lines)
        ax.text(
            x + w / 2,
            y + (h - 3.3) / 2,
            body,
            ha="center",
            va="center",
            fontsize=6.7,
            color="#222222",
            linespacing=1.28,
        )

    def arrow(x1, y1, x2, y2, color="#444444"):
        arr = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>,head_length=4.5,head_width=2.6",
            linewidth=1.1,
            color=color,
        )
        ax.add_patch(arr)

    # Column 1: Causal Partial-CV Sensor
    box(
        0.8,
        4.0,
        22.5,
        28.5,
        "(a) CV Inputs for Phase Scoring",
        [
            r"Trip-persistent CV sample ($p_{\mathrm{CV}}=10\%$)",
            "Phase scoring uses CV observations",
            "1-Hz causal lane state:",
            r" • Red shockwave tail $d_{\max}/L_{\mathrm{jam}}$",
            r" • Arrival-conservation queue $\hat{q}_{\mathrm{cons}}$",
            " • Hidden quadratic red wait integral",
            " • 1-hop upstream/downstream coupling",
        ],
        "#EEF6FB",
        "#2B6CB0",
        "#0B4F8A",
    )

    # Column 2: Zero-Restore Counterfactuals
    box(
        26.3,
        18.6,
        23.5,
        13.9,
        "(b) Native-Prefix Source Library",
        [
            "6 multi-city domains (411 groups)",
            "Fresh run for each branch, no state restore",
            "60-s local cost contrast",
        ],
        "#F0F9F4",
        "#1E7E5A",
        "#135D41",
    )
    box(
        26.3,
        4.0,
        23.5,
        12.8,
        "Few-Shot Target Rollouts",
        [
            "B in {0, 6, 36} on seeds (5057, 6067)",
            "Matching pre-branch summaries",
            "Group-centered, span-normalized costs",
        ],
        "#FFF8EB",
        "#C87D0A",
        "#8F5700",
    )

    # Column 3: Matched Ranker + Calibration
    box(
        52.8,
        4.0,
        23.0,
        28.5,
        "(c) Calibrated Pairwise Transfer",
        [
            "78-D endpoint-antisymmetric input:",
            r"$[x_g,\phi_i-\phi_j,\frac{1}{2}(\phi_i+\phi_j)]$",
            "Borda aggregation of pairwise costs",
            "Group-aware leaf capacity cap",
            r"Structural pool, two-fold choice of $\omega^\star$",
            r"Fold-to-full RMS shrinkage $\tau_{\mathrm{cal}}$",
            r"$(1-\omega^\star)\tau_{\mathrm{cal}}\widehat{T}_g+\omega^\star\widehat{S}_g$",
        ],
        "#F4F3FB",
        "#5A4FCF",
        "#382EA8",
    )

    # Column 4: Online Shielded Execution
    box(
        78.8,
        4.0,
        20.4,
        28.5,
        "(d) Online Phase Selection",
        [
            "Frozen scores, no online rollouts",
            "Demand-confirmed max-red priority",
            " 1) CV present on lane",
            " 2) Arrival flow and estimated queue",
            "Shared clearance checks all vehicles",
            "10-s phase decision interval",
        ],
        "#FDF2F2",
        "#B83232",
        "#8A1C1C",
    )

    arrow(23.3, 24.5, 26.3, 24.5)
    arrow(23.3, 10.8, 26.3, 10.8)
    arrow(49.8, 24.5, 52.8, 21.5)
    arrow(49.8, 10.8, 52.8, 14.5)
    arrow(75.8, 18.2, 78.8, 18.2)

    ax.text(
        50.0,
        1.2,
        "Source seeds (2027, 3037)  |  Adaptation (5057, 6067)  |  Development (7077, 8087)  |  Held-out test (920101–920110 / 930101–930110)",
        ha="center",
        va="center",
        fontsize=7.3,
        style="italic",
        color="#333333",
    )
    save_both(fig, out_dir, "fig1_framework")


def draw_fig2_distributions(exp_dir: Path, out_dir: Path) -> None:
    """Fig. 2: Held-out waiting-time distributions in three cities."""
    p2 = json.loads((exp_dir / "phase2_eval_fresh_result.json").read_text())["rows"]
    sl = json.loads((exp_dir / "phase3_saltlake_eval_fresh_result.json").read_text())["rows"]
    rl = json.loads((exp_dir / RL_RESULT_FILE).read_text())["rows"]
    rl.extend(json.loads((exp_dir / "phase3_rl_source_only_result.json").read_text())["rows"])

    spec = [
        ("cfcmt_b6", "CFCMT (B=6)", C_CFCMT),
        ("zero_shot_cfcmt_b0", "Zero-Shot CFCMT", C_ZS),
        ("transfer_mplight_cv_b0", "Trans-MPLight-CV\n(source only)", "#853D0C"),
        ("cfcmt_b36", "CFCMT (B=36)", "#1D4ED8"),
        ("maxred_conservation_gp_10", "MaxRed-Cons-GP", C_MAXRED),
        ("conservation_gp_10", "Cons-GP", C_CONS),
        ("transfer_mplight_cv_ep2", "Trans-MPLight-CV (ep2)", C_RL),
        ("mplight_cv_ep16", "MPLight-CV (ep16)", "#F97316"),
        ("target_only_b6", "Target-Only (B=6)", C_TO),
        ("source_placebo_b6", "Source-Placebo (B=6)", C_PLAC),
        ("fixed_program", "Fixed-Time", C_FIXED),
    ]

    families = [
        ("la", "Los Angeles (la_1x4)", p2, sorted({int(r["seed"]) for r in p2})),
        ("jinan", "Jinan Grid (3 Regimes)", p2, sorted({int(r["seed"]) for r in p2})),
        ("saltlake", "Salt Lake City (2 Corridors)", sl, sorted({int(r["seed"]) for r in sl})),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.20), sharey=True)
    for ax_i, (ax, (fam, title, base_rows, seeds)) in enumerate(zip(axes, families)):
        y_pos = np.arange(len(spec))[::-1]
        means = []
        errs_lo = []
        errs_hi = []
        colors = []
        for idx, (pol, label, col) in enumerate(spec):
            src = rl if (pol.startswith("mplight") or pol.startswith("transfer_")) else base_rows
            vals = np.array(
                [
                    np.mean([r["mean_waiting_sec"] for r in src if r["family"] == fam and r["policy"] == pol and int(r["seed"]) == s])
                    for s in seeds
                ]
            )
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
            means.append(m)
            errs_lo.append(min(m, 1.96 * sd))
            errs_hi.append(1.96 * sd)
            colors.append(col)
            # Jittered seed dots
            rng = np.random.default_rng(100 + ax_i * 20 + idx)
            jit = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(vals, np.full_like(vals, y_pos[idx]) + jit, s=9, color=col, alpha=0.45, zorder=3, edgecolors="none")

        ax.barh(y_pos, means, xerr=[errs_lo, errs_hi], height=0.58, color=colors, alpha=0.28, edgecolor=colors, linewidth=1.0, capsize=2.0, zorder=2)
        ax.scatter(means, y_pos, s=22, color=colors, zorder=4, edgecolors="white", linewidth=0.5)
        ax.set_title(f"({chr(97 + ax_i)}) {title}", pad=5, fontsize=8.8)
        ax.set_xlabel("Mean Waiting Time (s) ↓")
        ax.set_xlim(left=0)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        if ax_i == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([x[1] for x in spec])

    fig.tight_layout(w_pad=1.35)
    save_both(fig, out_dir, "fig2_benchmark_distributions")


def draw_fig3_ablation(exp_dir: Path, out_dir: Path) -> None:
    """Fig. 3: Archived adaptation choices and held-out vehicle-hour comparisons."""
    p2 = json.loads((exp_dir / "phase2_offline_fit_audit.json").read_text())["fit_audit"]
    sl = json.loads((exp_dir / "phase3_saltlake_fit_audit.json").read_text())["fit_audit"]
    budgets = [
        (f"{city}, B={budget}", audit["budgets"][f"b{budget}"])
        for city, audit in [("LA", p2["la"]), ("Jinan", p2["jinan"]), ("Salt Lake", sl["saltlake"])]
        for budget in (6, 36)
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.75))
    pos = np.arange(len(budgets))[::-1]
    target = [b["oof_target_only"]["mean_normalized_regret"] for _, b in budgets]
    transfer = [b["oof_cfcmt"]["mean_normalized_regret"] for _, b in budgets]
    ax1.hlines(pos, transfer, target, color="#B5BDC3", linewidth=1)
    ax1.scatter(target, pos, label="Target-only", color=C_TO, marker="s", s=25)
    ax1.scatter(transfer, pos, label="CFCMT", color=C_CFCMT, s=17)
    for y, (_, budget) in zip(pos, budgets):
        ax1.text(0.422, y, f"ω*={budget['cfcmt_omega']:.2f}", va="center", fontsize=7.1)
    ax1.set_yticks(pos, [name for name, _ in budgets])
    ax1.set_xlim(0.025, 0.513)
    ax1.set_xticks([0.05, 0.15, 0.25, 0.35])
    ax1.set_ylim(-0.65, len(budgets) + 0.2)
    ax1.set_xlabel("Out-of-fold normalized regret ↓")
    ax1.set_title("(a) Source Weight Selection")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", ncol=2, frameon=False)

    comps = [
        ("conservation_gp_10", "vs Cons-GP", C_CONS),
        ("maxred_conservation_gp_10", "vs MaxRed-Cons-GP", C_MAXRED),
        ("target_only_b6", "vs Target-Only (B=6)", C_TO),
        ("source_placebo_b6", "vs Source-Placebo (B=6)", C_PLAC),
        ("transfer_mplight_cv_ep2", "vs Trans-MPLight-CV", C_RL),
        ("mplight_cv_ep16", "vs MPLight-CV (ep16)", "#F97316"),
        ("fixed_program", "vs Fixed-Time", C_FIXED),
    ]
    y_idx = np.arange(len(comps))[::-1]
    results = summarize(exp_dir)
    comparisons = [results[policy]["system_vehicle_hours"] for policy, _, _ in comps]
    means = np.array([result["mean"] for result in comparisons])
    low, high = np.array([result["interval"] for result in comparisons]).T
    cis = [means - low, high - means]
    cols = [c[2] for c in comps]

    ax2.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax2.barh(y_idx, means, xerr=cis, height=0.55, color=cols, alpha=0.35, edgecolor=cols, linewidth=1.1, capsize=2.2)
    ax2.scatter(means, y_idx, color=cols, s=20, zorder=4)
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels([c[1] for c in comps])
    ax2.set_xlabel("CFCMT (B=6) − baseline (veh-h) ↓")
    ax2.set_title("(b) Three-City Differences and 95% CIs")
    ax2.set_xlim(right=20)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout(w_pad=1.4)
    save_both(fig, out_dir, "fig3_ablation_mechanisms")


def draw_fig4_penetration(exp_dir: Path, out_dir: Path) -> None:
    """Fig. 4: Zero-retraining CV penetration rate sensitivity curve (5% to 100%)."""
    pen = json.loads((exp_dir / "phase3_penetration_curve_result.json").read_text())
    rates = [float(x) for x in pen["penetration_rates"]]
    x_idx = np.arange(len(rates))
    x_labels = [f"{int(round(r * 100))}%" for r in rates]

    pol_specs = [
        ("cfcmt_b6", "CFCMT (B=6)", C_CFCMT, "-o"),
        ("zero_shot_cfcmt_b0", "CFCMT (B=0)", C_ZS, "--^"),
        ("cfcmt_b36", "CFCMT (B=36)", "#1D4ED8", "-.s"),
        ("maxred_conservation_gp_10", "MaxRed-Cons-GP", C_MAXRED, "-D"),
        ("conservation_gp_10", "Cons-GP", C_CONS, "--v"),
        ("observed_mp_10", "Observed-MP (raw CV)", C_OBS, ":x"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))
    for ax, (scen, title, use_log) in zip(
        (ax1, ax2),
        (
            ("la_1x4", "(a) Los Angeles Arterial (la_1x4)", True),
            ("jinan_3x4_real", "(b) Jinan 8-Phase Grid (jinan_3x4_real)", False),
        ),
    ):
        smap = pen["curve_summary"][scen]
        for pol, label, col, style in pol_specs:
            if scen == "la_1x4" and pol == "zero_shot_cfcmt_b0":
                continue
            if scen == "la_1x4" and pol == "cfcmt_b6":
                label = "CFCMT (B=0 and B=6)"
            ys = [smap[f"{r:.2f}"][pol]["mean_waiting_sec"] for r in rates]
            ax.plot(x_idx, ys, style, color=col, linewidth=1.5, markersize=4.5, label=label)
        if use_log:
            ax.set_yscale("log")
            ax.set_yticks([200, 500, 1000, 2000])
            ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(r"Connected-vehicle penetration $p_{\mathrm{CV}}$")
        ax.set_ylabel("Mean Waiting Time (s) ↓")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.32)
        ax.legend(loc="upper right", fontsize=6.9, frameon=True, facecolor="white", framealpha=0.88)

    fig.tight_layout(w_pad=1.3)
    save_both(fig, out_dir, "fig4_penetration_curve")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "figures"
    draw_fig1_framework(out_dir)
    draw_fig2_distributions(EXP_DIR, out_dir)
    draw_fig3_ablation(EXP_DIR, out_dir)
    draw_fig4_penetration(EXP_DIR, out_dir)
    print("ALL_FOUR_TITS_FIGURES_GENERATED", out_dir)


if __name__ == "__main__":
    main()
