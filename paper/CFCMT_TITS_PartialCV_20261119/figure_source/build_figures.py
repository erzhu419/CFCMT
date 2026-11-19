"""Generate publication-grade IEEE T-ITS vector PDF + PNG figures from exact JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

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
        "(a) Strict Non-Leaking CV Sensor",
        [
            "Persistent CV hash: p_cv = 10%",
            "Unequipped HDVs strictly hidden",
            "1-Hz causal lane state:",
            " • Red shockwave tail d_max / L_jam",
            " • Arrival-conservation queue q_cons",
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
            "Zero state_restores (exact replay)",
            "1-hop light-cone 60-s delay contrast",
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
            "Bit-exact pre-branch verification",
            "Normalized contrast z_g(a) = c_g/sigma_g",
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
            " [x_g, phi_i - phi_j, 0.5(phi_i + phi_j)]",
            "Exact Borda cost recovery P_g(a)",
            "Group-aware leaf capacity cap",
            "2-fold OOF source-pool & omega* gate",
            "Fold-to-full RMS parity factor tau_cal:",
            " (1 - omega*) tau_cal T_g + omega* S_g",
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
        "(d) Shielded Online Control",
        [
            "Frozen predictor (0 rollouts)",
            "Two-clause NEMA Max-Red:",
            " 1) Confirmed waiting CV cap",
            " 2) Flow-confirmed drought cap",
            "Dead-pocket false-trigger immune",
            "Shared yellow/all-red executor",
            "Delta t = 10 s closed-loop step",
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
        "Strict 4-way seed partition: Source (2027, 3037)  |  Target Adaptation (5057, 6067)  |  Dev Check (7077, 8087)  |  Single-Shot Virgin Test (920101–920110 / 930101–930110)",
        ha="center",
        va="center",
        fontsize=7.3,
        style="italic",
        color="#333333",
    )
    save_both(fig, out_dir, "fig1_framework")


def draw_fig2_distributions(exp_dir: Path, out_dir: Path) -> None:
    """Fig. 2: 3-city virgin-seed closed-loop waiting time & vehicle-hours comparison."""
    p2 = json.loads((exp_dir / "phase2_eval_fresh_result.json").read_text())["rows"]
    sl = json.loads((exp_dir / "phase3_saltlake_eval_fresh_result.json").read_text())["rows"]
    rl = json.loads((exp_dir / "phase3_rl_baselines_result.json").read_text())["rows"]

    spec = [
        ("cfcmt_b6", "CFCMT (B=6)", C_CFCMT),
        ("zero_shot_cfcmt_b0", "Zero-Shot CFCMT", C_ZS),
        ("cfcmt_b36", "CFCMT (B=36)", "#1D4ED8"),
        ("maxred_conservation_gp_10", "MaxRed-Cons-GP", C_MAXRED),
        ("conservation_gp_10", "Cons-GP", C_CONS),
        ("transfer_mplight_cv_ep2", "Trans-MPLight-CV", C_RL),
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

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.85), sharey=True)
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
            ax.set_xlim(0, min(max(means) * 1.18, 1450))

    fig.tight_layout(w_pad=1.35)
    save_both(fig, out_dir, "fig2_benchmark_distributions")


def draw_fig3_ablation(exp_dir: Path, out_dir: Path) -> None:
    """Fig. 3: (a) 2-Fold OOF convex blending U-curves; (b) 3-City paired vehicle-hour & waiting gains."""
    from experiments.tsc_partial_cv_pilot_20261118.phase2_residual_transfer import (
        ADAPT_SEEDS,
        _dict_to_rec,
        _eval_precomputed_scores,
        _to_action_group,
        fit_phase2_residual_suite,
        subset_residual_budget,
    )
    from experiments.tsc_partial_cv_pilot_20261118.cfcmt_cv_model import EndpointAntisymmetricActionRanker

    ds = json.loads((exp_dir / "phase2_augmented_dataset.json").read_text())
    src = [_dict_to_rec(x) for x in ds["source_groups"]]
    tgt = {k: [_dict_to_rec(x) for x in v] for k, v in ds["target_groups"].items()}
    models, _ = fit_phase2_residual_suite(src, tgt)

    omegas = np.linspace(0.0, 1.0, 21)
    curves = {}
    for fam, pool_key in (("la", "zero_shot_cfcmt_b0"), ("jinan", "cfcmt_b36")):
        sub = subset_residual_budget(tgt[fam], 18)
        act = [_to_action_group(g) for g in sub]
        s0, s1 = ADAPT_SEEDS
        a0 = [g for g in act if g.seed == s0]
        a1 = [g for g in act if g.seed == s1]
        f0 = EndpointAntisymmetricActionRanker(l2_regularization=15.0, random_state=42).fit(a0)
        f1 = EndpointAntisymmetricActionRanker(l2_regularization=15.0, random_state=42).fit(a1)
        val = a1 + a0
        t_oof = [f0.score_group(g.state_vec, g.action_mat) for g in a1] + [f1.score_group(g.state_vec, g.action_mat) for g in a0]
        srk = models[fam][pool_key].source_ranker
        p_sc = [srk.score_group(g.state_vec, g.action_mat) for g in val]
        curves[fam] = [
            _eval_precomputed_scores(val, [(1.0 - om) * ts + om * ps for ts, ps in zip(t_oof, p_sc)])["mean_normalized_regret"]
            for om in omegas
        ]

    p2 = json.loads((exp_dir / "phase2_eval_fresh_result.json").read_text())["rows"]
    sl = json.loads((exp_dir / "phase3_saltlake_eval_fresh_result.json").read_text())["rows"]
    rl = json.loads((exp_dir / "phase3_rl_baselines_result.json").read_text())["rows"]
    p2_seeds = sorted({int(r["seed"]) for r in p2})
    sl_seeds = sorted({int(r["seed"]) for r in sl})

    def comp3_vehh(pol: str) -> np.ndarray:
        s_la = rl if (pol.startswith("mplight") or pol.startswith("transfer_")) else p2
        s_jn = rl if (pol.startswith("mplight") or pol.startswith("transfer_")) else p2
        s_sl = rl if (pol.startswith("mplight") or pol.startswith("transfer_")) else sl
        v_la = np.array([np.mean([r["system_vehicle_hours"] for r in s_la if r["family"] == "la" and r["policy"] == pol and int(r["seed"]) == s]) for s in p2_seeds])
        v_jn = np.array([np.mean([r["system_vehicle_hours"] for r in s_jn if r["family"] == "jinan" and r["policy"] == pol and int(r["seed"]) == s]) for s in p2_seeds])
        v_sl = np.array([np.mean([r["system_vehicle_hours"] for r in s_sl if r["family"] == "saltlake" and r["policy"] == pol and int(r["seed"]) == s]) for s in sl_seeds])
        return (v_la + v_jn + v_sl) / 3.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.55))

    ax1.plot(omegas, curves["la"], "-o", markersize=3.2, color=C_CFCMT, linewidth=1.5, label="Los Angeles (real_cities, ω*=0.75)")
    ax1.plot(omegas, curves["jinan"], "-s", markersize=3.2, color=C_CONS, linewidth=1.5, label="Jinan (multi_phase_grids, ω*=0.25)")
    ax1.axvline(0.75, color=C_CFCMT, linestyle=":", alpha=0.75)
    ax1.axvline(0.25, color=C_CONS, linestyle=":", alpha=0.75)
    ax1.set_xlabel("Source Prior Weight ω")
    ax1.set_ylabel("2-Fold OOF Normalized Regret ↓")
    ax1.set_title("(a) Convex Source–Target Regret Curves (B=36)")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.85)

    cf_v = comp3_vehh("cfcmt_b6")
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
    diffs = [cf_v - comp3_vehh(c[0]) for c in comps]
    means = [float(np.mean(d)) for d in diffs]
    cis = [1.96 * float(np.std(d, ddof=1)) / np.sqrt(len(d)) for d in diffs]
    cols = [c[2] for c in comps]

    ax2.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax2.barh(y_idx, means, xerr=cis, height=0.55, color=cols, alpha=0.35, edgecolor=cols, linewidth=1.1, capsize=2.2)
    ax2.scatter(means, y_idx, color=cols, s=20, zorder=4)
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels([c[1] for c in comps])
    ax2.set_xlabel("3-City Paired Δ System Vehicle-Hours (CFCMT B=6 − Base) ↓")
    ax2.set_title("(b) Paired Vehicle-Hour Reductions (N=10 Virgin Seeds)")
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
        ("cfcmt_b6", "CFCMT (B=6, trained @ 10%)", C_CFCMT, "-o"),
        ("zero_shot_cfcmt_b0", "Zero-Shot CFCMT (trained @ 10%)", C_ZS, "--^"),
        ("cfcmt_b36", "CFCMT (B=36, trained @ 10%)", "#1D4ED8", "-.s"),
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
            ys = [smap[f"{r:.2f}"][pol]["mean_waiting_sec"] for r in rates]
            ax.plot(x_idx, ys, style, color=col, linewidth=1.5, markersize=4.5, label=label)
        if use_log:
            ax.set_yscale("log")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Connected-Vehicle Penetration Rate p_CV")
        ax.set_ylabel("Mean Waiting Time (s) ↓")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.32)
        if scen == "la_1x4":
            ax.legend(loc="upper right", fontsize=6.9, frameon=True, facecolor="white", framealpha=0.88)

    fig.tight_layout(w_pad=1.3)
    save_both(fig, out_dir, "fig4_penetration_curve")


def main() -> None:
    repo = Path("/home/erzhu419/mine_code/CFCMT")
    exp_dir = repo / "experiments/tsc_partial_cv_pilot_20261118"
    out_dir = repo / "paper/CFCMT_TITS_PartialCV_20261119/figures"
    draw_fig1_framework(out_dir)
    draw_fig2_distributions(exp_dir, out_dir)
    draw_fig3_ablation(exp_dir, out_dir)
    draw_fig4_penetration(exp_dir, out_dir)
    print("ALL_FOUR_TITS_FIGURES_GENERATED", out_dir)


if __name__ == "__main__":
    main()
