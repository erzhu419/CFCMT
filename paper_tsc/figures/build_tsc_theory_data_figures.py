from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import textwrap
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TABLE_DIR = REPO / "paper_tsc" / "tables"
SOURCE_DATA_DIR = REPO / "paper_tsc" / "source_data"
DEFAULT_MANIFEST = REPO / "cf_h2o" / "config" / "traffic_signal_cross_city_v1.json"
DEFAULT_RESULT = (
    REPO
    / "cf_h2o"
    / "results"
    / "cluster"
    / "tsc_v16r12_target_hierarchy_dev600_20260808"
    / "development"
    / "budget_000"
    / "result.json"
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.1,
        "legend.frameon": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


COL = {
    "ink": "#242A31",
    "muted": "#66717E",
    "line": "#B7C0CC",
    "blue": "#244E80",
    "blue_soft": "#EEF4FB",
    "teal": "#2E7C82",
    "teal_soft": "#ECF8F8",
    "green": "#2D7F50",
    "green_soft": "#EEF8EF",
    "gold": "#B2762C",
    "gold_soft": "#FFF6E8",
    "red": "#B34A43",
    "red_soft": "#FFF0EE",
    "grey": "#F5F6F8",
}


def wrap(text: str, width: int) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(line, width=width, break_long_words=False))
    return "\n".join(lines)


def box(ax, xy, w, h, title, body, fc, ec, body_width=34, title_size=7.8, body_size=6.4):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + 0.014, y + h - 0.026, title, ha="left", va="top", color=ec, fontsize=title_size, fontweight="bold")
    ax.text(x + 0.014, y + h - 0.062, wrap(body, body_width), ha="left", va="top", color=COL["ink"], fontsize=body_size, linespacing=1.16)


def arrow(ax, start, end, color=COL["muted"], lw=1.2, rad=0.0, dashed=False):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (3, 2)) if dashed else "solid",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)


def export(fig, stem: str) -> None:
    out = HERE / stem
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out}.tiff", dpi=600, bbox_inches="tight")
    print(out)


def latex_escape(text: object) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def parse_shape(shape: str) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for item in shape.split():
        x, y = item.split(",")[:2]
        pts.append((float(x), float(y)))
    return pts


def parse_net(entry: dict[str, object], manifest_path: Path) -> dict[str, object]:
    scenario = str(entry["scenario"])
    sumocfg_path = (manifest_path.parent / str(entry["sumocfg"])).resolve()
    sumocfg_root = ET.parse(sumocfg_path).getroot()
    net_element = sumocfg_root.find("./input/net-file")
    if net_element is None or not net_element.get("value"):
        raise ValueError(f"SUMO config has no net-file: {sumocfg_path}")
    net_path = (sumocfg_path.parent / str(net_element.get("value"))).resolve()
    if not net_path.exists():
        raise FileNotFoundError(f"missing SUMO net file: {net_path}")
    root = ET.parse(net_path).getroot()
    edges: list[list[tuple[float, float]]] = []
    lane_count = 0
    edge_count = 0
    for edge in root.findall("edge"):
        if edge.get("function") == "internal" or (edge.get("id") or "").startswith(":"):
            continue
        lanes = edge.findall("lane")
        if not lanes:
            continue
        edge_count += 1
        lane_count += len(lanes)
        shape = lanes[0].get("shape")
        if shape:
            edges.append(parse_shape(shape))
    signals: list[tuple[float, float]] = []
    for junction in root.findall("junction"):
        jtype = junction.get("type") or ""
        if "traffic_light" in jtype:
            signals.append((float(junction.get("x", "0")), float(junction.get("y", "0"))))
    tl_count = len(root.findall("tlLogic"))
    green_phase_count = 0
    for logic in root.findall("tlLogic"):
        for phase in logic.findall("phase"):
            state = phase.get("state") or ""
            if "G" in state or "g" in state:
                green_phase_count += 1
    points = [p for shape in edges for p in shape] + signals
    return {
        "scenario": scenario,
        "city_group": str(entry["city_group"]),
        "suite": str(entry["suite"]),
        "provenance": str(entry["provenance"]),
        "sumocfg": sumocfg_path,
        "path": net_path,
        "edges": edges,
        "signals": signals,
        "edge_count": edge_count,
        "lane_count": lane_count,
        "signal_count": len(signals),
        "tl_logic_count": tl_count,
        "green_phase_count": green_phase_count,
        "bbox": (
            min(x for x, _ in points),
            min(y for _, y in points),
            max(x for x, _ in points),
            max(y for _, y in points),
        ),
    }


SCENARIO_LABELS = {
    "grid4x4": "grid4x4",
    "arterial4x4": "arterial4x4",
    "cologne1": "cologne1",
    "cologne3": "cologne3",
    "cologne8": "cologne8",
    "ingolstadt1": "ingolstadt1",
    "ingolstadt7": "ingolstadt7",
    "ingolstadt21": "ingolstadt21",
    "atlanta_1x5": "Atlanta 1x5",
    "hangzhou_bc_tyc": "Hangzhou bc-tyc",
    "hangzhou_kn_hz": "Hangzhou kn-hz",
    "hangzhou_qc_yn": "Hangzhou qc-yn",
    "hangzhou_sb_sx": "Hangzhou sb-sx",
    "hangzhou_4x4": "Hangzhou 4x4",
    "hangzhou_4x4_hetero": "Hangzhou 4x4 hetero",
    "manhattan_28x7": "Manhattan 28x7",
}


CITY_LABELS = {
    "resco_synthetic": "Synthetic",
    "cologne": "Cologne",
    "ingolstadt": "Ingolstadt",
    "atlanta": "Atlanta",
    "hangzhou": "Hangzhou",
    "new_york": "New York",
}


def load_network_scope(manifest_path: Path, result_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    diagnostics = result["source_counterfactual_diagnostics"]
    networks = []
    for entry in manifest["scenarios"]:
        net = parse_net(entry, manifest_path)
        scenario = str(net["scenario"])
        if scenario not in diagnostics:
            raise ValueError(f"result omits source diagnostics for {scenario}")
        source = diagnostics[scenario]
        net.update(
            {
                "source_rows": int(source["rows"]),
                "source_groups": int(source["groups"]),
                "tls_coverage_fraction": float(source["tls_coverage_fraction"]),
            }
        )
        networks.append(net)
    if len(networks) != 16 or len({str(net["city_group"]) for net in networks}) != 6:
        raise ValueError("publication scope requires exactly 16 networks and 6 city groups")
    return networks


def theory_stack() -> None:
    fig = plt.figure(figsize=(7.35, 4.85), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.035, 0.950, "From MC-WM to causal cross-network transfer", fontsize=11, fontweight="bold", color=COL["ink"])
    ax.text(0.035, 0.914, "The paper's theory is a constrained world-model transfer argument, not a claim of black-box policy transfer.", fontsize=7.3, color=COL["muted"])

    # Layer labels.
    for y, label, color in [
        (0.735, "1  Mechanism-conditioned world model", COL["blue"]),
        (0.480, "2  Causal/factored residual transfer", COL["green"]),
        (0.225, "3  Target instantiation and guarded control", COL["teal"]),
    ]:
        ax.text(0.042, y + 0.105, label, fontsize=8.5, fontweight="bold", color=color)
        ax.plot([0.040, 0.960], [y + 0.092, y + 0.092], color="#E2E6EB", lw=1.0)

    box(
        ax,
        (0.055, 0.680),
        0.250,
        0.110,
        "Simulator prior",
        "f0(s,a): uncalibrated dynamics and feasible phases",
        COL["blue_soft"],
        COL["blue"],
        body_width=29,
    )
    box(
        ax,
        (0.370, 0.680),
        0.250,
        0.110,
        "Mechanism state",
        "theta_m,t encodes context for each transition mechanism",
        COL["blue_soft"],
        COL["blue"],
        body_width=31,
    )
    box(
        ax,
        (0.685, 0.680),
        0.250,
        0.110,
        "World-model output",
        "headway/queue, demand, speed, dwell, reward or cost",
        COL["blue_soft"],
        COL["blue"],
        body_width=31,
    )
    arrow(ax, (0.305, 0.735), (0.370, 0.735), COL["blue"])
    arrow(ax, (0.620, 0.735), (0.685, 0.735), COL["blue"])

    box(
        ax,
        (0.055, 0.425),
        0.250,
        0.120,
        "Residual decomposition",
        "f_c(s,a)=f0(s,a)+r_c(s,a), split by mechanism m",
        COL["green_soft"],
        COL["green"],
        body_width=30,
    )
    box(
        ax,
        (0.370, 0.425),
        0.250,
        0.120,
        "Parent-set invariance",
        "r_c,m depends only on Pa_m, not on every city/network feature",
        COL["green_soft"],
        COL["green"],
        body_width=31,
    )
    box(
        ax,
        (0.685, 0.425),
        0.250,
        0.120,
        "Trust / selector",
        "source-fold validation and target summaries gate residual evidence",
        COL["green_soft"],
        COL["green"],
        body_width=31,
    )
    arrow(ax, (0.305, 0.485), (0.370, 0.485), COL["green"])
    arrow(ax, (0.620, 0.485), (0.685, 0.485), COL["green"])

    box(
        ax,
        (0.055, 0.170),
        0.250,
        0.120,
        "Cross-city data",
        "bus cities supply passive residual validation and information budgets",
        COL["teal_soft"],
        COL["teal"],
        body_width=31,
    )
    box(
        ax,
        (0.370, 0.170),
        0.250,
        0.120,
        "Cross-network TSC",
        "RESCO targets instantiate the same mechanism-transfer idea",
        COL["teal_soft"],
        COL["teal"],
        body_width=31,
    )
    box(
        ax,
        (0.685, 0.170),
        0.250,
        0.120,
        "Guarded MPC",
        "learned phase only if it improves on pressure prior",
        COL["gold_soft"],
        COL["gold"],
        body_width=31,
    )
    arrow(ax, (0.305, 0.230), (0.370, 0.230), COL["teal"])
    arrow(ax, (0.620, 0.230), (0.685, 0.230), COL["gold"])

    # Vertical arrows showing logical refinement.
    arrow(ax, (0.496, 0.680), (0.496, 0.545), COL["muted"], dashed=True)
    arrow(ax, (0.496, 0.425), (0.496, 0.290), COL["muted"], dashed=True)

    ax.text(
        0.500,
        0.055,
        "Dense residual transfer removes these restrictions; CFCMT keeps mechanism parents, source trust and control safety separate.",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COL["muted"],
    )
    export(fig, "cfcmt_theory_stack_schematic")


def benchmark_scope() -> None:
    fig = plt.figure(figsize=(7.35, 4.55), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.035, 0.952, "Evidence stack: passive city data and counterfactual TSC networks", fontsize=10.5, fontweight="bold", color=COL["ink"])
    ax.text(0.035, 0.918, "Bus cities define the cross-city information budget; RESCO networks provide closed-loop counterfactual control.", fontsize=7.2, color=COL["muted"])

    # Bus city cards.
    ax.text(0.045, 0.855, "Open transit city bundles", fontsize=8.8, fontweight="bold", color=COL["teal"])
    bus = [
        ("Singapore", "789 lines\n1.24M transitions\nobserved OD/PV + speed bands", COL["teal_soft"], COL["teal"]),
        ("Austin", "234 lines\n0.31M transitions\nschedule demand/speed proxies", COL["grey"], COL["muted"]),
        ("Halifax", "176 lines\n0.27M transitions\nroute APC apportioned", COL["grey"], COL["muted"]),
        ("MBTA", "940 lines\n0.97M transitions\nstop board/alight apportioned", COL["grey"], COL["muted"]),
    ]
    x0 = 0.045
    for i, (name, body, fc, ec) in enumerate(bus):
        box(ax, (x0 + i * 0.230, 0.650), 0.198, 0.165, name, body, fc, ec, body_width=24, title_size=7.6, body_size=6.1)

    # TSC network cards.
    ax.text(0.045, 0.555, "RESCO traffic-signal targets", fontsize=8.8, fontweight="bold", color=COL["blue"])
    resco = [
        ("Regular grids", "grid4x4\narterial4x4\n880 transitions each", COL["blue_soft"], COL["blue"]),
        ("Cologne", "cologne1/3/8\n60, 180, 480 transitions", COL["blue_soft"], COL["blue"]),
        ("Ingolstadt", "ingolstadt1/7/21\n60, 420, 1260 transitions", COL["blue_soft"], COL["blue"]),
        ("Native RL checks", "official RESCO 42 runs\nLibSignal 36 runs", COL["gold_soft"], COL["gold"]),
    ]
    for i, (name, body, fc, ec) in enumerate(resco):
        box(ax, (x0 + i * 0.230, 0.350), 0.198, 0.165, name, body, fc, ec, body_width=24, title_size=7.6, body_size=6.1)

    # Protocol footer.
    box(
        ax,
        (0.075, 0.095),
        0.235,
        0.130,
        "Passive validation",
        "dynamics + information budget\nno action counterfactuals",
        COL["teal_soft"],
        COL["teal"],
        body_width=31,
    )
    box(
        ax,
        (0.380, 0.095),
        0.235,
        0.130,
        "Counterfactual rollout",
        "same SUMO seeds + phases\nclosed-loop queue/trip metrics",
        COL["blue_soft"],
        COL["blue"],
        body_width=31,
    )
    box(
        ax,
        (0.685, 0.095),
        0.235,
        0.130,
        "Claim boundary",
        "TSC is main policy evidence\ncity logs motivate transfer",
        COL["red_soft"],
        COL["red"],
        body_width=31,
    )
    arrow(ax, (0.165, 0.650), (0.190, 0.225), COL["teal"], rad=-0.18)
    arrow(ax, (0.505, 0.350), (0.500, 0.225), COL["blue"])
    arrow(ax, (0.780, 0.350), (0.800, 0.225), COL["red"], rad=0.18)

    export(fig, "cfcmt_benchmark_scope")


def traffic_signal_network_footprints(manifest_path: Path, result_path: Path) -> None:
    nets = load_network_scope(manifest_path, result_path)

    fig, axes = plt.subplots(4, 4, figsize=(7.35, 7.25), facecolor="white")
    fig.subplots_adjust(
        left=0.030,
        right=0.985,
        top=0.900,
        bottom=0.060,
        wspace=0.10,
        hspace=0.24,
    )
    fig.text(
        0.030,
        0.963,
        "Six-domain benchmark spans intersections, corridors and metropolitan grids",
        fontsize=10.6,
        fontweight="bold",
        color=COL["ink"],
    )
    fig.text(
        0.030,
        0.931,
        "Five urban domains and one synthetic domain retain their original SUMO geometry, demand files and tlLogic programs.",
        fontsize=7.2,
        color=COL["muted"],
    )

    for panel_index, (ax, net) in enumerate(zip(axes.flat, nets)):
        ax.set_aspect("equal")
        ax.set_axis_off()
        xmin, ymin, xmax, ymax = net["bbox"]  # type: ignore[index]
        dx = xmax - xmin
        dy = ymax - ymin
        pad = 0.075 * max(dx, dy)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

        for shape in net["edges"]:  # type: ignore[union-attr]
            xs = [p[0] for p in shape]
            ys = [p[1] for p in shape]
            ax.plot(xs, ys, color="#BFC7D2", lw=0.52, alpha=0.78, solid_capstyle="round")

        signals = net["signals"]  # type: ignore[assignment]
        if signals:
            sx = [p[0] for p in signals]
            sy = [p[1] for p in signals]
            ax.scatter(sx, sy, s=8, color=COL["red"], edgecolor="white", linewidth=0.35, zorder=4)

        title = SCENARIO_LABELS[str(net["scenario"])]
        text_bg = {"facecolor": "white", "edgecolor": "none", "boxstyle": "round,pad=0.12", "alpha": 0.92}
        ax.text(
            0.02,
            0.98,
            f"{chr(ord('a') + panel_index)}  {title}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            fontweight="bold",
            color=COL["blue"],
            bbox=text_bg,
            zorder=5,
        )
        ax.text(
            0.02,
            0.865,
            f"{CITY_LABELS[str(net['city_group'])]} | {net['tl_logic_count']} TLS | {net['lane_count']} lanes",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.2,
            color=COL["muted"],
            bbox=text_bg,
            zorder=5,
        )

    # Legend-like footer, kept outside the panels to avoid cluttering maps.
    fig.text(
        0.030,
        0.022,
        "Grey: non-internal lanes. Red: signalized junctions. TLS counts are original tlLogic controllers; geometry is not normalized across domains.",
        fontsize=6.1,
        color=COL["muted"],
    )
    export(fig, "traffic_signal_network_footprints")
    write_network_table(nets)


def write_network_table(nets: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Scenario & Domain & Suite & TLS & Lanes & Green phases & CF rows & CF groups \\",
        r"\midrule",
    ]
    csv_rows = []
    for net in nets:
        scenario = str(net["scenario"])
        rows.append(
            " & ".join(
                [
                    rf"\texttt{{{latex_escape(scenario)}}}",
                    latex_escape(CITY_LABELS[str(net["city_group"])]),
                    latex_escape(net["suite"]),
                    str(net["tl_logic_count"]),
                    str(net["lane_count"]),
                    str(net["green_phase_count"]),
                    str(net["source_rows"]),
                    str(net["source_groups"]),
                ]
            )
            + r" \\"
        )
        csv_rows.append(
            {
                "scenario": scenario,
                "city_group": net["city_group"],
                "suite": net["suite"],
                "tl_logic_count": net["tl_logic_count"],
                "signalized_junction_count": net["signal_count"],
                "edge_count": net["edge_count"],
                "lane_count": net["lane_count"],
                "green_phase_count": net["green_phase_count"],
                "source_counterfactual_rows": net["source_rows"],
                "source_counterfactual_groups": net["source_groups"],
                "tls_coverage_fraction": net["tls_coverage_fraction"],
                "provenance": net["provenance"],
            }
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLE_DIR / "traffic_signal_network_scope.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    csv_path = SOURCE_DATA_DIR / "traffic_signal_network_scope.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("theory", "scope", "networks"),
        default=("theory", "scope", "networks"),
    )
    args = parser.parse_args()
    if "theory" in args.figures:
        theory_stack()
    if "scope" in args.figures:
        benchmark_scope()
    if "networks" in args.figures:
        traffic_signal_network_footprints(args.manifest.resolve(), args.result.resolve())


if __name__ == "__main__":
    main()
