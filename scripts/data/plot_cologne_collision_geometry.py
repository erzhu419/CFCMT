"""Plot the measured vehicle polygons at the reproduced Cologne collision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
from shapely.geometry import LineString, Polygon


def plot(result_path: Path, output_prefix: Path) -> None:
    result = json.loads(result_path.read_text())
    if result["status"] != "PASS":
        raise ValueError("The original collision must first reproduce exactly")
    event = result["expected_first_collision"]
    sample = next(
        row for row in result["samples"]
        if row["time_sec"] == event["time_sec"]
        and row["stage"] == "after_simulation_step_before_executor"
    )
    vehicles = [sample["vehicles"][event[key]] for key in ("collider", "victim")]
    shapes = [Polygon(vehicle["passenger_polygon_xy_m"]) for vehicle in vehicles]
    overlap = shapes[0].intersection(shapes[1])
    if overlap.is_empty:
        raise ValueError("The selected measured polygons do not overlap")
    origin = (11780.0, 13320.0)

    def shifted(points):
        return [(x - origin[0], y - origin[1]) for x, y in points]

    colors = ("#2378A8", "#E28738")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax in axes:
        for vehicle in vehicles:
            lane = result["native_lane_catalog"][vehicle["lane_id"]]
            line = LineString(shifted(lane["shape_xy_m"]))
            surface = line.buffer(lane["width_m"] / 2, cap_style="flat")
            ax.add_patch(PolygonPatch(surface.exterior.coords, fc="#E8ECEF", ec="white", lw=1))
            x, y = line.xy
            ax.plot(x, y, "--", color="#A2ACB3", lw=0.8)
        for vehicle, color in zip(vehicles, colors):
            ax.add_patch(PolygonPatch(shifted(vehicle["passenger_polygon_xy_m"]),
                                     fc=color, ec=color, alpha=0.45, lw=1.4))
            front = shifted([vehicle["position_xy_m"]])[0]
            ax.plot(*front, "o", ms=3, color=color)
        ax.add_patch(PolygonPatch(shifted(overlap.exterior.coords), fc="#B51F36", ec="#B51F36", lw=1))
        ax.set_aspect("equal")
        ax.set_xlabel("Local x (m)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Local y (m)")
    axes[0].set_xlim(1, 14)
    axes[0].set_ylim(2, 12)
    axes[0].set_title("Recorded vehicle bodies at t = 25657 s", fontsize=11, loc="left")
    axes[0].annotate("Straight: 9.25 m/s", xy=shifted([vehicles[0]["position_xy_m"]])[0],
                     xytext=(1.5, 10.8), fontsize=10, color=colors[0],
                     arrowprops={"arrowstyle": "->", "color": colors[0]})
    axes[0].annotate("Left turn: stopped", xy=shifted([vehicles[1]["position_xy_m"]])[0],
                     xytext=(7.3, 3), fontsize=10, color=colors[1],
                     arrowprops={"arrowstyle": "->", "color": colors[1]})
    cx, cy = shifted([(overlap.centroid.x, overlap.centroid.y)])[0]
    axes[1].set_xlim(cx - 0.9, cx + 0.9)
    axes[1].set_ylim(cy - 0.9, cy + 0.9)
    axes[1].set_title("Contact detail", fontsize=11, loc="left")
    axes[1].text(0.03, 0.98, "Native junction collision\nPolygon overlap: 5.65 cm*",
                 transform=axes[1].transAxes, va="top", fontsize=10,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    fig.suptitle("Cologne internal waiting point: a stopped left-turn vehicle protrudes into through traffic",
                 fontsize=12, x=0.07, ha="left")
    fig.text(0.07, 0.03,
             "* Minimum separating-axis penetration of native-shape passenger polygons; zero extra margin.\n"
             "Source: V154 collision replay t90912, 23/23 original action records reproduced exactly.",
             fontsize=8, color="#46545D")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.81, wspace=0.18)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=180)
    fig.savefig(output_prefix.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--out-prefix", required=True, type=Path)
    args = parser.parse_args()
    plot(args.result, args.out_prefix)
