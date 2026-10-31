"""Compare measured waiting bodies against the same straight vehicle strip."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from shapely.geometry import LineString, Polygon


def plot(original_path, observation_path, out_prefix):
    original = json.loads(original_path.read_text())
    observations = json.loads(observation_path.read_text())
    old_sample = next(row for row in original["samples"] if row["time_sec"] == 25657
                      and row["stage"] == "after_simulation_step_before_executor")
    old = old_sample["vehicles"]["129962_409_0"]
    new = observations["minimum_clearance_observation"]
    center = original["native_lane_catalog"][":cluster_357187_359543_1_1"]["shape_xy_m"]
    origin = (11780, 13320)

    def shifted(points):
        return [(x - origin[0], y - origin[1]) for x, y in points]

    line = LineString(shifted(center))
    strip = line.buffer(0.9, cap_style="flat")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
    descriptions = [
        (old, "Original: 5.65 cm intrusion", "129962_409_0, t = 25657 s", "#B6293E"),
        (new, "Repaired: 3.56 cm clearance", "153000_419_0, t = 25665 s", "#18764E"),
    ]
    for ax, (vehicle, title, caption, color) in zip(axes, descriptions):
        polygon = Polygon(shifted(vehicle["passenger_polygon_xy_m"]))
        ax.add_patch(Patch(strip.exterior.coords, fc="#BDD5E2", ec="#5487A2", lw=1.2))
        ax.add_patch(Patch(polygon.exterior.coords, fc="#F0CFAB", ec="#BD793D", lw=1.5))
        intersection = polygon.intersection(strip)
        if not intersection.is_empty:
            ax.add_patch(Patch(intersection.exterior.coords, fc="#B6293E", ec="#B6293E"))
        ax.set_title(title, loc="left", fontsize=12, color=color, pad=13)
        ax.text(0.04, 0.94, "Straight vehicle swept strip\n(width 1.8 m)", transform=ax.transAxes,
                va="top", fontsize=10, color="#375F76")
        ax.text(0.04, 0.08, "Measured left-turn body\n" + caption, transform=ax.transAxes,
                va="bottom", fontsize=9, color="#7A512C")
        ax.set_xlim(6.5, 8.7)
        ax.set_ylim(5.7, 7.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Local x (m)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Local y (m)")
    fig.suptitle("Movement 13: measured vehicle-body clearance after moving the internal split",
                 x=0.07, ha="left", fontsize=12)
    fig.text(0.07, 0.045,
             "Different vehicles at the same waiting location; repaired vehicle speed = 0.000366 m/s.\n"
             "Local geometry improves. The paired full-window validation remains FAIL (unrepaired mirror collision).",
             fontsize=9, color="#42505A")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.24, wspace=0.14)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=180)
    fig.savefig(out_prefix.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    plot(args.original, args.observations, args.out_prefix)
