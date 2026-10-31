#!/usr/bin/env python3
"""Retain the Cologne 13/24 repair and apply the same rule to mirrored 3/20."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    from scripts.data.repair_cologne_internal_waiting_geometry import (
        LENGTH_GRID_M, PREFIX, VEHICLE_LENGTH_M, VEHICLE_WIDTH_M,
        passenger_polygon, point_at, points, repair_tree as repair_primary_tree,
        semantic_attribute_changes, shape_length,
    )
except ModuleNotFoundError:  # Direct execution by file path in the tooling snapshot.
    from repair_cologne_internal_waiting_geometry import (
        LENGTH_GRID_M, PREFIX, VEHICLE_LENGTH_M, VEHICLE_WIDTH_M,
        passenger_polygon, point_at, points, repair_tree as repair_primary_tree,
        semantic_attribute_changes, shape_length,
    )


PROTOCOL = "cologne-bilateral-waiting-geometry-repair-v1"
WAITING, CONTINUATION, FOE = PREFIX + "3_0", PREFIX + "20_0", PREFIX + "11_1"


def repair_mirror_tree(root):
    """Apply V154C's front-at-split criterion to the original mirrored split."""
    original = deepcopy(root)
    lanes = {lane.get("id"): lane for lane in root.iter("lane")}
    waiting, continuation, foe = (lanes[name] for name in (WAITING, CONTINUATION, FOE))
    junction = next(row for row in root.iter("junction") if row.get("id") == CONTINUATION)
    waiting_tokens = waiting.get("shape").split()
    continuation_tokens = continuation.get("shape").split()
    old_shape, later_shape, foe_shape = (points(tokens) for tokens in (
        waiting_tokens, continuation_tokens, foe.get("shape").split()))
    if old_shape[-1] != later_shape[0] or len(foe_shape) != 2:
        raise ValueError("Expected connected Cologne waiting/continuation and straight foe shapes")
    old_length, later_length = float(waiting.get("length")), float(continuation.get("length"))
    if (old_length, later_length) != (8.62, 19.58):
        raise ValueError("Expected the original Cologne 3/20 split")
    geometry_length = shape_length(old_shape)
    ratio = geometry_length / old_length
    dx, dy = (foe_shape[1][k] - foe_shape[0][k] for k in (0, 1))
    foe_length = math.hypot(dx, dy)
    direction = (dx / foe_length, dy / foe_length)
    normal = (-direction[1], direction[0])

    def project(point, axis):
        return sum((point[k] - foe_shape[0][k]) * axis[k] for k in (0, 1))

    def body_at(front_position):
        front = point_at(old_shape, front_position * ratio)
        back = point_at(old_shape, (front_position - VEHICLE_LENGTH_M) * ratio)
        return passenger_polygon(front, back, VEHICLE_WIDTH_M)

    def clearance(front_position):
        return min(project(point, normal) for point in body_at(front_position)) - VEHICLE_WIDTH_M / 2

    # Match V154C: solve from static geometry only, placing the front at the
    # split, then floor the safe logical position to the upstream centimetre.
    lo, hi = VEHICLE_LENGTH_M, old_length
    if clearance(lo) <= 0 or clearance(hi) >= 0:
        raise ValueError("The recorded mirrored geometry does not bracket the clearance repair")
    for _ in range(60):
        middle = (lo + hi) / 2
        if clearance(middle) >= 0:
            lo = middle
        else:
            hi = middle
    new_length_text = f"{math.floor(lo / LENGTH_GRID_M) * LENGTH_GRID_M:.2f}"
    new_length = float(new_length_text)
    split_distance = new_length * ratio
    split_point = point_at(old_shape, split_distance)
    split_token = ",".join(f"{value:.12f}" for value in split_point)
    remaining = split_distance
    for segment, (a, b) in enumerate(zip(old_shape, old_shape[1:])):
        segment_length = math.dist(a, b)
        if remaining < segment_length:
            break
        remaining -= segment_length
    before_tokens = waiting_tokens[:segment + 1] + [split_token]
    after_tokens = [split_token] + waiting_tokens[segment + 1:] + continuation_tokens[1:]
    waiting.set("shape", " ".join(before_tokens))
    waiting.set("length", new_length_text)
    continuation.set("shape", " ".join(after_tokens))
    continuation.set("length", f"{old_length + later_length - new_length:.2f}")
    junction.set("x", split_token.split(",")[0])
    junction.set("y", split_token.split(",")[1])

    changes = semantic_attribute_changes(original, root)
    allowed = {("lane", lane, field) for lane in (WAITING, CONTINUATION) for field in ("shape", "length")}
    allowed |= {("junction", CONTINUATION, field) for field in ("x", "y")}
    if {(row["tag"], row["id"], row["attribute"]) for row in changes} != allowed or len(changes) != 6:
        raise ValueError("The mirrored repair must change exactly six geometry attributes")

    written_shape, written_later_shape = points(before_tokens), points(after_tokens)
    written_ratio = shape_length(written_shape) / new_length
    written_body = passenger_polygon(
        written_shape[-1], point_at(written_shape, (new_length - VEHICLE_LENGTH_M) * written_ratio),
        VEHICLE_WIDTH_M,
    )
    written_clearance = min(project(point, normal) for point in written_body) - VEHICLE_WIDTH_M / 2
    projections = [project(point, direction) for point in written_body]
    old_total_geometry = shape_length(old_shape) + shape_length(later_shape)
    new_total_geometry = shape_length(written_shape) + shape_length(written_later_shape)
    invariants = {
        "exactly_six_allowed_attribute_changes": True,
        "old_vertices_preserved_with_one_new_collinear_split": [
            token for token in before_tokens + after_tokens[1:] if token != split_token
        ] == waiting_tokens + continuation_tokens[1:],
        "old_split_retained_inside_continuation": waiting_tokens[-1] in after_tokens[1:-1],
        "combined_geometry_length_preserved": abs(new_total_geometry - old_total_geometry) < 1e-9,
        "combined_nominal_length_preserved": abs(float(waiting.get("length")) + float(continuation.get("length")) - old_length - later_length) < 1e-12,
        "waiting_prefix_position_mapping_preserved": abs(written_ratio - ratio) < 1e-12,
        "front_at_split_body_clearance_nonnegative": written_clearance >= 0,
        "waiting_body_projects_within_finite_foe_segment": min(projections) >= 0 and max(projections) <= foe_length,
    }
    if not all(invariants.values()):
        raise ValueError(f"Frozen mirrored geometry invariants failed: {invariants}")
    return {
        "lanes": {"waiting": WAITING, "continuation": CONTINUATION, "foe": FOE},
        "geometry": {
            "original_split_nominal_m": old_length, "original_continuation_nominal_m": later_length,
            "unrounded_clearance_boundary_nominal_m": lo, "new_split_nominal_m": new_length,
            "new_continuation_nominal_m": float(continuation.get("length")),
            "split_xy_m": list(map(float, split_token.split(","))),
            "original_front_at_split_clearance_m": clearance(old_length),
            "boundary_body_clearance_m": written_clearance,
            "geometric_retreat_m": geometry_length - split_distance,
            "old_combined_geometry_length_m": old_total_geometry,
            "new_combined_geometry_length_m": new_total_geometry,
            "combined_nominal_length_m": old_length + later_length,
            "waiting_geometry_per_nominal_m_before": ratio,
            "waiting_geometry_per_nominal_m_after": written_ratio,
            "continuation_geometry_per_nominal_m_before": shape_length(later_shape) / later_length,
            "continuation_geometry_per_nominal_m_after": shape_length(written_later_shape) / float(continuation.get("length")),
            "front_at_split_body_xy_m": written_body,
            "finite_foe_projection_range_m": [min(projections), max(projections)],
        },
        "semantic_changes": changes,
        "invariants": invariants,
    }


def repair_tree(root):
    original = deepcopy(root)
    primary_report = repair_primary_tree(root)
    single_side = deepcopy(root)
    mirror_report = repair_mirror_tree(root)
    changes = semantic_attribute_changes(original, root)
    mirror_changes = semantic_attribute_changes(single_side, root)
    expected = {
        (row["tag"], row["id"], row["attribute"])
        for report in (primary_report, mirror_report) for row in report["semantic_changes"]
    }
    invariants = {
        "exactly_twelve_allowed_attribute_changes_vs_original": len(changes) == 12 and {
            (row["tag"], row["id"], row["attribute"]) for row in changes
        } == expected,
        "exactly_six_mirrored_attribute_changes_vs_v154c": mirror_changes == mirror_report["semantic_changes"] and len(mirror_changes) == 6,
        "primary_v154c_changes_preserved": all(row in changes for row in primary_report["semantic_changes"]),
        "primary_geometry_invariants_pass": all(primary_report["invariants"].values()),
        "mirror_geometry_invariants_pass": all(mirror_report["invariants"].values()),
    }
    if not all(invariants.values()):
        raise ValueError(f"Bilateral geometry invariants failed: {invariants}")
    return {
        "protocol": PROTOCOL, "parameters": primary_report["parameters"],
        "primary_repair": primary_report, "mirror_repair": mirror_report,
        "semantic_changes": changes, "changes_vs_v154c": mirror_changes,
        "invariants": invariants,
        "scope": "Retain V154C movement13/24; apply the identical geometric rule to mirrored movement3/20 against straight11_1.",
        "distance_mapping": primary_report["distance_mapping"],
        "claim_boundary": "Static geometry clearance for the recorded 4.3x1.8m passenger class; no rollout or broader safety admission is implied.",
    }


def repair_network(source_net: Path, output_net: Path):
    if source_net.resolve() == output_net.resolve() or output_net.exists():
        raise FileExistsError("Use a fresh separate repaired network path")
    tree = ET.parse(source_net)
    original = deepcopy(tree.getroot())
    report = repair_tree(tree.getroot())
    output_net.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_net, encoding="utf-8", xml_declaration=True)
    if semantic_attribute_changes(original, ET.parse(output_net).getroot()) != report["semantic_changes"]:
        raise ValueError("Serialized network differs from the twelve validated attribute edits")
    report.update(source_net=str(source_net.resolve()), output_net=str(output_net.resolve()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = repair_network(args.net, args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"protocol": PROTOCOL,
                      "primary_geometry": report["primary_repair"]["geometry"],
                      "mirror_geometry": report["mirror_repair"]["geometry"],
                      "invariants": report["invariants"]}))


if __name__ == "__main__":
    main()
