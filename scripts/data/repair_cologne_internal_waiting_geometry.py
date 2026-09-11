#!/usr/bin/env python3
"""Move one Cologne internal waiting split while preserving its full path."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


PROTOCOL = "cologne-internal-waiting-geometry-repair-v1"
PREFIX = ":cluster_357187_359543_"
WAITING, CONTINUATION, FOE = PREFIX + "13_0", PREFIX + "24_0", PREFIX + "1_1"
VEHICLE_LENGTH_M, VEHICLE_WIDTH_M, LENGTH_GRID_M = 4.3, 1.8, .01


def points(tokens):
    return [tuple(map(float, token.split(","))) for token in tokens]


def shape_length(shape):
    return sum(math.dist(a, b) for a, b in zip(shape, shape[1:]))


def point_at(shape, distance):
    for a, b in zip(shape, shape[1:]):
        length = math.dist(a, b)
        if distance <= length:
            return tuple(a[k] + (b[k] - a[k]) * distance / length for k in (0, 1))
        distance -= length
    return shape[-1]


def passenger_polygon(front, back, width):
    """SUMO1.22 zero-margin passenger octagon from its front/back points."""
    dx, dy = back[0] - front[0], back[1] - front[1]
    length = math.hypot(dx, dy)
    nx, ny = dy / length, -dx / length
    return [[front[0] + along * dx + side * width * nx,
             front[1] + along * dy + side * width * ny]
            for along, side in ((0, .3), (.1, .5), (.9, .5), (1, .3),
                                (1, -.3), (.9, -.5), (.1, -.5), (0, -.3))]


def semantic_attribute_changes(before, after):
    old_nodes, new_nodes = list(before.iter()), list(after.iter())
    if len(old_nodes) != len(new_nodes):
        raise ValueError("Repair changed the network element count")
    changes = []
    for old, new in zip(old_nodes, new_nodes):
        if old.tag != new.tag or old.get("id") != new.get("id") or (old.text or "").strip() != (new.text or "").strip():
            raise ValueError("Repair changed network element identity/order/text")
        for key in sorted(set(old.attrib) | set(new.attrib)):
            if old.get(key) != new.get(key):
                changes.append({"tag": old.tag, "id": old.get("id"), "attribute": key,
                                "old": old.get(key), "new": new.get(key)})
    return changes


def repair_tree(root):
    original = deepcopy(root)
    lanes = {lane.get("id"): lane for lane in root.iter("lane")}
    waiting, continuation, foe = (lanes[name] for name in (WAITING, CONTINUATION, FOE))
    junction = next(row for row in root.iter("junction") if row.get("id") == CONTINUATION)
    waiting_tokens, continuation_tokens = waiting.get("shape").split(), continuation.get("shape").split()
    old_shape, later_shape, foe_shape = (points(tokens) for tokens in (
        waiting_tokens, continuation_tokens, foe.get("shape").split()))
    if old_shape[-1] != later_shape[0] or len(foe_shape) != 2:
        raise ValueError("Expected connected Cologne waiting/continuation and straight foe shapes")
    old_length, later_length = float(waiting.get("length")), float(continuation.get("length"))
    if (old_length, later_length) != (8.76, 19.77):
        raise ValueError("Expected the original Cologne 13/24 split, not a previously repaired input")
    geometry_length = shape_length(old_shape)
    ratio = geometry_length / old_length
    foe_dx, foe_dy = (foe_shape[1][k] - foe_shape[0][k] for k in (0, 1))
    foe_length = math.hypot(foe_dx, foe_dy)
    direction = (foe_dx / foe_length, foe_dy / foe_length)
    normal = (-direction[1], direction[0])

    def project(point, axis):
        return sum((point[k] - foe_shape[0][k]) * axis[k] for k in (0, 1))

    def body_at(front_position):
        front = point_at(old_shape, front_position * ratio)
        back = point_at(old_shape, (front_position - VEHICLE_LENGTH_M) * ratio)
        return passenger_polygon(front, back, VEHICLE_WIDTH_M)

    def clearance(front_position):
        return min(project(point, normal) for point in body_at(front_position)) - VEHICLE_WIDTH_M / 2

    # Search the physical geometric boundary, without running SUMO or reading
    # rollout performance. The car front is allowed all the way to the split.
    lo, hi = VEHICLE_LENGTH_M, old_length
    if clearance(lo) <= 0 or clearance(hi) >= 0:
        raise ValueError("The recorded Cologne geometry does not bracket this clearance repair")
    for _ in range(60):
        middle = (lo + hi) / 2
        if clearance(middle) >= 0:
            lo = middle
        else:
            hi = middle
    new_length = math.floor(lo / LENGTH_GRID_M) * LENGTH_GRID_M
    new_length_text = f"{new_length:.2f}"
    new_length = float(new_length_text)
    new_later_length = old_length + later_length - new_length
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
    continuation.set("length", f"{new_later_length:.2f}")
    junction.set("x", split_token.split(",")[0])
    junction.set("y", split_token.split(",")[1])

    changes = semantic_attribute_changes(original, root)
    allowed = {("lane", lane, field) for lane in (WAITING, CONTINUATION) for field in ("shape", "length")}
    allowed |= {("junction", CONTINUATION, field) for field in ("x", "y")}
    if {(row["tag"], row["id"], row["attribute"]) for row in changes} != allowed or len(changes) != 6:
        raise ValueError("The repair must change exactly the six frozen geometry attributes")

    written_shape = points(before_tokens)
    written_later_shape = points(after_tokens)
    written_ratio = shape_length(written_shape) / new_length
    written_body = passenger_polygon(written_shape[-1], point_at(written_shape, (new_length - VEHICLE_LENGTH_M) * written_ratio), VEHICLE_WIDTH_M)
    written_clearance = min(project(point, normal) for point in written_body) - VEHICLE_WIDTH_M / 2
    projections = [project(point, direction) for point in written_body]
    old_joined = waiting_tokens + continuation_tokens[1:]
    new_joined = before_tokens + after_tokens[1:]
    old_total_geometry = shape_length(old_shape) + shape_length(later_shape)
    new_total_geometry = shape_length(written_shape) + shape_length(written_later_shape)
    invariants = {
        "exactly_six_allowed_attribute_changes": True,
        "old_vertices_preserved_with_one_new_collinear_split": [token for token in new_joined if token != split_token] == old_joined,
        "old_split_retained_inside_continuation": waiting_tokens[-1] in after_tokens[1:-1],
        "combined_geometry_length_preserved": abs(new_total_geometry - old_total_geometry) < 1e-9,
        "combined_nominal_length_preserved": abs(float(waiting.get("length")) + float(continuation.get("length")) - old_length - later_length) < 1e-12,
        "waiting_prefix_position_mapping_preserved": abs(written_ratio - ratio) < 1e-12,
        "front_at_split_body_clearance_nonnegative": written_clearance >= 0,
        "waiting_body_projects_within_finite_foe_segment": min(projections) >= 0 and max(projections) <= foe_length,
    }
    if not all(invariants.values()):
        raise ValueError(f"Frozen geometry invariants failed: {invariants}")
    return {
        "protocol": PROTOCOL,
        "parameters": {"vehicle_length_m": VEHICLE_LENGTH_M, "vehicle_width_m": VEHICLE_WIDTH_M,
                       "foe_vehicle_width_m": VEHICLE_WIDTH_M, "length_grid_m": LENGTH_GRID_M,
                       "criterion": "front_at_split_passenger_polygon_outside_straight_swept_strip"},
        "lanes": {"waiting": WAITING, "continuation": CONTINUATION, "foe": FOE},
        "geometry": {
            "original_split_nominal_m": old_length, "original_continuation_nominal_m": later_length,
            "unrounded_clearance_boundary_nominal_m": lo, "new_split_nominal_m": new_length,
            "new_continuation_nominal_m": float(continuation.get("length")), "split_xy_m": list(map(float, split_token.split(","))),
            "original_front_at_split_clearance_m": clearance(old_length), "boundary_body_clearance_m": written_clearance,
            "geometric_retreat_m": geometry_length - split_distance,
            "old_combined_geometry_length_m": old_total_geometry, "new_combined_geometry_length_m": new_total_geometry,
            "combined_nominal_length_m": old_length + later_length,
            "waiting_geometry_per_nominal_m_before": ratio, "waiting_geometry_per_nominal_m_after": written_ratio,
            "continuation_geometry_per_nominal_m_before": shape_length(later_shape) / later_length,
            "continuation_geometry_per_nominal_m_after": shape_length(written_later_shape) / float(continuation.get("length")),
            "front_at_split_body_xy_m": written_body,
            "finite_foe_projection_range_m": [min(projections), max(projections)],
        },
        "semantic_changes": changes, "invariants": invariants,
        "scope": "Only movement13 waiting split against straight2; mirrored movement3 is unchanged.",
        "distance_mapping": "Waiting prefix preserves its original geometry/logical ratio. Continuation absorbs the shifted nominal length; its internal mapping changes slightly because the two original ratios differed.",
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
    written_changes = semantic_attribute_changes(original, ET.parse(output_net).getroot())
    if written_changes != report["semantic_changes"]:
        raise ValueError("Serialized network differs from the six validated attribute edits")
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
    print(json.dumps({"protocol": PROTOCOL, "geometry": report["geometry"], "invariants": report["invariants"]}))


if __name__ == "__main__":
    main()
