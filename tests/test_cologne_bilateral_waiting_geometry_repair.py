from copy import deepcopy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data.repair_cologne_bilateral_waiting_geometry import (
    CONTINUATION, FOE, WAITING, passenger_polygon, point_at, points,
    repair_network, repair_primary_tree, repair_tree, semantic_attribute_changes,
    shape_length,
)


STATIC = Path(__file__).resolve().parents[1] / "cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/collision_static_geometry.json"


def recorded_network():
    data = json.loads(STATIC.read_text())
    root = ET.Element("net", data["net_attributes"])
    for edge in data["internal_edges"]:
        node = ET.SubElement(root, "edge", edge["attrs"])
        for lane in edge["lanes"]:
            ET.SubElement(node, "lane", lane)
    tls = ET.SubElement(root, "tlLogic", {"id": "cluster_357187_359543", "type": "static", "programID": "0"})
    ET.SubElement(tls, "phase", {"state": "GGGggrrrrrGGGggrrrrr", "duration": "5"})
    for junction in data["junctions"]:
        node = ET.SubElement(root, "junction", junction["attrs"])
        for request in junction["requests"]:
            ET.SubElement(node, "request", request)
    for connection in data["connections"]:
        ET.SubElement(root, "connection", connection)
    return root


def test_bilateral_repair_preserves_v154c_and_changes_only_twelve_geometry_attributes():
    root = recorded_network()
    original = deepcopy(root)
    v154c = deepcopy(root)
    primary_report = repair_primary_tree(v154c)
    report = repair_tree(root)
    assert report["primary_repair"] == primary_report
    assert len(semantic_attribute_changes(original, root)) == 12
    assert semantic_attribute_changes(v154c, root) == report["changes_vs_v154c"]
    assert len(report["changes_vs_v154c"]) == 6
    assert {row["id"] for row in report["changes_vs_v154c"]} == {WAITING, CONTINUATION}
    assert all(report["invariants"].values())
    for side in ("primary_repair", "mirror_repair"):
        assert all(report[side]["invariants"].values())
    lanes = {row.get("id"): row for row in root.iter("lane")}
    old_lanes = {row.get("id"): row for row in original.iter("lane")}
    old_joined = old_lanes[WAITING].get("shape").split() + old_lanes[CONTINUATION].get("shape").split()[1:]
    before, after = (lanes[lane].get("shape").split() for lane in (WAITING, CONTINUATION))
    new_joined = before + after[1:]
    new_joined.remove(before[-1])
    assert old_joined == new_joined
    assert lanes[FOE].attrib == old_lanes[FOE].attrib
    assert float(lanes[WAITING].get("length")) + float(lanes[CONTINUATION].get("length")) == pytest.approx(28.2)


def test_mirror_uses_front_at_split_clearance_and_most_downstream_safe_centimetre():
    root = recorded_network()
    lanes = {row.get("id"): row for row in root.iter("lane")}
    shape = points(lanes[WAITING].get("shape").split())
    factor = shape_length(shape) / float(lanes[WAITING].get("length"))
    a, b = points(lanes[FOE].get("shape").split())
    dx, dy = b[0] - a[0], b[1] - a[1]
    normal = (-dy / math.hypot(dx, dy), dx / math.hypot(dx, dy))

    def clearance(front_s):
        body = passenger_polygon(point_at(shape, front_s * factor), point_at(shape, (front_s - 4.3) * factor), 1.8)
        return min(sum((v[k] - a[k]) * normal[k] for k in (0, 1)) for v in body) - .9

    report = repair_tree(root)
    geometry = report["mirror_repair"]["geometry"]
    split = geometry["new_split_nominal_m"]
    assert clearance(split) > 0
    assert clearance(split + .01) < 0
    assert clearance(8.519) < 0
    assert split <= geometry["unrounded_clearance_boundary_nominal_m"] < split + .01
    assert geometry["boundary_body_clearance_m"] == pytest.approx(clearance(split), abs=1e-9)
    assert geometry["waiting_geometry_per_nominal_m_after"] == pytest.approx(factor, abs=1e-12)


def test_serialized_bilateral_package_preserves_original_and_validated_semantic_scope(tmp_path):
    source, output = tmp_path / "original.net.xml", tmp_path / "bilateral.net.xml"
    ET.ElementTree(recorded_network()).write(source, encoding="utf-8", xml_declaration=True)
    original_bytes = source.read_bytes()
    report = repair_network(source, output)
    assert source.read_bytes() == original_bytes
    assert report["source_net"] == str(source.resolve())
    assert report["output_net"] == str(output.resolve())
    assert semantic_attribute_changes(ET.parse(source).getroot(), ET.parse(output).getroot()) == report["semantic_changes"]
    with pytest.raises(FileExistsError):
        repair_network(source, output)
