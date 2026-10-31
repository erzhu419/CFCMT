from copy import deepcopy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data.repair_cologne_internal_waiting_geometry import (
    CONTINUATION, FOE, PREFIX, WAITING, passenger_polygon, point_at, points,
    repair_network, repair_tree, semantic_attribute_changes, shape_length,
)


STATIC = Path(__file__).resolve().parents[1] / "cf_h2o/results/cluster/tsc_v154_cologne_service_diagnostic_20260909/collision_static_geometry.json"


def recorded_network():
    data = json.loads(STATIC.read_text())
    root = ET.Element("net", data["net_attributes"])
    for edge in data["internal_edges"]:
        node = ET.SubElement(root, "edge", edge["attrs"])
        for lane in edge["lanes"]:
            ET.SubElement(node, "lane", lane)
    # A signal program must be semantically untouched along with foes/via.
    tls = ET.SubElement(root, "tlLogic", {"id": "cluster_357187_359543", "type": "static", "programID": "0"})
    ET.SubElement(tls, "phase", {"state": "GGGggrrrrrGGGggrrrrr", "duration": "5"})
    for junction in data["junctions"]:
        node = ET.SubElement(root, "junction", junction["attrs"])
        for request in junction["requests"]:
            ET.SubElement(node, "request", request)
    for connection in data["connections"]:
        ET.SubElement(root, "connection", connection)
    return root


def test_recorded_geometry_repair_preserves_route_and_only_six_attributes():
    root = recorded_network()
    before = deepcopy(root)
    old_lanes = {row.get("id"): row for row in before.iter("lane")}
    report = repair_tree(root)
    lanes = {row.get("id"): row for row in root.iter("lane")}
    assert lanes[WAITING].get("length") == "8.48"
    assert lanes[CONTINUATION].get("length") == "20.05"
    assert len(semantic_attribute_changes(before, root)) == 6
    assert all(report["invariants"].values())
    for lane in (PREFIX + "3_0", PREFIX + "20_0", FOE):
        assert lanes[lane].attrib == old_lanes[lane].attrib
    original_joined = old_lanes[WAITING].get("shape").split() + old_lanes[CONTINUATION].get("shape").split()[1:]
    new_waiting, new_continuation = (lanes[lane].get("shape").split() for lane in (WAITING, CONTINUATION))
    new_joined = new_waiting + new_continuation[1:]
    new_joined.remove(new_waiting[-1])
    assert new_joined == original_joined
    # High-precision split coordinates matter: old two-decimal xy formatting
    # would be inappropriate for the frozen millimetre-scale geometric margin.
    assert new_waiting[-1] == "11788.046881453862,13325.914587635907"
    assert report["geometry"]["boundary_body_clearance_m"] == pytest.approx(.0023675159789, abs=1e-9)


def test_rounding_is_the_most_downstream_safe_centimetre_without_stop_offset():
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
    assert clearance(8.48) > 0
    assert clearance(8.49) < 0
    assert clearance(8.659) == pytest.approx(-.0565085577509, abs=1e-9)
    # The criterion is evaluated at the boundary, not at boundary-minus-0.101.
    report = repair_tree(root)
    assert report["geometry"]["unrounded_clearance_boundary_nominal_m"] == pytest.approx(8.487193932616, abs=1e-9)


def test_independent_output_preserves_source_and_serialized_semantic_scope(tmp_path):
    source, output = tmp_path / "original.net.xml", tmp_path / "new.net.xml"
    ET.ElementTree(recorded_network()).write(source, encoding="utf-8", xml_declaration=True)
    original_bytes = source.read_bytes()
    report = repair_network(source, output)
    assert source.read_bytes() == original_bytes
    assert report["source_net"] == str(source.resolve())
    assert report["output_net"] == str(output.resolve())
    assert semantic_attribute_changes(ET.parse(source).getroot(), ET.parse(output).getroot()) == report["semantic_changes"]
    with pytest.raises(FileExistsError):
        repair_network(source, source)
