from xml.etree import ElementTree as ET

from cf_h2o.eval.traffic_signal_resco_probe import _count_route_rows, _sumocfg_input_files


def test_sumocfg_input_files(tmp_path):
    cfg = tmp_path / "toy.sumocfg"
    root = ET.Element("configuration")
    input_node = ET.SubElement(root, "input")
    ET.SubElement(input_node, "net-file", value="toy.net.xml")
    ET.SubElement(input_node, "route-files", value="a.rou.xml,b.rou.xml")
    ET.ElementTree(root).write(cfg, encoding="utf-8", xml_declaration=True)

    nets, routes = _sumocfg_input_files(cfg)

    assert nets == ["toy.net.xml"]
    assert routes == ["a.rou.xml", "b.rou.xml"]


def test_count_route_rows(tmp_path):
    route = tmp_path / "toy.rou.xml"
    root = ET.Element("routes")
    ET.SubElement(root, "route", id="r", edges="a b")
    ET.SubElement(root, "vehicle", id="v", route="r", depart="0")
    ET.SubElement(root, "trip", id="t", **{"from": "a", "to": "b", "depart": "1"})
    ET.SubElement(root, "flow", id="f", **{"from": "a", "to": "b", "begin": "0", "end": "10"})
    ET.ElementTree(root).write(route, encoding="utf-8", xml_declaration=True)

    assert _count_route_rows(route) == {"vehicles": 1, "trips": 1, "flows": 1, "routes": 1}
