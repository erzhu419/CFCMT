import json
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.data import acquire_lust_scenario as acquisition
from scripts.data import package_lust_sumo as package


def _acquisition(tmp_path: Path) -> Path:
    root = tmp_path / "acquisition"
    scenario = root / "scenario"
    (scenario / "DUERoutes").mkdir(parents=True)
    net = (
        '<net version="0.27">'
        '<edge id="a" from="n0" to="n1"><lane id="a_0" index="0" length="10"/></edge>'
        '<edge id="b" from="n1" to="n2"><lane id="b_0" index="0" length="10"/></edge>'
        '<junction id="n0" type="dead_end"/>'
        '<junction id="n1" type="traffic_light"/>'
        '<junction id="n2" type="dead_end"/>'
        '<connection from="a" to="b" fromLane="0" toLane="0" tl="n1"/>'
        '<tlLogic id="n1" programID="0"><phase duration="30" state="G"/></tlLogic>'
        '</net>\n'
    )
    (scenario / "lust.net.xml").write_text(net, encoding="utf-8")
    (scenario / "tll.static.xml").write_text(
        '<additional><tlLogics><tlLogic id="n1" programID="0">'
        '<phase duration="30" state="G"/></tlLogic></tlLogics></additional>\n',
        encoding="utf-8",
    )
    (scenario / "vtypes.add.xml").write_text(
        '<routes><vType id="car"/></routes>\n', encoding="utf-8"
    )
    (scenario / "busstops.add.xml").write_text("<add/>\n", encoding="utf-8")
    departures = [0, 10, 20, 30, 86_395]
    for relative, departure in zip(acquisition.DEMAND_FILES, departures):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = path.name.replace(".", "_")
        path.write_text(
            f'<routes><vehicle id="{identity}" type="car" depart="{departure}">'
            '<route edges="a b"/></vehicle></routes>\n',
            encoding="utf-8",
        )
    (scenario / "due.static.sumocfg").write_text(
        "<configuration/>\n", encoding="utf-8"
    )
    for name in ("CHANGELOG.md", "LICENSE.md", "README.md"):
        (root / name).write_text(f"{name}\n", encoding="utf-8")
    files = {}
    for relative in acquisition.FILES:
        path = root / relative
        files[relative] = {"size_bytes": path.stat().st_size}
    (root / "acquisition_manifest.json").write_text(
        json.dumps(
            {
                "protocol": acquisition.PROTOCOL,
                "commit": acquisition.COMMIT,
                "demand_files": list(acquisition.DEMAND_FILES),
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_package_lust_preserves_full_due_static_variant(tmp_path: Path) -> None:
    source = _acquisition(tmp_path)
    output = tmp_path / "package"

    payload = package.package_lust(
        acquisition_root=source,
        output_root=output,
    )

    network = payload["networks"][package.SCENARIO]
    assert network["vehicle_count"] == 5
    assert network["controlled_intersection_count"] == 1
    assert network["demand_audit"]["all_source_demand_files_loaded"] is True
    assert network["demand_audit"]["maximum_departure_sec"] == 86_395
    assert (output / package.SCENARIO / "lust.net.xml").read_bytes() == (
        source / "scenario/lust.net.xml"
    ).read_bytes()
    config = ET.parse(
        output / package.SCENARIO / f"{package.SCENARIO}.sumocfg"
    ).getroot()
    assert config.find(".//end").attrib["value"] == "86400"
    assert "e1detectors" not in ET.tostring(config, encoding="unicode")
