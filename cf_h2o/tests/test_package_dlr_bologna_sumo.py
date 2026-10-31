from pathlib import Path

from scripts.data.package_dlr_bologna_sumo import (
    HORIZON_SEC,
    PROTOCOL,
    canonical_config_inputs,
)


def test_bologna_package_keeps_behavioral_inputs_and_drops_passive_outputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "run.sumocfg").write_text(
        "<configuration><input>"
        '<net-file value="city.net.xml"/>'
        '<route-files value="cars.rou.xml"/>'
        '<additional-files value="buses.rou.xml,detectors.add.xml,'
        'stops.add.xml,tls.add.xml,vtypes.add.xml"/>'
        "</input></configuration>\n",
        encoding="utf-8",
    )
    (tmp_path / "buses.rou.xml").write_text("<routes/>\n", encoding="utf-8")
    (tmp_path / "detectors.add.xml").write_text(
        '<additional><e1Detector id="d" lane="e_0" pos="1"/></additional>\n',
        encoding="utf-8",
    )
    (tmp_path / "stops.add.xml").write_text(
        '<additional><busStop id="s" lane="e_0" startPos="0" '
        'endPos="1"/></additional>\n',
        encoding="utf-8",
    )
    (tmp_path / "tls.add.xml").write_text(
        '<additional><tlLogic id="j"><phase duration="30" state="G"/>'
        "</tlLogic></additional>\n",
        encoding="utf-8",
    )
    (tmp_path / "vtypes.add.xml").write_text(
        '<additional><vType id="car"/></additional>\n',
        encoding="utf-8",
    )

    inputs = canonical_config_inputs(
        source_directory=tmp_path,
        source_config=tmp_path / "run.sumocfg",
    )

    assert PROTOCOL == "cfcmt-dlr-bologna-native-sumo-package-v1"
    assert HORIZON_SEC == 3600.0
    assert inputs == {
        "net_files": ["city.net.xml"],
        "route_files": ["cars.rou.xml", "buses.rou.xml"],
        "additional_files": [
            "stops.add.xml",
            "tls.add.xml",
            "vtypes.add.xml",
        ],
        "dropped_passive_output_files": ["detectors.add.xml"],
    }
