import json
from pathlib import Path

from scripts.data import convert_libsignal_external_to_sumo as base_converter
from scripts.data.convert_cityflow_external_confirmation_to_sumo import (
    JINAN_VARIANTS,
    PROTOCOL,
    convert_external_confirmation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBSIGNAL_ROOT = (
    PROJECT_ROOT
    / "H2Oplus/downloads/traffic_signal_libsignal/repo/data/raw_data"
)
JINAN_ROOT = (
    PROJECT_ROOT
    / "H2Oplus/downloads/traffic_signal_external/colight/data/Jinan/3_4"
)


def test_external_confirmation_conversion_uses_all_jinan_flows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_netconvert(
        netconvert: Path,
        *,
        node_file: Path,
        edge_file: Path,
        output_file: Path,
        connection_file: Path | None = None,
        tls_file: Path | None = None,
        artifact_root: Path | None = None,
    ) -> dict:
        del node_file, edge_file, artifact_root
        assert connection_file is not None
        assert tls_file is not None
        connections = base_converter.ET.parse(connection_file).getroot()
        tls = base_converter.ET.parse(tls_file).getroot()
        net = base_converter.ET.Element("net")
        controlled = [
            row for row in connections.findall("connection") if row.get("tl")
        ]
        counts: dict[str, int] = {}
        for row in controlled:
            counts[str(row.get("tl"))] = counts.get(str(row.get("tl")), 0) + 1
        for row in connections.findall("connection"):
            attributes = dict(row.attrib)
            if attributes.get("tl"):
                attributes["linkIndex"] = str(
                    counts[attributes["tl"]] - 1 - int(attributes["linkIndex"])
                )
            base_converter.ET.SubElement(net, "connection", attributes)
        for logic in tls.findall("tlLogic"):
            net.append(
                base_converter.ET.fromstring(base_converter.ET.tostring(logic))
            )
        base_converter._write_xml(output_file, net)
        return {
            "command": [str(netconvert)],
            "returncode": 0,
            "stdout_tail": "Success.\n",
            "stderr_tail": "",
        }

    monkeypatch.setattr(base_converter, "_run_netconvert", fake_netconvert)
    netconvert = tmp_path / "netconvert"
    netconvert.write_text("test runtime\n", encoding="utf-8")
    output_root = tmp_path / "external_cityflow_v6"
    commit = "0" * 40

    payload = convert_external_confirmation(
        la_roadnet=LIBSIGNAL_ROOT / "LA_1x4/roadnet_LA.json",
        la_flow=LIBSIGNAL_ROOT / "LA_1x4/LA.json",
        jinan_roadnet=JINAN_ROOT / "roadnet_3_4.json",
        jinan_flow_dir=JINAN_ROOT,
        jinan_repository_url="https://example.test/colight.git",
        jinan_repository_commit=commit,
        output_root=output_root,
        netconvert=netconvert,
    )

    manifest = json.loads(
        (output_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == payload
    assert manifest["protocol"] == PROTOCOL
    assert manifest["source_repositories"]["jinan"]["commit"] == commit
    assert manifest["city_groups"] == {
        "jinan": [scenario for scenario, _ in JINAN_VARIANTS],
        "los_angeles": ["la_1x4"],
    }
    assert set(manifest["networks"]) == {
        "la_1x4",
        *(scenario for scenario, _ in JINAN_VARIANTS),
    }
    assert manifest["networks"]["la_1x4"]["vehicle_count"] == 2203
    assert [
        manifest["networks"][scenario]["vehicle_count"]
        for scenario, _ in JINAN_VARIANTS
    ] == [6295, 4365, 5494]
    for scenario, _ in JINAN_VARIANTS:
        network = manifest["networks"][scenario]
        assert network["city"] == "jinan"
        assert network["controlled_intersection_count"] == 12
        assert network["route_audit"]["missing_edge_route_count"] == 0
        assert network["route_audit"]["disconnected_route_count"] == 0
        assert (
            network["tls_semantic_audit"][
                "post_remap_phase_link_symmetric_difference"
            ]
            == 0
        )
    assert ".staging-" not in json.dumps(manifest, sort_keys=True)
    for network in manifest["networks"].values():
        assert (output_root / network["sumocfg"]).is_file()


def test_external_confirmation_rejects_abbreviated_repository_commit(
    tmp_path: Path,
) -> None:
    netconvert = tmp_path / "netconvert"
    netconvert.write_text("test runtime\n", encoding="utf-8")

    try:
        convert_external_confirmation(
            la_roadnet=LIBSIGNAL_ROOT / "LA_1x4/roadnet_LA.json",
            la_flow=LIBSIGNAL_ROOT / "LA_1x4/LA.json",
            jinan_roadnet=JINAN_ROOT / "roadnet_3_4.json",
            jinan_flow_dir=JINAN_ROOT,
            jinan_repository_url="https://example.test/colight.git",
            jinan_repository_commit="04440b7",
            output_root=tmp_path / "external_cityflow_v6",
            netconvert=netconvert,
        )
    except ValueError as exc:
        assert "40-character" in str(exc)
    else:
        raise AssertionError("abbreviated repository commit was accepted")
