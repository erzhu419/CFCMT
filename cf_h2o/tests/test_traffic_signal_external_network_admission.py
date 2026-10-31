import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from cf_h2o.eval import traffic_signal_external_network_admission as admission


def _conversion_root(tmp_path: Path) -> Path:
    root = tmp_path / "conversion"
    network = root / "mini"
    network.mkdir(parents=True)
    (network / "mini.sumocfg").write_text(
        '<configuration><input><net-file value="mini.net.xml"/>'
        '<route-files value="mini.rou.xml"/></input></configuration>\n',
        encoding="utf-8",
    )
    (network / "mini.net.xml").write_text("<net/>\n", encoding="utf-8")
    (network / "mini.rou.xml").write_text(
        "<routes>"
        + "".join(
            f'<vehicle id="v{index}" depart="{index / 2:g}"/>'
            for index in range(5)
        )
        + "</routes>\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol": admission.EXPECTED_CONVERSION_PROTOCOL,
        "networks": {
            "mini": {
                "sumocfg": "mini/mini.sumocfg",
                "route_audit": {
                    "route_count": 5,
                    "missing_edge_route_count": 0,
                    "disconnected_route_count": 0,
                },
                "netconvert": {"returncode": 0},
                "connection_audit": {"missing_route_edge_pair_count": 0},
                "vehicle_count": 5,
            }
        },
    }
    (root / "conversion_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return root


def test_resolve_external_network_rejects_non_relocatable_path(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["networks"]["mini"]["sumocfg"] = (
        "/tmp/.conversion.staging-1/mini.sumocfg"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        admission._resolve_network(root, "mini")
    except ValueError as exc:
        assert "non-relocatable" in str(exc)
    else:
        raise AssertionError("absolute staging path was accepted")


def test_resolve_external_network_accepts_frozen_cityflow_v6(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = "cfcmt-cityflow-full-external-sumo-conversion-v6"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    sumocfg, resolved_manifest, network = admission._resolve_network(root, "mini")

    assert sumocfg == (root / "mini/mini.sumocfg").resolve()
    assert resolved_manifest["protocol"].endswith("conversion-v6")
    assert network["vehicle_count"] == 5


def test_resolve_external_network_requires_v7_tls_semantic_audit(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = "cfcmt-cityflow-full-external-sumo-conversion-v7"
    network = manifest["networks"]["mini"]
    network["controlled_intersection_count"] = 1
    network["tls_semantic_audit"] = {
        "protocol": (
            "cityflow-roadlink-to-netconvert-linkindex-remap-"
            "permissive-intergreen-v2"
        ),
        "controlled_intersection_count": 1,
        "phase_count": 2,
        "post_remap_phase_link_symmetric_difference": 1,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        admission._resolve_network(root, "mini")
    except ValueError as exc:
        assert "TLS semantic audit failed" in str(exc)
    else:
        raise AssertionError("v7 conversion with a mismatched TLS phase was accepted")

    network["tls_semantic_audit"][
        "post_remap_phase_link_symmetric_difference"
    ] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, _, resolved = admission._resolve_network(root, "mini")
    assert resolved["tls_semantic_audit"]["phase_count"] == 2


def test_resolve_external_network_accepts_v8_merge_yield_audit(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = "cfcmt-cityflow-full-external-sumo-conversion-v8"
    network = manifest["networks"]["mini"]
    network["controlled_intersection_count"] = 1
    network["tls_semantic_audit"] = {
        "protocol": (
            "cityflow-roadlink-to-netconvert-linkindex-remap-"
            "permissive-intergreen-merge-yield-v3"
        ),
        "controlled_intersection_count": 1,
        "phase_count": 2,
        "post_remap_phase_link_symmetric_difference": 0,
        "merge_yield_phase_link_count": 3,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, resolved_manifest, resolved = admission._resolve_network(root, "mini")

    assert resolved_manifest["protocol"].endswith("conversion-v8")
    assert resolved["tls_semantic_audit"]["merge_yield_phase_link_count"] == 3


def test_resolve_external_network_requires_v9_monotone_virtual_lane_audit(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = "cfcmt-cityflow-full-external-sumo-conversion-v9"
    network = manifest["networks"]["mini"]
    network["controlled_intersection_count"] = 1
    network["tls_semantic_audit"] = {
        "protocol": (
            "cityflow-roadlink-to-netconvert-linkindex-remap-"
            "permissive-intergreen-merge-yield-v3"
        ),
        "controlled_intersection_count": 1,
        "phase_count": 2,
        "post_remap_phase_link_symmetric_difference": 0,
    }
    network["virtual_lane_link_reduction"] = {
        "protocol": "cityflow-virtual-cartesian-lane-link-monotone-rank-v1",
        "source_unique_lane_pair_count": 30,
        "retained_unique_lane_pair_count": 6,
        "removed_unique_lane_pair_count": 24,
        "post_reduction_crossing_road_link_count": 1,
        "source_lane_coverage_failures": 0,
        "target_lane_coverage_failures": 0,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        admission._resolve_network(root, "mini")
    except ValueError as exc:
        assert "virtual lane-link audit failed" in str(exc)
    else:
        raise AssertionError("v9 conversion with a crossing virtual link was accepted")

    network["virtual_lane_link_reduction"][
        "post_reduction_crossing_road_link_count"
    ] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, resolved_manifest, resolved = admission._resolve_network(root, "mini")
    assert resolved_manifest["protocol"].endswith("conversion-v9")
    assert (
        resolved["virtual_lane_link_reduction"][
            "removed_unique_lane_pair_count"
        ]
        == 24
    )


def test_explicit_vehicle_inventory_uses_half_open_rollout_horizon(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)

    inventory = admission._explicit_vehicle_inventory(
        root / "mini/mini.sumocfg",
        horizon_sec=2.0,
    )

    assert inventory == {
        "total": 5,
        "before_horizon": 4,
        "at_or_after_horizon": 1,
        "minimum_departure_sec": 0.0,
        "maximum_departure_sec": 2.0,
    }


def test_demand_inventory_expands_deterministic_flow_and_separates_people(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rich"
    root.mkdir()
    (root / "run.sumocfg").write_text(
        '<configuration><input><route-files value="demand.rou.xml"/>'
        "</input></configuration>\n",
        encoding="utf-8",
    )
    (root / "demand.rou.xml").write_text(
        "<routes>"
        '<vehicle id="v1" depart="1"/>'
        '<vehicle id="v2" depart="7"/>'
        '<flow id="f" begin="0" end="10" period="3"/>'
        '<person id="p1" depart="2"/>'
        '<person id="p2" depart="7"/>'
        "</routes>\n",
        encoding="utf-8",
    )

    inventory = admission._explicit_vehicle_inventory(
        root / "run.sumocfg",
        horizon_sec=7.0,
    )

    assert inventory == {
        "total": 6,
        "before_horizon": 4,
        "at_or_after_horizon": 2,
        "minimum_departure_sec": 0.0,
        "maximum_departure_sec": 9.0,
        "demand_protocol": "deterministic-vehicle-flow-person-v1",
        "explicit_vehicle_count": 2,
        "flow_vehicle_count": 4,
        "deterministic_flow_count": 1,
        "triggered_vehicle_count": 0,
        "person_total": 2,
        "person_before_horizon": 1,
        "person_at_or_after_horizon": 1,
    }


def test_demand_inventory_accepts_human_readable_departures(
    tmp_path: Path,
) -> None:
    (tmp_path / "run.sumocfg").write_text(
        '<configuration><input><route-files value="demand.rou.xml"/>'
        "</input></configuration>\n",
        encoding="utf-8",
    )
    (tmp_path / "demand.rou.xml").write_text(
        '<routes><vehicle id="pt" depart="1:19:46:50"/></routes>\n',
        encoding="utf-8",
    )

    inventory = admission._explicit_vehicle_inventory(
        tmp_path / "run.sumocfg", horizon_sec=160_000.0
    )

    assert inventory["total"] == 1
    assert inventory["before_horizon"] == 1
    assert inventory["minimum_departure_sec"] == 157_610.0
    assert inventory["maximum_departure_sec"] == 157_610.0


def test_demand_inventory_reads_explicit_vehicles_from_additional_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "run.sumocfg").write_text(
        '<configuration><input><route-files value="routes.rou.xml"/>'
        '<additional-files value="tls.add.xml,emitters.emi.xml"/>'
        "</input></configuration>\n",
        encoding="utf-8",
    )
    (tmp_path / "routes.rou.xml").write_text(
        '<routes><route id="r" edges="a b"/></routes>\n', encoding="utf-8"
    )
    (tmp_path / "tls.add.xml").write_text(
        '<additional><tlLogic id="t"><phase duration="30" state="G"/>'
        "</tlLogic></additional>\n",
        encoding="utf-8",
    )
    (tmp_path / "emitters.emi.xml").write_text(
        '<additional><vehicle id="v0" route="r" depart="0"/>'
        '<vehicle id="v1" route="r" depart="5"/></additional>\n',
        encoding="utf-8",
    )

    inventory = admission._explicit_vehicle_inventory(
        tmp_path / "run.sumocfg", horizon_sec=5.0
    )

    assert inventory == {
        "total": 2,
        "before_horizon": 1,
        "at_or_after_horizon": 1,
        "minimum_departure_sec": 0.0,
        "maximum_departure_sec": 5.0,
    }


def test_demand_inventory_reads_gzip_route_and_additional_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "run.sumocfg").write_text(
        '<configuration><input><route-files value="routes.rou.xml.gz"/>'
        '<additional-files value="people.add.xml.gz"/>'
        "</input></configuration>\n",
        encoding="utf-8",
    )
    with gzip.open(tmp_path / "routes.rou.xml.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            '<routes><vehicle id="v0" depart="0"/>'
            '<flow id="f0" begin="2" end="8" period="2"/></routes>\n'
        )
    with gzip.open(tmp_path / "people.add.xml.gz", "wt", encoding="utf-8") as handle:
        handle.write('<additional><person id="p0" depart="3"/></additional>\n')

    inventory = admission._explicit_vehicle_inventory(
        tmp_path / "run.sumocfg", horizon_sec=6.0
    )

    assert inventory == {
        "total": 4,
        "before_horizon": 3,
        "at_or_after_horizon": 1,
        "minimum_departure_sec": 0.0,
        "maximum_departure_sec": 6.0,
        "demand_protocol": "deterministic-vehicle-flow-person-v1",
        "explicit_vehicle_count": 1,
        "flow_vehicle_count": 3,
        "deterministic_flow_count": 1,
        "triggered_vehicle_count": 0,
        "person_total": 1,
        "person_before_horizon": 1,
        "person_at_or_after_horizon": 0,
    }


def test_demand_inventory_resolves_triggered_vehicle_and_ignores_calibrator_flow(
    tmp_path: Path,
) -> None:
    (tmp_path / "run.sumocfg").write_text(
        '<configuration><input><route-files value="demand.rou.xml"/>'
        '<additional-files value="calib.add.xml"/>'
        "</input></configuration>\n",
        encoding="utf-8",
    )
    (tmp_path / "demand.rou.xml").write_text(
        """<routes>
  <vehicle id="car0" depart="triggered"><route edges="a b"/></vehicle>
  <person id="p0" depart="12"><ride from="a" to="b" lines="car0"/></person>
</routes>""",
        encoding="utf-8",
    )
    (tmp_path / "calib.add.xml").write_text(
        '<additional><calibrator id="c0" edge="a" pos="0">'
        '<flow begin="0" end="100" type="passenger"/>'
        "</calibrator></additional>\n",
        encoding="utf-8",
    )

    inventory = admission._explicit_vehicle_inventory(
        tmp_path / "run.sumocfg", horizon_sec=100.0
    )

    assert inventory == {
        "total": 1,
        "before_horizon": 1,
        "at_or_after_horizon": 0,
        "minimum_departure_sec": 12.0,
        "maximum_departure_sec": 12.0,
        "demand_protocol": "deterministic-vehicle-flow-person-v1",
        "explicit_vehicle_count": 1,
        "flow_vehicle_count": 0,
        "deterministic_flow_count": 0,
        "triggered_vehicle_count": 1,
        "person_total": 1,
        "person_before_horizon": 1,
        "person_at_or_after_horizon": 0,
    }


def test_resolve_external_network_accepts_native_roadnetsz_package(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = "cfcmt-roadnetsz-native-sumo-window-package-v1"
    network = manifest["networks"]["mini"]
    network.pop("netconvert")
    network["controlled_intersection_count"] = 36
    network["source_window"] = {
        "routing_retention": 0.92,
        "minimum_required_retention": 0.90,
        "routed_vehicle_count": 5,
    }
    network["network_build"] = {
        "protocol": "roadnetsz-native-sumo-window-duarouter-v1",
        "mode": "source_native_sumo_with_strict_explicit_routing",
        "duarouter": {"returncode": 0},
    }
    network["tls_semantic_audit"] = {
        "protocol": "roadnetsz-native-sumo-tls-preserved-v1",
        "controlled_intersection_count": 36,
        "phase_count": 120,
        "source_tls_programs_preserved_byte_for_byte": True,
        "source_net_sha256": "same",
        "packaged_net_sha256": "same",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, resolved_manifest, resolved = admission._resolve_network(root, "mini")

    assert resolved_manifest["protocol"].endswith("window-package-v1")
    assert resolved["source_window"]["routed_vehicle_count"] == 5


def test_resolve_external_network_accepts_native_dlr_bologna_package(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = admission.DLR_BOLOGNA_NATIVE_V1
    network = manifest["networks"]["mini"]
    network.pop("netconvert")
    network["controlled_intersection_count"] = 7
    network["person_count"] = 2
    network["network_build"] = {
        "protocol": "dlr-bologna-native-read-only-package-v1",
        "mode": "source_native_sumo_read_only_inputs",
        "source_network_and_demand_files_preserved_byte_for_byte": True,
        "passive_output_definitions_removed": True,
        "traffic_semantics_rebuilt_or_calibrated": False,
    }
    network["demand_audit"] = {
        "protocol": "dlr-bologna-deterministic-demand-inventory-v1",
        "all_source_demand_files_loaded": True,
        "total": 5,
        "person_total": 2,
    }
    network["tls_semantic_audit"] = {
        "protocol": "dlr-bologna-native-sumo-tls-preserved-v1",
        "controlled_intersection_count": 7,
        "program_count": 7,
        "phase_count": 28,
        "source_net_and_tls_files_preserved_byte_for_byte": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, resolved_manifest, resolved = admission._resolve_network(root, "mini")

    assert resolved_manifest["protocol"] == admission.DLR_BOLOGNA_NATIVE_V1
    assert resolved["person_count"] == 2


def test_resolve_external_network_accepts_native_figshare_xuancheng_package(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = admission.FIGSHARE_XUANCHENG_NATIVE_V1
    network = manifest["networks"]["mini"]
    network.pop("netconvert")
    network["controlled_intersection_count"] = 132
    network["person_count"] = 0
    network["network_build"] = {
        "protocol": "figshare-xuancheng-native-sumo-read-only-v1",
        "mode": "source_native_sumo_read_only_with_cityflow_anchor_completion",
        "source_network_preserved_byte_for_byte": True,
        "traffic_signal_programs_preserved_byte_for_byte": True,
        "traffic_semantics_rebuilt_or_calibrated": False,
        "route_completion_protocol": (
            "native-sumo-length-dijkstra-cityflow-anchor-completion-v2"
        ),
    }
    network["demand_audit"] = {
        "protocol": "figshare-xuancheng-cityflow-anchor-completion-v1",
        "all_source_flow_records_loaded": True,
        "all_anchor_edges_preserved_in_order": True,
        "all_completed_route_pairs_native_sumo_connected": True,
        "source_flow_record_count": 5,
        "total": 5,
    }
    network["tls_semantic_audit"] = {
        "protocol": "figshare-xuancheng-native-sumo-tls-preserved-v1",
        "controlled_intersection_count": 132,
        "program_count": 132,
        "phase_count": 969,
        "source_native_sumo_tls_preserved_byte_for_byte": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, resolved_manifest, resolved = admission._resolve_network(root, "mini")

    assert resolved_manifest["protocol"] == admission.FIGSHARE_XUANCHENG_NATIVE_V1
    assert resolved["controlled_intersection_count"] == 132


def test_resolve_external_network_accepts_native_lust_due_static_package(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = admission.LUST_NATIVE_DUE_STATIC_V1
    network = manifest["networks"]["mini"]
    network.pop("netconvert")
    network["controlled_intersection_count"] = 201
    network["person_count"] = 0
    network["network_build"] = {
        "protocol": "lust-native-sumo-read-only-due-static-v1",
        "mode": "source_native_sumo_due_static_read_only_inputs",
        "source_network_and_demand_files_preserved_byte_for_byte": True,
        "passive_output_detector_and_polygon_files_removed": True,
        "traffic_semantics_rebuilt_or_calibrated": False,
        "modern_sumo_real_data_validation_claimed": False,
    }
    network["demand_audit"] = {
        "protocol": "lust-due-static-complete-demand-inventory-v1",
        "all_source_demand_files_loaded": True,
        "source_network_and_demand_files_preserved_byte_for_byte": True,
        "duplicate_vehicle_id_count": 0,
        "embedded_route_count": 5,
        "total": 5,
    }
    network["tls_semantic_audit"] = {
        "protocol": "lust-native-static-tls-preserved-v1",
        "controlled_intersection_count": 201,
        "program_count": 201,
        "phase_count": 1298,
        "source_native_net_and_static_tls_preserved_byte_for_byte": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, resolved_manifest, resolved = admission._resolve_network(root, "mini")

    assert resolved_manifest["protocol"] == admission.LUST_NATIVE_DUE_STATIC_V1
    assert resolved["controlled_intersection_count"] == 201


def test_resolve_external_network_accepts_native_dublin_urban_package(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = admission.DUBLIN_URBAN_NATIVE_V2
    network = manifest["networks"]["mini"]
    network.pop("netconvert")
    network["controlled_intersection_count"] = 204
    network["person_count"] = 0
    network["network_build"] = {
        "protocol": "dublin-urban-native-read-only-package-v2",
        "mode": "source_native_sumo_full_day_read_only_inputs",
        "source_network_route_emitter_tls_files_preserved_byte_for_byte": True,
        "all_six_source_vehicle_composition_scenarios_packaged": True,
        "passive_detector_and_poi_definition_removed": True,
        "source_config_input_order_preserved_except_passive_detector": True,
        "packaged_additional_file_order": [
            "../common/DCC_trafficlights.add.xml",
            "vtypes.add.xml",
            "../common/DCC_routes.rou.xml",
            "../common/DCC_emitters.emi.xml",
        ],
        "traffic_semantics_rebuilt_or_calibrated": False,
        "source_horizon_sec": 86_400.0,
        "source_step_length_sec": 0.5,
    }
    network["demand_audit"] = {
        "protocol": "dublin-urban-full-day-explicit-demand-v1",
        "all_source_demand_files_loaded": True,
        "source_network_route_emitter_tls_files_preserved_byte_for_byte": True,
        "total": 5,
        "explicit_vehicle_count": 5,
        "trip_count": 0,
        "flow_count": 0,
        "duplicate_vehicle_id_count": 0,
        "unknown_vehicle_route_reference_count": 0,
        "unknown_distribution_route_reference_count": 0,
    }
    network["tls_semantic_audit"] = {
        "protocol": "dublin-urban-native-sumo-tls-preserved-v1",
        "controlled_intersection_count": 204,
        "program_count": 204,
        "phase_count": 1200,
        "source_native_net_and_tls_preserved_byte_for_byte": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, resolved_manifest, resolved = admission._resolve_network(root, "mini")

    assert resolved_manifest["protocol"] == admission.DUBLIN_URBAN_NATIVE_V2
    assert resolved["demand_audit"]["explicit_vehicle_count"] == 5


def test_resolve_external_network_accepts_native_dlr_badhersfeld_package(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = admission.DLR_BADHERSFELD_NATIVE_V1
    network = manifest["networks"]["mini"]
    network.pop("netconvert")
    network["controlled_intersection_count"] = 21
    network["person_count"] = 3
    network["additional_files"] = [
        "../osm/pt_vtypes.xml",
        "../osm/gtfs_publictransport.add.xml",
        "../osm/gtfs_publictransport.rou.xml",
        "../osm/defaults/basic.vType.xml",
        "../osm/osm_complete_parking_areas.add.xml",
        "../osm/osm_parking_rerouters.add.xml",
        "../osm/obstacle_seilerweg.add.xml",
        "../demand/osm_activitygen.lkw.rou.xml",
        "../osm/calib.add.xml",
    ]
    network["dropped_passive_output_files"] = [
        "../osm/osm_polygons.add.xml"
    ]
    network["removed_passive_config_tags"] = [
        "device.taxi.dispatch-algorithm.output",
        "device.taxi.idle-algorithm.output",
        "gui_only",
    ]
    network["network_build"] = {
        "protocol": "dlr-badhersfeld-native-read-only-package-v1",
        "mode": "source_native_sumo_full_day_read_only_inputs",
        "source_network_and_demand_files_preserved_byte_for_byte": True,
        "complete_source_directory_preserved": True,
        "source_input_order_preserved_except_passive_polygon": True,
        "passive_polygon_gui_and_output_destinations_removed": True,
        "published_calibrator_input_preserved": True,
        "source_dynamic_route_repair_preserved": True,
        "traffic_semantics_rebuilt_or_calibrated": False,
        "source_horizon_sec": 86_400.0,
        "source_step_length_sec": 1.0,
    }
    network["dynamic_routing_audit"] = {
        "ignore_route_errors": "true",
        "rerouting_probability": "1",
        "rerouting_period_sec": "300",
        "rerouting_pre_period_sec": "300",
    }
    network["demand_audit"] = {
        "protocol": "dlr-badhersfeld-complete-rich-demand-inventory-v1",
        "all_source_demand_files_loaded": True,
        "all_triggered_vehicles_resolved_to_person_plans": True,
        "published_calibrator_input_preserved": True,
        "source_network_and_demand_files_preserved_byte_for_byte": True,
        "total": 5,
        "person_total": 3,
        "explicit_vehicle_count": 5,
        "triggered_vehicle_count": 2,
        "road_demand_id_count": 5,
        "person_demand_id_count": 3,
        "duplicate_road_demand_id_count": 0,
        "duplicate_person_demand_id_count": 0,
        "nested_calibrator_flow_count": 6,
        "at_or_after_horizon": 0,
        "person_at_or_after_horizon": 0,
    }
    network["tls_semantic_audit"] = {
        "protocol": "dlr-badhersfeld-native-sumo-tls-preserved-v1",
        "controlled_intersection_count": 21,
        "program_count": 21,
        "phase_count": 141,
        "source_native_net_and_tls_preserved_byte_for_byte": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, resolved_manifest, resolved = admission._resolve_network(root, "mini")

    assert resolved_manifest["protocol"] == admission.DLR_BADHERSFELD_NATIVE_V1
    assert resolved["demand_audit"]["triggered_vehicle_count"] == 2


def test_resolve_external_network_accepts_sumo122_rebuilt_roadnetsz_package(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = "cfcmt-roadnetsz-native-sumo122-window-package-v2"
    network = manifest["networks"]["mini"]
    network.pop("netconvert")
    network["controlled_intersection_count"] = 140
    network["source_window"] = {
        "routing_retention": 0.92,
        "minimum_required_retention": 0.90,
        "routed_vehicle_count": 5,
    }
    network["network_build"] = {
        "protocol": "roadnetsz-source-components-sumo122-window-duarouter-v2",
        "mode": (
            "source_components_rebuilt_with_sumo122_and_strict_explicit_routing"
        ),
        "netconvert": {"returncode": 0},
        "duarouter": {"returncode": 0},
    }
    network["tls_semantic_audit"] = {
        "protocol": "roadnetsz-source-tll-sumo122-rebuild-v2",
        "declared_tls_node_count": 140,
        "explicit_source_tls_id_count": 36,
        "rebuilt_tls_id_count": 140,
        "auto_generated_tls_id_count": 104,
        "rebuilt_phase_count": 433,
        "missing_declared_tls_id_count": 0,
        "unexpected_rebuilt_tls_id_count": 0,
        "missing_explicit_tls_program_count": 0,
        "explicit_phase_count_mismatch_count": 0,
        "all_declared_tls_nodes_rebuilt": True,
        "explicit_source_tls_program_phase_counts_preserved": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, resolved_manifest, resolved = admission._resolve_network(root, "mini")

    assert resolved_manifest["protocol"].endswith("window-package-v2")
    assert resolved["controlled_intersection_count"] == 140

    manifest["protocol"] = "cfcmt-roadnetsz-native-sumo122-window-package-v3"
    network["network_build"] = {
        "protocol": (
            "roadnetsz-source-components-sumo122-turnspeed55-"
            "window-duarouter-v3"
        ),
        "mode": (
            "source_components_rebuilt_with_sumo122_turnspeed55_"
            "and_strict_explicit_routing"
        ),
        "junction_turn_speed_limit_mps": 5.5,
        "netconvert": {"returncode": 0},
        "duarouter": {"returncode": 0},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, resolved_manifest, _ = admission._resolve_network(root, "mini")

    assert resolved_manifest["protocol"].endswith("window-package-v3")

    manifest["protocol"] = "cfcmt-roadnetsz-native-sumo122-window-package-v4"
    network["network_build"] = {
        "protocol": (
            "roadnetsz-source-components-sumo122-protected-incoming-"
            "window-duarouter-v4"
        ),
        "mode": (
            "source_components_rebuilt_with_sumo122_protected_"
            "incoming_and_strict_explicit_routing"
        ),
        "auto_generated_tls_phase_layout": "incoming",
        "netconvert": {"returncode": 0},
        "duarouter": {"returncode": 0},
    }
    network["tls_semantic_audit"].update(
        {
            "auto_generated_phase_layout": "incoming",
            "auto_generated_minor_green_signal_count": 0,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, resolved_manifest, _ = admission._resolve_network(root, "mini")

    assert resolved_manifest["protocol"].endswith("window-package-v4")


def test_external_admission_rejects_conversion_hash_mismatch(
    tmp_path: Path,
) -> None:
    root = _conversion_root(tmp_path)

    try:
        admission.run_external_network_admission(
            conversion_root=root,
            scenario="mini",
            seed=5057,
            horizon_sec=3.0,
            expected_sumo_version="1.22.0",
            minimum_controllable_tls=1,
            maximum_collision_count=0,
            maximum_teleport_fraction=0.0,
            expected_conversion_manifest_sha256="0" * 64,
        )
    except ValueError as exc:
        assert "manifest SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("mismatched conversion manifest hash was accepted")


def test_partial_horizon_allows_sumo_parser_lookahead() -> None:
    inventory = {
        "total": 100,
        "before_horizon": 40,
        "at_or_after_horizon": 60,
        "minimum_departure_sec": 0.0,
        "maximum_departure_sec": 1000.0,
    }

    assert admission._explicit_demand_loading_checks(
        inventory,
        summary_loaded=47,
        horizon_sec=600.0,
    ) == {
        "demand_loading_consistent": True,
        "demand_conservation": True,
    }


def test_full_horizon_requires_all_explicit_demand_loaded() -> None:
    inventory = {
        "total": 100,
        "before_horizon": 99,
        "at_or_after_horizon": 1,
        "minimum_departure_sec": 0.0,
        "maximum_departure_sec": 600.0,
    }

    assert admission._explicit_demand_loading_checks(
        inventory,
        summary_loaded=99,
        horizon_sec=600.0,
    ) == {
        "demand_loading_consistent": True,
        "demand_conservation": False,
    }


def test_full_horizon_external_admission_accepts_eagerly_loaded_demand(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _conversion_root(tmp_path)

    class Simulation:
        time = 0.0

        def getTime(self):
            return self.time

        def getMinExpectedNumber(self):
            return max(5 - int(self.time), 0)

        def getLoadedNumber(self):
            return 0

        def getDepartedNumber(self):
            return int(self.time <= 3)

        def getArrivedNumber(self):
            return int(self.time >= 2)

        def getStartingTeleportNumber(self):
            return 0

        def getEndingTeleportNumber(self):
            return 0

        def getCollisions(self):
            return ()

        def getPendingVehicles(self):
            return ()

    class TrafficLight:
        @staticmethod
        def getIDList():
            return ("tls0",)

    class Vehicle:
        @staticmethod
        def getIDCount():
            return 2

    class FakeSumo:
        def __init__(self):
            self.simulation = Simulation()
            self.trafficlight = TrafficLight()
            self.vehicle = Vehicle()

        def simulationStep(self):
            self.simulation.time += 1.0

        def close(self):
            return None

    fake = FakeSumo()
    monkeypatch.setattr(admission, "load_libsumo", lambda: fake)
    monkeypatch.setattr(admission, "libsumo_version", lambda: "1.22.0")
    def fake_start_sumo(*args, summary_output: Path, **kwargs) -> None:
        summary_output.write_text(
            '<summary><step time="3" loaded="5" inserted="3" '
            'running="2" waiting="2" ended="1" arrived="1" '
            'collisions="0" teleports="0"/></summary>\n',
            encoding="utf-8",
        )

    monkeypatch.setattr(admission, "_start_admission_sumo", fake_start_sumo)
    monkeypatch.setattr(admission, "runtime_metadata", lambda: {"libsumo_version": "1.22.0"})
    monkeypatch.setattr(
        admission,
        "_scenario_input_fingerprint",
        lambda path: {"sumocfg": str(path)},
    )

    result = admission.run_external_network_admission(
        conversion_root=root,
        scenario="mini",
        seed=5057,
        horizon_sec=3.0,
        expected_sumo_version="1.22.0",
        minimum_controllable_tls=1,
        maximum_collision_count=0,
        maximum_teleport_fraction=0.01,
        progress_interval_sec=3,
    )

    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["observed"]["traci_step_loaded_total"] == 0
    assert result["observed"]["summary"]["loaded"] == 5
    assert result["checks"]["demand_conservation"] is True
    assert result["observed"]["controllable_tls_count"] == 1
    assert result["observed"]["unique_collision_incidents"] == 0
    assert result["observed"]["final_time_sec"] == 3.0
    assert result["observed"]["terminated_early"] is False


def test_external_admission_stops_after_irreversible_collision_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _conversion_root(tmp_path)

    class Simulation:
        time = 0.0

        def getTime(self):
            return self.time

        def getMinExpectedNumber(self):
            return 5

        def getLoadedNumber(self):
            return 1

        def getDepartedNumber(self):
            return 1

        def getArrivedNumber(self):
            return 0

        def getStartingTeleportNumber(self):
            return 0

        def getEndingTeleportNumber(self):
            return 0

        def getCollisions(self):
            return (
                SimpleNamespace(
                    collider="v0",
                    victim="v1",
                    type="junction",
                    lane=":tls0_0_0",
                    pos=1.0,
                ),
            )

        def getPendingVehicles(self):
            return ()

    class TrafficLight:
        @staticmethod
        def getIDList():
            return ("tls0",)

    class Vehicle:
        @staticmethod
        def getIDCount():
            return 2

    class FakeSumo:
        def __init__(self):
            self.simulation = Simulation()
            self.trafficlight = TrafficLight()
            self.vehicle = Vehicle()
            self.step_count = 0

        def simulationStep(self):
            self.step_count += 1
            self.simulation.time += 1.0

        def close(self):
            return None

    fake = FakeSumo()
    monkeypatch.setattr(admission, "load_libsumo", lambda: fake)
    monkeypatch.setattr(admission, "libsumo_version", lambda: "1.22.0")

    def fake_start_sumo(*args, summary_output: Path, **kwargs) -> None:
        summary_output.write_text(
            '<summary><step time="1" loaded="1" inserted="1" '
            'running="1" waiting="0" ended="0" arrived="0" '
            'collisions="1" teleports="0"/></summary>\n',
            encoding="utf-8",
        )

    monkeypatch.setattr(admission, "_start_admission_sumo", fake_start_sumo)
    monkeypatch.setattr(
        admission,
        "runtime_metadata",
        lambda: {"libsumo_version": "1.22.0"},
    )
    monkeypatch.setattr(
        admission,
        "_scenario_input_fingerprint",
        lambda path: {"sumocfg": str(path)},
    )

    result = admission.run_external_network_admission(
        conversion_root=root,
        scenario="mini",
        seed=5057,
        horizon_sec=3.0,
        expected_sumo_version="1.22.0",
        minimum_controllable_tls=1,
        maximum_collision_count=0,
        maximum_teleport_fraction=0.0,
        progress_interval_sec=3,
    )

    assert result["passed"] is False
    assert fake.step_count == 1
    assert result["checks"]["collision_limit"] is False
    assert result["checks"]["horizon_reached"] is False
    assert result["observed"]["terminated_early"] is True
    assert result["observed"]["termination_reason"] == (
        "unique_collision_limit_irreversibly_exceeded"
    )
    assert result["observed"]["unique_collision_incidents"] == 1
    assert result["observed"]["final_time_sec"] == 1.0
