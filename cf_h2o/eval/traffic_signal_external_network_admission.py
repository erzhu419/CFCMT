"""Run full-horizon safety admission for converted external TSC networks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import _collision_incident_key
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _scenario_input_fingerprint,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import _sumo_arguments
from cf_h2o.sumo_runtime import (
    libsumo_version,
    load_libsumo,
    runtime_metadata,
    sumo_state_directory,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.sumo_static_inputs import (
    DEMAND_ROOT_TAGS,
    iterparse_xml,
    parse_sumo_time_seconds,
    parse_xml,
    triggered_vehicle_departure_times,
)


EXPECTED_CONVERSION_PROTOCOL = (
    "cfcmt-libsignal-full-external-sumo-conversion-v5"
)
EXPECTED_CONVERSION_PROTOCOLS = frozenset(
    {
        EXPECTED_CONVERSION_PROTOCOL,
        "cfcmt-cityflow-full-external-sumo-conversion-v6",
        "cfcmt-cityflow-full-external-sumo-conversion-v7",
        "cfcmt-cityflow-full-external-sumo-conversion-v8",
        "cfcmt-cityflow-full-external-sumo-conversion-v9",
        "cfcmt-cityflow-full-external-sumo-conversion-v10",
        "cfcmt-roadnetsz-native-sumo-window-package-v1",
        "cfcmt-roadnetsz-native-sumo122-window-package-v2",
        "cfcmt-roadnetsz-native-sumo122-window-package-v3",
        "cfcmt-roadnetsz-native-sumo122-window-package-v4",
        "cfcmt-dlr-bologna-native-sumo-package-v1",
        "cfcmt-figshare-xuancheng-native-sumo-package-v1",
        "cfcmt-lust-native-sumo-due-static-package-v1",
        "cfcmt-dublin-urban-native-sumo-package-v2",
        "cfcmt-dlr-badhersfeld-native-sumo-package-v1",
    }
)

ROADNETSZ_NATIVE_V1 = "cfcmt-roadnetsz-native-sumo-window-package-v1"
ROADNETSZ_NATIVE_V2 = "cfcmt-roadnetsz-native-sumo122-window-package-v2"
ROADNETSZ_NATIVE_V3 = "cfcmt-roadnetsz-native-sumo122-window-package-v3"
ROADNETSZ_NATIVE_V4 = "cfcmt-roadnetsz-native-sumo122-window-package-v4"
DLR_BOLOGNA_NATIVE_V1 = "cfcmt-dlr-bologna-native-sumo-package-v1"
FIGSHARE_XUANCHENG_NATIVE_V1 = (
    "cfcmt-figshare-xuancheng-native-sumo-package-v1"
)
LUST_NATIVE_DUE_STATIC_V1 = "cfcmt-lust-native-sumo-due-static-package-v1"
DUBLIN_URBAN_NATIVE_V2 = "cfcmt-dublin-urban-native-sumo-package-v2"
DLR_BADHERSFELD_NATIVE_V1 = (
    "cfcmt-dlr-badhersfeld-native-sumo-package-v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conversion_tree_sha256(conversion_root: Path) -> str:
    digest = hashlib.sha256()
    root = Path(conversion_root).resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"conversion root contains no files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_network(
    conversion_root: Path,
    scenario: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(conversion_root).resolve()
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") not in EXPECTED_CONVERSION_PROTOCOLS:
        raise ValueError(
            "external conversion protocol mismatch: "
            f"{manifest.get('protocol')!r}"
        )
    networks = dict(manifest.get("networks", {}))
    if scenario not in networks:
        raise ValueError(f"scenario is absent from conversion manifest: {scenario}")
    network = dict(networks[scenario])
    relative = Path(str(network.get("sumocfg", "")))
    if relative.is_absolute() or ".staging-" in relative.as_posix():
        raise ValueError(f"non-relocatable converted sumocfg path: {relative}")
    sumocfg = (root / relative).resolve()
    if root not in sumocfg.parents or not sumocfg.is_file():
        raise ValueError(f"converted sumocfg escapes or is missing: {sumocfg}")
    route_audit = dict(network.get("route_audit", {}))
    source_reroute_skeletons = int(
        route_audit.get("source_declared_dynamic_reroute_skeleton_count", 0)
    )
    allows_published_reroute_skeletons = (
        manifest.get("protocol") == DLR_BADHERSFELD_NATIVE_V1
        and source_reroute_skeletons == 29
        and int(route_audit.get("disconnected_route_count", -1))
        == source_reroute_skeletons
        and int(route_audit.get("unclassified_disconnected_route_count", -1))
        == 0
    )
    if (
        int(route_audit.get("route_count", 0)) <= 0
        or int(route_audit.get("missing_edge_route_count", -1)) != 0
        or (
            int(route_audit.get("disconnected_route_count", -1)) != 0
            and not allows_published_reroute_skeletons
        )
    ):
        raise ValueError(f"converted route audit failed: {route_audit}")
    conversion_protocol = str(manifest.get("protocol"))
    if conversion_protocol in {
        ROADNETSZ_NATIVE_V1,
        ROADNETSZ_NATIVE_V2,
        ROADNETSZ_NATIVE_V3,
        ROADNETSZ_NATIVE_V4,
    }:
        network_build = dict(network.get("network_build", {}))
        if conversion_protocol == ROADNETSZ_NATIVE_V1:
            valid_network_build = (
                network_build.get("protocol")
                == "roadnetsz-native-sumo-window-duarouter-v1"
                and network_build.get("mode")
                == "source_native_sumo_with_strict_explicit_routing"
                and int(
                    dict(network_build.get("duarouter", {})).get(
                        "returncode", -1
                    )
                )
                == 0
            )
        elif conversion_protocol == ROADNETSZ_NATIVE_V2:
            valid_network_build = (
                network_build.get("protocol")
                == "roadnetsz-source-components-sumo122-window-duarouter-v2"
                and network_build.get("mode")
                == (
                    "source_components_rebuilt_with_sumo122_and_"
                    "strict_explicit_routing"
                )
                and int(
                    dict(network_build.get("netconvert", {})).get(
                        "returncode", -1
                    )
                )
                == 0
                and int(
                    dict(network_build.get("duarouter", {})).get(
                        "returncode", -1
                    )
                )
                == 0
            )
        elif conversion_protocol == ROADNETSZ_NATIVE_V3:
            valid_network_build = (
                network_build.get("protocol")
                == (
                    "roadnetsz-source-components-sumo122-turnspeed55-"
                    "window-duarouter-v3"
                )
                and network_build.get("mode")
                == (
                    "source_components_rebuilt_with_sumo122_turnspeed55_"
                    "and_strict_explicit_routing"
                )
                and float(
                    network_build.get("junction_turn_speed_limit_mps", -1.0)
                )
                == 5.5
                and int(
                    dict(network_build.get("netconvert", {})).get(
                        "returncode", -1
                    )
                )
                == 0
                and int(
                    dict(network_build.get("duarouter", {})).get(
                        "returncode", -1
                    )
                )
                == 0
            )
        else:
            valid_network_build = (
                network_build.get("protocol")
                == (
                    "roadnetsz-source-components-sumo122-protected-incoming-"
                    "window-duarouter-v4"
                )
                and network_build.get("mode")
                == (
                    "source_components_rebuilt_with_sumo122_protected_"
                    "incoming_and_strict_explicit_routing"
                )
                and network_build.get("auto_generated_tls_phase_layout")
                == "incoming"
                and int(
                    dict(network_build.get("netconvert", {})).get(
                        "returncode", -1
                    )
                )
                == 0
                and int(
                    dict(network_build.get("duarouter", {})).get(
                        "returncode", -1
                    )
                )
                == 0
            )
        if not valid_network_build:
            raise ValueError(f"native SUMO network-build audit failed: {network_build}")
        source_window = dict(network.get("source_window", {}))
        if (
            float(source_window.get("routing_retention", -1.0))
            < float(source_window.get("minimum_required_retention", 1.0))
            or int(source_window.get("routed_vehicle_count", -1))
            != int(network.get("vehicle_count", -2))
        ):
            raise ValueError(f"native SUMO routing-retention audit failed: {source_window}")
    elif conversion_protocol == DLR_BOLOGNA_NATIVE_V1:
        network_build = dict(network.get("network_build", {}))
        demand_audit = dict(network.get("demand_audit", {}))
        if (
            network_build.get("protocol")
            != "dlr-bologna-native-read-only-package-v1"
            or network_build.get("mode")
            != "source_native_sumo_read_only_inputs"
            or network_build.get(
                "source_network_and_demand_files_preserved_byte_for_byte"
            )
            is not True
            or network_build.get("passive_output_definitions_removed") is not True
            or network_build.get("traffic_semantics_rebuilt_or_calibrated")
            is not False
            or demand_audit.get("protocol")
            != "dlr-bologna-deterministic-demand-inventory-v1"
            or demand_audit.get("all_source_demand_files_loaded") is not True
            or int(demand_audit.get("total", -1))
            != int(network.get("vehicle_count", -2))
            or int(network.get("person_count", -1))
            != int(demand_audit.get("person_total", 0))
        ):
            raise ValueError(
                "DLR Bologna native package audit failed: "
                f"build={network_build} demand={demand_audit}"
            )
    elif conversion_protocol == FIGSHARE_XUANCHENG_NATIVE_V1:
        network_build = dict(network.get("network_build", {}))
        demand_audit = dict(network.get("demand_audit", {}))
        if (
            network_build.get("protocol")
            != "figshare-xuancheng-native-sumo-read-only-v1"
            or network_build.get("mode")
            != "source_native_sumo_read_only_with_cityflow_anchor_completion"
            or network_build.get("source_network_preserved_byte_for_byte")
            is not True
            or network_build.get(
                "traffic_signal_programs_preserved_byte_for_byte"
            )
            is not True
            or network_build.get("traffic_semantics_rebuilt_or_calibrated")
            is not False
            or network_build.get("route_completion_protocol")
            != "native-sumo-length-dijkstra-cityflow-anchor-completion-v2"
            or demand_audit.get("protocol")
            != "figshare-xuancheng-cityflow-anchor-completion-v1"
            or demand_audit.get("all_source_flow_records_loaded") is not True
            or demand_audit.get("all_anchor_edges_preserved_in_order") is not True
            or demand_audit.get(
                "all_completed_route_pairs_native_sumo_connected"
            )
            is not True
            or int(demand_audit.get("total", -1))
            != int(network.get("vehicle_count", -2))
            or int(demand_audit.get("source_flow_record_count", -1)) <= 0
        ):
            raise ValueError(
                "Figshare Xuancheng native package audit failed: "
                f"build={network_build} demand={demand_audit}"
            )
    elif conversion_protocol == LUST_NATIVE_DUE_STATIC_V1:
        network_build = dict(network.get("network_build", {}))
        demand_audit = dict(network.get("demand_audit", {}))
        if (
            network_build.get("protocol")
            != "lust-native-sumo-read-only-due-static-v1"
            or network_build.get("mode")
            != "source_native_sumo_due_static_read_only_inputs"
            or network_build.get(
                "source_network_and_demand_files_preserved_byte_for_byte"
            )
            is not True
            or network_build.get(
                "passive_output_detector_and_polygon_files_removed"
            )
            is not True
            or network_build.get("traffic_semantics_rebuilt_or_calibrated")
            is not False
            or network_build.get("modern_sumo_real_data_validation_claimed")
            is not False
            or demand_audit.get("protocol")
            != "lust-due-static-complete-demand-inventory-v1"
            or demand_audit.get("all_source_demand_files_loaded") is not True
            or demand_audit.get(
                "source_network_and_demand_files_preserved_byte_for_byte"
            )
            is not True
            or int(demand_audit.get("total", -1))
            != int(network.get("vehicle_count", -2))
            or int(demand_audit.get("embedded_route_count", -1))
            != int(network.get("vehicle_count", -2))
            or int(demand_audit.get("duplicate_vehicle_id_count", -1)) != 0
        ):
            raise ValueError(
                "LuST native package audit failed: "
                f"build={network_build} demand={demand_audit}"
            )
    elif conversion_protocol == DUBLIN_URBAN_NATIVE_V2:
        network_build = dict(network.get("network_build", {}))
        demand_audit = dict(network.get("demand_audit", {}))
        if (
            network_build.get("protocol")
            != "dublin-urban-native-read-only-package-v2"
            or network_build.get("mode")
            != "source_native_sumo_full_day_read_only_inputs"
            or network_build.get(
                "source_network_route_emitter_tls_files_preserved_byte_for_byte"
            )
            is not True
            or network_build.get(
                "all_six_source_vehicle_composition_scenarios_packaged"
            )
            is not True
            or network_build.get(
                "passive_detector_and_poi_definition_removed"
            )
            is not True
            or network_build.get(
                "source_config_input_order_preserved_except_passive_detector"
            )
            is not True
            or tuple(network_build.get("packaged_additional_file_order", ()))
            != (
                "../common/DCC_trafficlights.add.xml",
                "vtypes.add.xml",
                "../common/DCC_routes.rou.xml",
                "../common/DCC_emitters.emi.xml",
            )
            or network_build.get("traffic_semantics_rebuilt_or_calibrated")
            is not False
            or float(network_build.get("source_horizon_sec", -1.0)) != 86_400.0
            or float(network_build.get("source_step_length_sec", -1.0)) != 0.5
            or demand_audit.get("protocol")
            != "dublin-urban-full-day-explicit-demand-v1"
            or demand_audit.get("all_source_demand_files_loaded") is not True
            or demand_audit.get(
                "source_network_route_emitter_tls_files_preserved_byte_for_byte"
            )
            is not True
            or int(demand_audit.get("total", -1))
            != int(network.get("vehicle_count", -2))
            or int(demand_audit.get("explicit_vehicle_count", -1))
            != int(network.get("vehicle_count", -2))
            or int(demand_audit.get("trip_count", -1)) != 0
            or int(demand_audit.get("flow_count", -1)) != 0
            or int(demand_audit.get("duplicate_vehicle_id_count", -1)) != 0
            or int(
                demand_audit.get("unknown_vehicle_route_reference_count", -1)
            )
            != 0
            or int(
                demand_audit.get(
                    "unknown_distribution_route_reference_count", -1
                )
            )
            != 0
        ):
            raise ValueError(
                "Dublin Urban native package audit failed: "
                f"build={network_build} demand={demand_audit}"
            )
    elif conversion_protocol == DLR_BADHERSFELD_NATIVE_V1:
        network_build = dict(network.get("network_build", {}))
        demand_audit = dict(network.get("demand_audit", {}))
        dynamic_routing_audit = dict(network.get("dynamic_routing_audit", {}))
        expected_additional_order = (
            "../osm/pt_vtypes.xml",
            "../osm/gtfs_publictransport.add.xml",
            "../osm/gtfs_publictransport.rou.xml",
            "../osm/defaults/basic.vType.xml",
            "../osm/osm_complete_parking_areas.add.xml",
            "../osm/osm_parking_rerouters.add.xml",
            "../osm/obstacle_seilerweg.add.xml",
            "../demand/osm_activitygen.lkw.rou.xml",
            "../osm/calib.add.xml",
        )
        if (
            network_build.get("protocol")
            != "dlr-badhersfeld-native-read-only-package-v1"
            or network_build.get("mode")
            != "source_native_sumo_full_day_read_only_inputs"
            or network_build.get(
                "source_network_and_demand_files_preserved_byte_for_byte"
            )
            is not True
            or network_build.get("complete_source_directory_preserved")
            is not True
            or network_build.get(
                "source_input_order_preserved_except_passive_polygon"
            )
            is not True
            or network_build.get(
                "passive_polygon_gui_and_output_destinations_removed"
            )
            is not True
            or network_build.get("published_calibrator_input_preserved")
            is not True
            or network_build.get("source_dynamic_route_repair_preserved")
            is not True
            or network_build.get("traffic_semantics_rebuilt_or_calibrated")
            is not False
            or float(network_build.get("source_horizon_sec", -1.0))
            != 86_400.0
            or float(network_build.get("source_step_length_sec", -1.0)) != 1.0
            or tuple(network.get("additional_files", ()))
            != expected_additional_order
            or tuple(network.get("dropped_passive_output_files", ()))
            != ("../osm/osm_polygons.add.xml",)
            or set(network.get("removed_passive_config_tags", ()))
            != {
                "gui_only",
                "device.taxi.dispatch-algorithm.output",
                "device.taxi.idle-algorithm.output",
            }
            or demand_audit.get("protocol")
            != "dlr-badhersfeld-complete-rich-demand-inventory-v1"
            or demand_audit.get("all_source_demand_files_loaded") is not True
            or demand_audit.get(
                "all_triggered_vehicles_resolved_to_person_plans"
            )
            is not True
            or demand_audit.get("published_calibrator_input_preserved")
            is not True
            or demand_audit.get(
                "source_network_and_demand_files_preserved_byte_for_byte"
            )
            is not True
            or int(demand_audit.get("total", -1))
            != int(network.get("vehicle_count", -2))
            or int(demand_audit.get("person_total", -1))
            != int(network.get("person_count", -2))
            or int(demand_audit.get("explicit_vehicle_count", -1))
            != int(network.get("vehicle_count", -2))
            or int(demand_audit.get("triggered_vehicle_count", 0)) <= 0
            or int(demand_audit.get("road_demand_id_count", -1))
            != int(network.get("vehicle_count", -2))
            or int(demand_audit.get("person_demand_id_count", -1))
            != int(network.get("person_count", -2))
            or int(demand_audit.get("duplicate_road_demand_id_count", -1))
            != 0
            or int(demand_audit.get("duplicate_person_demand_id_count", -1))
            != 0
            or int(demand_audit.get("nested_calibrator_flow_count", -1)) != 6
            or int(demand_audit.get("at_or_after_horizon", -1)) != 0
            or int(demand_audit.get("person_at_or_after_horizon", -1)) != 0
            or dynamic_routing_audit
            != {
                "ignore_route_errors": "true",
                "rerouting_probability": "1",
                "rerouting_period_sec": "300",
                "rerouting_pre_period_sec": "300",
            }
        ):
            raise ValueError(
                "DLR Bad Hersfeld native package audit failed: "
                f"build={network_build} demand={demand_audit}"
            )
    elif int(dict(network.get("netconvert", {})).get("returncode", -1)) != 0:
        raise ValueError("netconvert did not report a successful conversion")
    if int(network.get("vehicle_count", 0)) <= 0:
        raise ValueError("converted network has no exact positive vehicle_count")
    connection_audit = network.get("connection_audit")
    if connection_audit is not None and int(
        dict(connection_audit).get("missing_route_edge_pair_count", -1)
    ) != 0:
        raise ValueError(f"converted connection audit failed: {connection_audit}")
    if manifest.get("protocol") in {
        "cfcmt-cityflow-full-external-sumo-conversion-v7",
        "cfcmt-cityflow-full-external-sumo-conversion-v8",
        "cfcmt-cityflow-full-external-sumo-conversion-v9",
        "cfcmt-cityflow-full-external-sumo-conversion-v10",
    }:
        tls_audit = dict(network.get("tls_semantic_audit", {}))
        expected_tls_protocol = {
            "cfcmt-cityflow-full-external-sumo-conversion-v7": (
                "cityflow-roadlink-to-netconvert-linkindex-remap-"
                "permissive-intergreen-v2"
            ),
            "cfcmt-cityflow-full-external-sumo-conversion-v8": (
                "cityflow-roadlink-to-netconvert-linkindex-remap-"
                "permissive-intergreen-merge-yield-v3"
            ),
            "cfcmt-cityflow-full-external-sumo-conversion-v9": (
                "cityflow-roadlink-to-netconvert-linkindex-remap-"
                "permissive-intergreen-merge-yield-v3"
            ),
            "cfcmt-cityflow-full-external-sumo-conversion-v10": (
                "cityflow-roadlink-to-netconvert-linkindex-remap-"
                "permissive-intergreen-merge-conflict-yield-v4"
            ),
        }[str(manifest.get("protocol"))]
        if tls_audit.get("protocol") != expected_tls_protocol:
            raise ValueError(f"converted TLS semantic protocol failed: {tls_audit}")
        if (
            int(tls_audit.get("controlled_intersection_count", -1))
            != int(network.get("controlled_intersection_count", -2))
            or int(tls_audit.get("phase_count", 0)) <= 0
            or int(
                tls_audit.get(
                    "post_remap_phase_link_symmetric_difference",
                    -1,
                )
            )
            != 0
        ):
            raise ValueError(f"converted TLS semantic audit failed: {tls_audit}")
    if manifest.get("protocol") in {
        "cfcmt-cityflow-full-external-sumo-conversion-v9",
        "cfcmt-cityflow-full-external-sumo-conversion-v10",
    }:
        reduction = dict(network.get("virtual_lane_link_reduction", {}))
        if reduction.get("protocol") != (
            "cityflow-virtual-cartesian-lane-link-monotone-rank-v1"
        ):
            raise ValueError(
                f"converted virtual lane-link protocol failed: {reduction}"
            )
        source_pairs = int(reduction.get("source_unique_lane_pair_count", -1))
        retained_pairs = int(
            reduction.get("retained_unique_lane_pair_count", -1)
        )
        removed_pairs = int(
            reduction.get("removed_unique_lane_pair_count", -1)
        )
        if (
            source_pairs < 0
            or retained_pairs < 0
            or removed_pairs < 0
            or source_pairs != retained_pairs + removed_pairs
            or int(
                reduction.get(
                    "post_reduction_crossing_road_link_count",
                    -1,
                )
            )
            != 0
            or int(reduction.get("source_lane_coverage_failures", -1)) != 0
            or int(reduction.get("target_lane_coverage_failures", -1)) != 0
        ):
            raise ValueError(
                f"converted virtual lane-link audit failed: {reduction}"
            )
    if manifest.get("protocol") == (
        "cfcmt-cityflow-full-external-sumo-conversion-v10"
    ) and int(
        dict(network.get("tls_semantic_audit", {})).get(
            "conflict_yield_phase_link_count", 0
        )
    ) <= 0:
        raise ValueError(
            "v10 conversion did not apply any conflict-priority yield links"
        )
    if conversion_protocol == ROADNETSZ_NATIVE_V1:
        tls_audit = dict(network.get("tls_semantic_audit", {}))
        if (
            tls_audit.get("protocol")
            != "roadnetsz-native-sumo-tls-preserved-v1"
            or int(tls_audit.get("controlled_intersection_count", -1))
            != int(network.get("controlled_intersection_count", -2))
            or int(tls_audit.get("phase_count", 0)) <= 0
            or tls_audit.get("source_tls_programs_preserved_byte_for_byte")
            is not True
            or tls_audit.get("source_net_sha256")
            != tls_audit.get("packaged_net_sha256")
        ):
            raise ValueError(f"native SUMO TLS identity audit failed: {tls_audit}")
    if conversion_protocol in {
        ROADNETSZ_NATIVE_V2,
        ROADNETSZ_NATIVE_V3,
        ROADNETSZ_NATIVE_V4,
    }:
        tls_audit = dict(network.get("tls_semantic_audit", {}))
        declared_count = int(tls_audit.get("declared_tls_node_count", -1))
        explicit_id_count = int(
            tls_audit.get("explicit_source_tls_id_count", -1)
        )
        rebuilt_count = int(tls_audit.get("rebuilt_tls_id_count", -1))
        generated_count = int(tls_audit.get("auto_generated_tls_id_count", -1))
        if (
            tls_audit.get("protocol")
            != "roadnetsz-source-tll-sumo122-rebuild-v2"
            or declared_count <= 0
            or explicit_id_count <= 0
            or rebuilt_count != declared_count
            or rebuilt_count != int(network.get("controlled_intersection_count", -2))
            or generated_count != rebuilt_count - explicit_id_count
            or int(tls_audit.get("rebuilt_phase_count", 0)) <= 0
            or int(tls_audit.get("missing_declared_tls_id_count", -1)) != 0
            or int(tls_audit.get("unexpected_rebuilt_tls_id_count", -1)) != 0
            or int(tls_audit.get("missing_explicit_tls_program_count", -1))
            != 0
            or int(tls_audit.get("explicit_phase_count_mismatch_count", -1))
            != 0
            or tls_audit.get("all_declared_tls_nodes_rebuilt") is not True
            or tls_audit.get(
                "explicit_source_tls_program_phase_counts_preserved"
            )
            is not True
            or (
                conversion_protocol == ROADNETSZ_NATIVE_V4
                and (
                    tls_audit.get("auto_generated_phase_layout") != "incoming"
                    or int(
                        tls_audit.get(
                            "auto_generated_minor_green_signal_count", -1
                        )
                    )
                    != 0
                )
            )
        ):
            raise ValueError(
                f"native SUMO TLS reconstruction audit failed: {tls_audit}"
            )
    if conversion_protocol == DLR_BOLOGNA_NATIVE_V1:
        tls_audit = dict(network.get("tls_semantic_audit", {}))
        if (
            tls_audit.get("protocol")
            != "dlr-bologna-native-sumo-tls-preserved-v1"
            or int(tls_audit.get("controlled_intersection_count", -1))
            != int(network.get("controlled_intersection_count", -2))
            or int(tls_audit.get("program_count", 0)) <= 0
            or int(tls_audit.get("phase_count", 0)) <= 0
            or tls_audit.get(
                "source_net_and_tls_files_preserved_byte_for_byte"
            )
            is not True
        ):
            raise ValueError(
                f"DLR Bologna native TLS identity audit failed: {tls_audit}"
            )
    if conversion_protocol == FIGSHARE_XUANCHENG_NATIVE_V1:
        tls_audit = dict(network.get("tls_semantic_audit", {}))
        if (
            tls_audit.get("protocol")
            != "figshare-xuancheng-native-sumo-tls-preserved-v1"
            or int(tls_audit.get("controlled_intersection_count", -1))
            != int(network.get("controlled_intersection_count", -2))
            or int(tls_audit.get("program_count", 0)) <= 0
            or int(tls_audit.get("phase_count", 0)) <= 0
            or tls_audit.get(
                "source_native_sumo_tls_preserved_byte_for_byte"
            )
            is not True
        ):
            raise ValueError(
                f"Figshare Xuancheng native TLS identity audit failed: {tls_audit}"
            )
    if conversion_protocol == LUST_NATIVE_DUE_STATIC_V1:
        tls_audit = dict(network.get("tls_semantic_audit", {}))
        if (
            tls_audit.get("protocol") != "lust-native-static-tls-preserved-v1"
            or int(tls_audit.get("controlled_intersection_count", -1))
            != int(network.get("controlled_intersection_count", -2))
            or int(tls_audit.get("program_count", 0)) <= 0
            or int(tls_audit.get("phase_count", 0)) <= 0
            or tls_audit.get(
                "source_native_net_and_static_tls_preserved_byte_for_byte"
            )
            is not True
        ):
            raise ValueError(f"LuST native TLS identity audit failed: {tls_audit}")
    if conversion_protocol == DUBLIN_URBAN_NATIVE_V2:
        tls_audit = dict(network.get("tls_semantic_audit", {}))
        if (
            tls_audit.get("protocol")
            != "dublin-urban-native-sumo-tls-preserved-v1"
            or int(tls_audit.get("controlled_intersection_count", -1))
            != int(network.get("controlled_intersection_count", -2))
            or int(tls_audit.get("program_count", 0)) <= 0
            or int(tls_audit.get("phase_count", 0)) <= 0
            or tls_audit.get(
                "source_native_net_and_tls_preserved_byte_for_byte"
            )
            is not True
        ):
            raise ValueError(
                f"Dublin Urban native TLS identity audit failed: {tls_audit}"
            )
    if conversion_protocol == DLR_BADHERSFELD_NATIVE_V1:
        tls_audit = dict(network.get("tls_semantic_audit", {}))
        if (
            tls_audit.get("protocol")
            != "dlr-badhersfeld-native-sumo-tls-preserved-v1"
            or int(tls_audit.get("controlled_intersection_count", -1))
            != int(network.get("controlled_intersection_count", -2))
            or int(tls_audit.get("program_count", 0)) <= 0
            or int(tls_audit.get("phase_count", 0)) <= 0
            or tls_audit.get(
                "source_native_net_and_tls_preserved_byte_for_byte"
            )
            is not True
        ):
            raise ValueError(
                f"DLR Bad Hersfeld native TLS identity audit failed: {tls_audit}"
            )
    return sumocfg, manifest, network


def _start_admission_sumo(
    sumo_api: Any,
    sumocfg: Path,
    seed: int,
    *,
    summary_output: Path,
) -> None:
    try:
        sumo_api.close()
    except Exception:
        pass
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.unlink(missing_ok=True)
    arguments = _sumo_arguments(sumocfg, seed, include_executable=True)
    arguments.extend(
        [
            "--summary-output",
            str(summary_output),
            "--summary-output.period",
            "1",
        ]
    )
    sumo_api.start(arguments)


def _final_summary_step(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"SUMO summary output is missing: {path}")
    final: dict[str, Any] | None = None
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "step":
            final = dict(element.attrib)
        element.clear()
    if final is None:
        raise ValueError(f"SUMO summary output contains no step: {path}")
    integer_fields = (
        "loaded",
        "inserted",
        "running",
        "waiting",
        "ended",
        "arrived",
        "collisions",
        "teleports",
    )
    parsed: dict[str, Any] = {"time": float(final.get("time", 0.0))}
    for field in integer_fields:
        parsed[field] = int(float(final.get(field, 0)))
    return parsed


def _explicit_vehicle_inventory(
    sumocfg: Path,
    *,
    horizon_sec: float,
) -> dict[str, Any]:
    config_root = parse_xml(sumocfg).getroot()
    route_paths = []
    for tag in ("route-files", "additional-files"):
        for item in config_root.findall(f".//{tag}"):
            for value in str(item.attrib.get("value", "")).split(","):
                if value.strip():
                    route_paths.append((sumocfg.parent / value.strip()).resolve())
    route_paths = list(dict.fromkeys(route_paths))
    if not route_paths:
        raise ValueError(f"SUMO config has no route or additional files: {sumocfg}")
    total = 0
    before_horizon = 0
    at_or_after_horizon = 0
    explicit_vehicle_count = 0
    flow_vehicle_count = 0
    person_total = 0
    person_before_horizon = 0
    person_at_or_after_horizon = 0
    deterministic_flow_count = 0
    triggered_vehicle_count = 0
    encountered_rich_demand = False
    minimum_departure_sec: float | None = None
    maximum_departure_sec: float | None = None

    def add_road_departure(depart: float) -> None:
        nonlocal total, before_horizon, at_or_after_horizon
        nonlocal minimum_departure_sec, maximum_departure_sec
        total += 1
        minimum_departure_sec = (
            depart
            if minimum_departure_sec is None
            else min(minimum_departure_sec, depart)
        )
        maximum_departure_sec = (
            depart
            if maximum_departure_sec is None
            else max(maximum_departure_sec, depart)
        )
        if depart < float(horizon_sec) - 1e-9:
            before_horizon += 1
        else:
            at_or_after_horizon += 1

    def deterministic_departures(element: ET.Element) -> tuple[float, ...]:
        begin = parse_sumo_time_seconds(element.attrib.get("begin", 0.0))
        end = parse_sumo_time_seconds(element.attrib.get("end", horizon_sec))
        if end < begin:
            raise ValueError(f"flow end precedes begin in {route_path}")
        if "period" in element.attrib:
            period = parse_sumo_time_seconds(element.attrib["period"])
            if period <= 0.0:
                raise ValueError(f"flow period must be positive in {route_path}")
            count = max(0, int(math.ceil((end - begin) / period - 1e-12)))
            return tuple(begin + index * period for index in range(count))
        if "number" in element.attrib:
            count = int(element.attrib["number"])
            if count < 0:
                raise ValueError(f"flow number must be nonnegative in {route_path}")
            if count == 0:
                return ()
            if count == 1:
                return (begin,)
            interval = (end - begin) / count
            return tuple(begin + index * interval for index in range(count))
        raise ValueError(
            "external admission requires deterministic flow demand with "
            f"period or number in {route_path}"
        )

    triggered_departures = triggered_vehicle_departure_times(route_paths)
    for route_path in route_paths:
        ancestors: list[str] = []
        for event, element in iterparse_xml(
            route_path, events=("start", "end")
        ):
            if event == "start":
                parent = ancestors[-1] if ancestors else None
                ancestors.append(str(element.tag))
                continue
            parent = ancestors[-2] if len(ancestors) >= 2 else None
            if element.tag in {"vehicle", "trip"} and parent in DEMAND_ROOT_TAGS:
                depart_text = element.attrib.get("depart")
                if depart_text is None:
                    raise ValueError(
                        f"explicit demand has no depart time in {route_path}"
                    )
                if depart_text == "triggered":
                    identity = str(element.attrib.get("id", ""))
                    if identity not in triggered_departures:
                        raise ValueError(
                            "triggered SUMO vehicle has no person-plan "
                            f"reference in {route_path}: {identity}"
                        )
                    depart = float(triggered_departures[identity])
                    triggered_vehicle_count += 1
                    encountered_rich_demand = True
                else:
                    depart = parse_sumo_time_seconds(depart_text)
                explicit_vehicle_count += 1
                add_road_departure(depart)
            elif element.tag == "person" and parent in DEMAND_ROOT_TAGS:
                encountered_rich_demand = True
                depart_text = element.attrib.get("depart")
                if depart_text is None:
                    raise ValueError(
                        f"explicit person has no depart time in {route_path}"
                    )
                depart = parse_sumo_time_seconds(depart_text)
                person_total += 1
                if depart < float(horizon_sec) - 1e-9:
                    person_before_horizon += 1
                else:
                    person_at_or_after_horizon += 1
            elif (
                element.tag in {"flow", "personFlow"}
                and parent in DEMAND_ROOT_TAGS
            ):
                encountered_rich_demand = True
                departures = deterministic_departures(element)
                deterministic_flow_count += 1
                if element.tag == "flow":
                    flow_vehicle_count += len(departures)
                    for depart in departures:
                        add_road_departure(depart)
                else:
                    person_total += len(departures)
                    person_before_horizon += sum(
                        depart < float(horizon_sec) - 1e-9
                        for depart in departures
                    )
                    person_at_or_after_horizon += sum(
                        depart >= float(horizon_sec) - 1e-9
                        for depart in departures
                    )
            element.clear()
            if not ancestors or ancestors[-1] != element.tag:
                raise ValueError(f"invalid SUMO XML nesting in {route_path}")
            ancestors.pop()
    result = {
        "total": total,
        "before_horizon": before_horizon,
        "at_or_after_horizon": at_or_after_horizon,
        "minimum_departure_sec": minimum_departure_sec,
        "maximum_departure_sec": maximum_departure_sec,
    }
    if encountered_rich_demand:
        result.update(
            {
                "demand_protocol": "deterministic-vehicle-flow-person-v1",
                "explicit_vehicle_count": explicit_vehicle_count,
                "flow_vehicle_count": flow_vehicle_count,
                "deterministic_flow_count": deterministic_flow_count,
                "triggered_vehicle_count": triggered_vehicle_count,
                "person_total": person_total,
                "person_before_horizon": person_before_horizon,
                "person_at_or_after_horizon": person_at_or_after_horizon,
            }
        )
    return result


def _explicit_demand_loading_checks(
    inventory: dict[str, Any],
    *,
    summary_loaded: int,
    horizon_sec: float,
) -> dict[str, bool]:
    """Validate SUMO loading without treating parser look-ahead as departures."""

    total = int(inventory["total"])
    due_before_horizon = int(inventory["before_horizon"])
    maximum_departure = inventory.get("maximum_departure_sec")
    full_demand_horizon = (
        maximum_departure is not None
        and float(maximum_departure) <= float(horizon_sec) + 1e-9
    )
    return {
        "demand_loading_consistent": (
            due_before_horizon <= int(summary_loaded) <= total
        ),
        "demand_conservation": (
            not full_demand_horizon or int(summary_loaded) == total
        ),
    }


def _collision_sample(collision: Any, *, time_sec: float) -> dict[str, Any]:
    sample = {"repr": repr(collision), "time_sec": float(time_sec)}
    for field in ("collider", "victim", "type", "lane", "pos"):
        value = getattr(collision, field, None)
        if value is not None:
            sample[field] = value
    return sample


def run_external_network_admission(
    *,
    conversion_root: Path,
    scenario: str,
    seed: int,
    horizon_sec: float,
    expected_sumo_version: str,
    minimum_controllable_tls: int,
    maximum_collision_count: int,
    maximum_teleport_fraction: float,
    expected_conversion_manifest_sha256: str | None = None,
    expected_conversion_tree_sha256: str | None = None,
    progress_interval_sec: int = 300,
) -> dict[str, Any]:
    started = time.perf_counter()
    sumocfg, manifest, network = _resolve_network(conversion_root, scenario)
    conversion_manifest_sha256 = _sha256(
        Path(conversion_root).resolve() / "conversion_manifest.json"
    )
    conversion_tree_digest = conversion_tree_sha256(conversion_root)
    if (
        expected_conversion_manifest_sha256 is not None
        and conversion_manifest_sha256
        != str(expected_conversion_manifest_sha256)
    ):
        raise ValueError(
            "external conversion manifest SHA-256 mismatch: "
            f"{conversion_manifest_sha256} != "
            f"{expected_conversion_manifest_sha256}"
        )
    if (
        expected_conversion_tree_sha256 is not None
        and conversion_tree_digest != str(expected_conversion_tree_sha256)
    ):
        raise ValueError(
            "external conversion tree SHA-256 mismatch: "
            f"{conversion_tree_digest} != {expected_conversion_tree_sha256}"
        )
    vehicle_inventory = _explicit_vehicle_inventory(
        sumocfg,
        horizon_sec=float(horizon_sec),
    )
    if vehicle_inventory["total"] != int(network["vehicle_count"]):
        raise ValueError(
            "explicit route demand differs from conversion manifest: "
            f"{vehicle_inventory['total']} != {int(network['vehicle_count'])}"
        )
    actual_sumo_version = libsumo_version()
    if actual_sumo_version != str(expected_sumo_version):
        raise RuntimeError(
            "external admission SUMO version mismatch: "
            f"expected {expected_sumo_version!r}, found {actual_sumo_version!r}"
        )
    sumo_api = load_libsumo()
    route_load_success = False
    initial_min_expected = 0
    tls_ids: tuple[str, ...] = ()
    final_time = 0.0
    loaded = 0
    departed = 0
    arrived = 0
    starting_teleports = 0
    ending_teleports = 0
    raw_collision_events = 0
    collision_incidents: set[tuple[str, str, str, str]] = set()
    collision_samples: list[dict[str, Any]] = []
    first_collision_time: float | None = None
    last_collision_time: float | None = None
    max_active_vehicles = 0
    max_min_expected = 0
    final_active_vehicles = 0
    final_pending_vehicles = 0
    final_min_expected = 0
    summary: dict[str, Any] | None = None
    error: str | None = None
    termination_reason: str | None = None
    next_progress = float(progress_interval_sec)
    with sumo_state_directory(prefix=f"cfcmt-admission-{scenario}-{seed}-") as scratch:
        summary_path = scratch / "summary.xml"
        try:
            _start_admission_sumo(
                sumo_api,
                sumocfg,
                int(seed),
                summary_output=summary_path,
            )
            route_load_success = True
            initial_min_expected = int(sumo_api.simulation.getMinExpectedNumber())
            tls_ids = tuple(str(value) for value in sumo_api.trafficlight.getIDList())
            while float(sumo_api.simulation.getTime()) + 1e-9 < float(horizon_sec):
                sumo_api.simulationStep()
                final_time = float(sumo_api.simulation.getTime())
                loaded += int(sumo_api.simulation.getLoadedNumber())
                departed += int(sumo_api.simulation.getDepartedNumber())
                arrived += int(sumo_api.simulation.getArrivedNumber())
                starting_teleports += int(
                    sumo_api.simulation.getStartingTeleportNumber()
                )
                ending_teleports += int(sumo_api.simulation.getEndingTeleportNumber())
                collisions = tuple(sumo_api.simulation.getCollisions())
                raw_collision_events += len(collisions)
                for collision in collisions:
                    incident = _collision_incident_key(collision)
                    if incident not in collision_incidents and len(collision_samples) < 12:
                        collision_samples.append(
                            _collision_sample(collision, time_sec=final_time)
                        )
                    collision_incidents.add(incident)
                    first_collision_time = (
                        final_time
                        if first_collision_time is None
                        else min(first_collision_time, final_time)
                    )
                    last_collision_time = (
                        final_time
                        if last_collision_time is None
                        else max(last_collision_time, final_time)
                    )
                if len(collision_incidents) > int(maximum_collision_count):
                    termination_reason = (
                        "unique_collision_limit_irreversibly_exceeded"
                    )
                    break
                maximum_possible_departures = int(
                    vehicle_inventory["before_horizon"]
                )
                if starting_teleports > (
                    float(maximum_teleport_fraction)
                    * float(maximum_possible_departures)
                    + 1e-12
                ):
                    termination_reason = (
                        "teleport_fraction_limit_irreversibly_exceeded"
                    )
                    break
                active = int(sumo_api.vehicle.getIDCount())
                minimum_expected = int(sumo_api.simulation.getMinExpectedNumber())
                max_active_vehicles = max(max_active_vehicles, active)
                max_min_expected = max(max_min_expected, minimum_expected)
                if final_time + 1e-9 >= next_progress:
                    print(
                        f"ADMISSION_PROGRESS scenario={scenario} seed={seed} "
                        f"time={final_time:.0f}/{float(horizon_sec):.0f} "
                        f"active={active} expected={minimum_expected}",
                        flush=True,
                    )
                    next_progress += float(progress_interval_sec)
            final_active_vehicles = int(sumo_api.vehicle.getIDCount())
            final_pending_vehicles = len(sumo_api.simulation.getPendingVehicles())
            final_min_expected = int(sumo_api.simulation.getMinExpectedNumber())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                sumo_api.close()
            except Exception:
                pass
        try:
            summary = _final_summary_step(summary_path)
        except Exception as exc:
            if error is None:
                error = f"{type(exc).__name__}: {exc}"

    teleport_fraction = float(starting_teleports / max(departed, 1))
    summary_teleport_fraction = float(
        int((summary or {}).get("teleports", 0))
        / max(int((summary or {}).get("inserted", 0)), 1)
    )
    unique_collision_count = len(collision_incidents)
    expected_vehicle_count = int(vehicle_inventory["before_horizon"])
    collision_lane_counts = Counter(incident[3] for incident in collision_incidents)
    collision_junction_counts: Counter[str] = Counter()
    for lane, count in collision_lane_counts.items():
        if lane.startswith(":"):
            junction = lane.split("_", 1)[0].removeprefix(":")
            collision_junction_counts[junction] += int(count)
    summary_time = float((summary or {}).get("time", 0.0))
    summary_loaded = int((summary or {}).get("loaded", -1))
    demand_loading_checks = _explicit_demand_loading_checks(
        vehicle_inventory,
        summary_loaded=summary_loaded,
        horizon_sec=float(horizon_sec),
    )
    checks = {
        "route_load_success": route_load_success,
        "summary_output_present": summary is not None,
        "summary_horizon_reached": summary_time + 1.0 + 1e-9
        >= float(horizon_sec),
        **demand_loading_checks,
        "positive_loaded_demand": (
            loaded > 0
            or int((summary or {}).get("loaded", 0)) > 0
            or initial_min_expected > 0
        ),
        "horizon_reached": final_time + 1e-9 >= float(horizon_sec),
        "minimum_controllable_tls": len(tls_ids) >= int(minimum_controllable_tls),
        "collision_limit": unique_collision_count <= int(maximum_collision_count),
        "summary_collision_limit": int((summary or {}).get("collisions", 0))
        <= int(maximum_collision_count),
        "teleport_fraction_limit": teleport_fraction
        <= float(maximum_teleport_fraction),
        "summary_teleport_fraction_limit": summary_teleport_fraction
        <= float(maximum_teleport_fraction),
        "runtime_exception_free": error is None,
    }
    return {
        "experiment": "traffic_signal_external_network_admission",
        "protocol": (
            "fail-fast-irreversible-safety-libsumo-demand-conserving-"
            "admission-v5"
        ),
        "scenario": str(scenario),
        "seed": int(seed),
        "passed": all(checks.values()),
        "checks": checks,
        "error": error,
        "runtime": runtime_metadata(),
        "setting": {
            "sumocfg": str(sumocfg),
            "horizon_sec": float(horizon_sec),
            "minimum_controllable_tls": int(minimum_controllable_tls),
            "maximum_collision_count": int(maximum_collision_count),
            "maximum_teleport_fraction": float(maximum_teleport_fraction),
            "backend": "libsumo",
            "expected_vehicle_count": expected_vehicle_count,
            "full_horizon_demand_conservation_required": (
                vehicle_inventory.get("maximum_departure_sec") is not None
                and float(vehicle_inventory["maximum_departure_sec"])
                <= float(horizon_sec) + 1e-9
            ),
            "explicit_vehicle_inventory": vehicle_inventory,
        },
        "conversion": {
            "root": str(Path(conversion_root).resolve()),
            "manifest_sha256": conversion_manifest_sha256,
            "tree_sha256": conversion_tree_digest,
            "protocol": manifest["protocol"],
            "network": network,
            "scenario_input_fingerprint": _scenario_input_fingerprint(sumocfg),
        },
        "observed": {
            "initial_min_expected": initial_min_expected,
            "traci_step_loaded_total": loaded,
            "departed": departed,
            "arrived": arrived,
            "max_active_vehicles": max_active_vehicles,
            "max_min_expected": max_min_expected,
            "final_active_vehicles": final_active_vehicles,
            "final_pending_vehicles": final_pending_vehicles,
            "final_min_expected": final_min_expected,
            "controllable_tls_count": len(tls_ids),
            "starting_teleports": starting_teleports,
            "ending_teleports": ending_teleports,
            "teleport_fraction": teleport_fraction,
            "summary_teleport_fraction": summary_teleport_fraction,
            "summary": summary,
            "raw_collision_events": raw_collision_events,
            "unique_collision_incidents": unique_collision_count,
            "collision_samples": collision_samples,
            "first_collision_time_sec": first_collision_time,
            "last_collision_time_sec": last_collision_time,
            "unique_collision_incidents_by_lane": dict(
                sorted(collision_lane_counts.items())
            ),
            "unique_collision_incidents_by_junction": dict(
                sorted(collision_junction_counts.items())
            ),
            "final_time_sec": final_time,
            "terminated_early": termination_reason is not None,
            "termination_reason": termination_reason,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--horizon-sec", type=float, default=3600.0)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument("--minimum-controllable-tls", type=int, default=1)
    parser.add_argument("--maximum-collision-count", type=int, default=0)
    parser.add_argument("--maximum-teleport-fraction", type=float, default=0.01)
    parser.add_argument("--expected-conversion-manifest-sha256")
    parser.add_argument("--expected-conversion-tree-sha256")
    parser.add_argument("--progress-interval-sec", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_external_network_admission(
        conversion_root=args.conversion_root,
        scenario=args.scenario,
        seed=args.seed,
        horizon_sec=args.horizon_sec,
        expected_sumo_version=args.expected_sumo_version,
        minimum_controllable_tls=args.minimum_controllable_tls,
        maximum_collision_count=args.maximum_collision_count,
        maximum_teleport_fraction=args.maximum_teleport_fraction,
        expected_conversion_manifest_sha256=(
            args.expected_conversion_manifest_sha256
        ),
        expected_conversion_tree_sha256=args.expected_conversion_tree_sha256,
        progress_interval_sec=args.progress_interval_sec,
    )
    atomic_write_json(args.out, result)
    print(
        f"network admission {'PASS' if result['passed'] else 'REJECT'}: "
        f"{args.scenario} seed={args.seed}",
        flush=True,
    )
    print(f"Results saved to: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
