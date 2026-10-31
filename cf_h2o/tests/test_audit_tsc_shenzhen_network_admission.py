from __future__ import annotations

import json
from pathlib import Path

from scripts.cluster.audit_tsc_shenzhen_network_admission import (
    CITYFLOW_TLS_PROTOCOL,
    NATIVE_TLS_PROTOCOL,
    NATIVE_TLS_REBUILD_PROTOCOL,
    RESULT_PROTOCOL,
    VIRTUAL_LANE_PROTOCOL,
    audit_matrix,
)


def test_admission_audit_keeps_failed_scenario_as_rejection_evidence(
    tmp_path: Path,
) -> None:
    scenarios = ["low", "medium", "high"]
    seeds = [1, 2]
    protocol = {
        "protocol": "tsc-v94-roadnetsz-shenzhen-pre-efficacy-network-admission-v1",
        "scenarios": scenarios,
        "seeds": seeds,
        "conversion": {
            "protocol": "cfcmt-cityflow-full-external-sumo-conversion-v10",
            "manifest_sha256": "manifest",
            "tree_sha256": "tree",
        },
        "runtime": {
            "sumo_version": "1.22.0",
            "horizon_sec": 3600,
            "minimum_controllable_tls": 35,
        },
        "admission": {"minimum_eligible_scenarios": 2},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    results = tmp_path / "results"
    for scenario in scenarios:
        for seed in seeds:
            passed = not (scenario == "high" and seed == 2)
            path = results / f"{scenario}_seed{seed}" / "admission.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "protocol": RESULT_PROTOCOL,
                        "scenario": scenario,
                        "seed": seed,
                        "passed": passed,
                        "checks": {
                            "collision_limit": passed,
                            "runtime_exception_free": True,
                        },
                        "runtime": {
                            "libsumo_version": "1.22.0",
                            "hostname": "node001",
                        },
                        "conversion": {
                            "protocol": protocol["conversion"]["protocol"],
                            "manifest_sha256": "manifest",
                            "tree_sha256": "tree",
                            "network": {
                                "tls_semantic_audit": {
                                    "protocol": CITYFLOW_TLS_PROTOCOL,
                                    "conflict_yield_phase_link_count": 1,
                                },
                                "virtual_lane_link_reduction": {
                                    "protocol": VIRTUAL_LANE_PROTOCOL,
                                    "post_reduction_crossing_road_link_count": 0,
                                    "source_lane_coverage_failures": 0,
                                    "target_lane_coverage_failures": 0,
                                },
                            },
                        },
                        "observed": {
                            "final_time_sec": 3600,
                            "controllable_tls_count": 35,
                            "unique_collision_incidents": 0 if passed else 1,
                            "starting_teleports": 0,
                            "departed": 10,
                            "arrived": 8,
                        },
                        "elapsed_seconds": 1.0,
                    }
                ),
                encoding="utf-8",
            )

    audit = audit_matrix(protocol_path=protocol_path, results_root=results)

    assert audit["status"] == "PASS"
    assert audit["eligible_scenarios"] == ["low", "medium"]
    assert audit["rejected_scenarios"] == ["high"]
    assert audit["scenario_decisions"]["high"]["failed_seeds"] == [2]


def test_admission_audit_accepts_native_pcl_identity(tmp_path: Path) -> None:
    protocol = {
        "protocol": (
            "tsc-v94-roadnetsz-shenzhen-pcl-pre-efficacy-network-admission-v1"
        ),
        "scenarios": ["pcl"],
        "seeds": [1],
        "conversion": {
            "protocol": "cfcmt-roadnetsz-native-sumo-window-package-v1",
            "manifest_sha256": "manifest",
            "tree_sha256": "tree",
        },
        "runtime": {
            "sumo_version": "1.22.0",
            "horizon_sec": 3600,
            "minimum_controllable_tls": 36,
        },
        "admission": {
            "minimum_eligible_scenarios": 1,
            "minimum_routing_retention": 0.9,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    result_path = tmp_path / "results/pcl_seed1/admission.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "protocol": RESULT_PROTOCOL,
                "scenario": "pcl",
                "seed": 1,
                "passed": True,
                "checks": {"runtime_exception_free": True},
                "runtime": {
                    "libsumo_version": "1.22.0",
                    "hostname": "node001",
                },
                "conversion": {
                    "protocol": protocol["conversion"]["protocol"],
                    "manifest_sha256": "manifest",
                    "tree_sha256": "tree",
                    "network": {
                        "source_window": {"routing_retention": 0.92},
                        "network_build": {
                            "protocol": (
                                "roadnetsz-native-sumo-window-duarouter-v1"
                            ),
                            "duarouter": {"returncode": 0},
                        },
                        "tls_semantic_audit": {
                            "protocol": NATIVE_TLS_PROTOCOL,
                            "source_tls_programs_preserved_byte_for_byte": True,
                            "source_net_sha256": "same",
                            "packaged_net_sha256": "same",
                        },
                    },
                },
                "observed": {
                    "final_time_sec": 3600,
                    "controllable_tls_count": 36,
                    "unique_collision_incidents": 0,
                    "starting_teleports": 0,
                    "departed": 10,
                    "arrived": 9,
                },
                "elapsed_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )

    audit = audit_matrix(
        protocol_path=protocol_path,
        results_root=tmp_path / "results",
    )

    assert audit["status"] == "PASS"

    assert audit["eligible_scenarios"] == ["pcl"]

def test_admission_audit_accepts_sumo122_rebuilt_pcl_identity(
    tmp_path: Path,
) -> None:
    protocol = {
        "protocol": (
            "tsc-v94-roadnetsz-shenzhen-pcl-pre-efficacy-network-admission-v1"
        ),
        "scenarios": ["pcl"],
        "seeds": [1],
        "conversion": {
            "protocol": "cfcmt-roadnetsz-native-sumo122-window-package-v2",
            "manifest_sha256": "manifest",
            "tree_sha256": "tree",
        },
        "runtime": {
            "sumo_version": "1.22.0",
            "horizon_sec": 3600,
            "minimum_controllable_tls": 140,
        },
        "admission": {
            "minimum_eligible_scenarios": 1,
            "minimum_routing_retention": 0.9,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    result_path = tmp_path / "results/pcl_seed1/admission.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "protocol": RESULT_PROTOCOL,
                "scenario": "pcl",
                "seed": 1,
                "passed": True,
                "checks": {"runtime_exception_free": True},
                "runtime": {
                    "libsumo_version": "1.22.0",
                    "hostname": "node001",
                },
                "conversion": {
                    "protocol": protocol["conversion"]["protocol"],
                    "manifest_sha256": "manifest",
                    "tree_sha256": "tree",
                    "network": {
                        "controlled_intersection_count": 140,
                        "source_window": {"routing_retention": 0.92},
                        "network_build": {
                            "protocol": (
                                "roadnetsz-source-components-sumo122-window-"
                                "duarouter-v2"
                            ),
                            "netconvert": {"returncode": 0},
                            "duarouter": {"returncode": 0},
                        },
                        "tls_semantic_audit": {
                            "protocol": NATIVE_TLS_REBUILD_PROTOCOL,
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
                        },
                    },
                },
                "observed": {
                    "final_time_sec": 3600,
                    "controllable_tls_count": 140,
                    "unique_collision_incidents": 0,
                    "starting_teleports": 0,
                    "departed": 10,
                    "arrived": 9,
                },
                "elapsed_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )

    audit = audit_matrix(
        protocol_path=protocol_path,
        results_root=tmp_path / "results",
    )

    assert audit["status"] == "PASS"

    assert audit["eligible_scenarios"] == ["pcl"]

    protocol["protocol"] = (
        "tsc-v94-roadnetsz-shenzhen-pcl-pre-efficacy-network-admission-v2"
    )
    protocol["conversion"]["protocol"] = (
        "cfcmt-roadnetsz-native-sumo122-window-package-v3"
    )
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["conversion"]["protocol"] = protocol["conversion"]["protocol"]
    result["conversion"]["network"]["network_build"] = {
        "protocol": (
            "roadnetsz-source-components-sumo122-turnspeed55-"
            "window-duarouter-v3"
        ),
        "junction_turn_speed_limit_mps": 5.5,
        "netconvert": {"returncode": 0},
        "duarouter": {"returncode": 0},
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    audit = audit_matrix(
        protocol_path=protocol_path,
        results_root=tmp_path / "results",
    )

    assert audit["status"] == "PASS"

    protocol["protocol"] = (
        "tsc-v94-roadnetsz-shenzhen-pcl-pre-efficacy-network-admission-v3"
    )
    protocol["conversion"]["protocol"] = (
        "cfcmt-roadnetsz-native-sumo122-window-package-v4"
    )
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    result["conversion"]["protocol"] = protocol["conversion"]["protocol"]
    result["conversion"]["network"]["network_build"] = {
        "protocol": (
            "roadnetsz-source-components-sumo122-protected-incoming-"
            "window-duarouter-v4"
        ),
        "auto_generated_tls_phase_layout": "incoming",
        "netconvert": {"returncode": 0},
        "duarouter": {"returncode": 0},
    }
    result["conversion"]["network"]["tls_semantic_audit"].update(
        {
            "auto_generated_phase_layout": "incoming",
            "auto_generated_minor_green_signal_count": 0,
        }
    )
    result_path.write_text(json.dumps(result), encoding="utf-8")

    audit = audit_matrix(
        protocol_path=protocol_path,
        results_root=tmp_path / "results",
    )

    assert audit["status"] == "PASS"
