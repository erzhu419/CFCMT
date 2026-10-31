from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_external_closed_loop_aggregate import (
    PRIMARY_METRIC,
    aggregate_closed_loop_results,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    METHOD_POLICY,
    POLICIES,
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_full_heldout_evaluation import (
    RESULT_PROTOCOL as OFFLINE_RESULT_PROTOCOL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v29_external_full_budget_confirmation.json"
)


def _write_json(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_matrix(tmp_path: Path):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    city_scenarios = protocol["target_protocol"]["external_city_scenarios"]
    seeds = protocol["target_protocol"]["closed_loop_seeds"]
    joint_path = tmp_path / "joint.json"
    joint = {
        "status": "PASS",
        "cities": {
            city: {"method_model": {"sha256": f"{city}-model"}}
            for city in city_scenarios
        },
    }
    joint_sha = _write_json(joint_path, joint)
    authorization_path = tmp_path / "authorization.json"
    authorization = {
        "protocol": OFFLINE_RESULT_PROTOCOL,
        "decision": "authorize_closed_loop_external_confirmation",
        "offline_confirmation_gate": {"passed": True},
        "joint_freeze_audit_sha256": joint_sha,
        "city_results": {
            city: {"selected_candidate": f"selected-{city}"}
            for city in city_scenarios
        },
    }
    authorization_sha = _write_json(authorization_path, authorization)
    result_root = tmp_path / "results"
    source_tree = "source-tree"
    specs = []
    index = 0
    protocol_sha = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    for city, scenarios in city_scenarios.items():
        for scenario in scenarios:
            for seed in seeds:
                for policy_index, policy in enumerate(POLICIES):
                    node = f"node{index % 6 + 1:03d}"
                    result_dir = result_root / scenario / f"seed_{seed}" / policy
                    tripinfo = result_dir / "tripinfo.xml"
                    tripinfo.parent.mkdir(parents=True, exist_ok=True)
                    tripinfo.write_text("<tripinfos/>\n", encoding="utf-8")
                    tripinfo_sha = hashlib.sha256(tripinfo.read_bytes()).hexdigest()
                    waiting = 10.0 if policy == METHOD_POLICY else 12.0 + policy_index
                    completion = 0.9 if policy == METHOD_POLICY else 0.8
                    metrics = {
                        "ok": True,
                        PRIMARY_METRIC: waiting,
                        "mean_tripinfo_duration": waiting + 20.0,
                        "mean_tripinfo_time_loss": waiting + 5.0,
                        "mean_completed_tripinfo_waiting_time": waiting - 1.0,
                        "mean_completed_tripinfo_duration": waiting + 19.0,
                        "tripinfo_unfinished_fraction": 2.0 / 12.0,
                        "system_vehicle_hours": waiting / 2.0,
                        "mean_queue_per_lane": waiting / 10.0,
                        "completion_ratio": completion,
                        "departure_service_ratio": completion + 0.05,
                        "throughput_ratio": completion + 0.02,
                        "starting_teleports": 0,
                        "ending_teleports": 0,
                        "collision_incidents": (
                            2 if policy == "fixed_time" else 0
                        ),
                        "collision_events": 2 if policy == "fixed_time" else 0,
                        "collision_incident_rate": (
                            2.0 / 12.0 if policy == "fixed_time" else 0.0
                        ),
                        "emergency_stop_rate": 0.0,
                        "tripinfo_count": 12,
                        "tripinfo_completed_count": 10,
                        "tripinfo_unfinished_count": 2,
                        "arrived": 10,
                        "departed": 12,
                        "demand_population_at_horizon": 13,
                        "emergency_stops": 0,
                    }
                    result = {
                        "protocol": RESULT_PROTOCOL,
                        "hostname": node,
                        "runtime": {
                            "hostname": node,
                            "source_tree_sha256": source_tree,
                            "libsumo_version": "1.22.0",
                        },
                        "protocol_spec_sha256": protocol_sha,
                        "offline_authorization_sha256": authorization_sha,
                        "joint_freeze_audit_sha256": joint_sha,
                        "external_manifest_sha256": "external",
                        "conversion_manifest_sha256": "conversion",
                        "conversion_tree_sha256": "tree",
                        "sumo_version": "1.22.0",
                        "city": city,
                        "scenario": scenario,
                        "seed": seed,
                        "policy": policy,
                        "selected_candidate": (
                            f"selected-{city}" if policy == METHOD_POLICY else None
                        ),
                        "model_artifacts": joint["cities"][city],
                        "anchored_blend_audit": (
                            {
                                "protocol": "online-frozen-anchored-blend-audit-v1",
                                "prediction_calls": 5,
                                "action_groups": 20,
                            }
                            if policy == METHOD_POLICY
                            else None
                        ),
                        "tripinfo_evidence": {
                            "filename": "tripinfo.xml",
                            "sha256": tripinfo_sha,
                            "size_bytes": tripinfo.stat().st_size,
                        },
                        "metrics": metrics,
                        "elapsed_sec": 1.0,
                    }
                    _write_json(result_dir / "result.json", result)
                    specs.append(
                        {
                            "signature": (
                                "CFCMT/v43r39/closed-loop-v4/"
                                f"{scenario}/seed-{seed}/{policy}"
                            ),
                            "require_node": node,
                        }
                    )
                    index += 1
    launch_path = tmp_path / "launch.json"
    launch = {
        "protocol": "v43r39-external-closed-loop-submission-v4",
        "result_protocol": RESULT_PROTOCOL,
        "submitted": True,
        "task_count": 56,
        "source_tree_sha256": source_tree,
        "offline_authorization_sha256": authorization_sha,
        "joint_freeze_audit_sha256": joint_sha,
        "external_manifest_sha256": "external",
        "conversion_manifest_sha256": "conversion",
        "conversion_tree_sha256": "tree",
        "specs": specs,
    }
    _write_json(launch_path, launch)
    return authorization_path, joint_path, launch_path, result_root


def test_closed_loop_aggregate_audits_and_weights_complete_matrix(tmp_path):
    authorization, joint, launch, result_root = _synthetic_matrix(tmp_path)

    payload = aggregate_closed_loop_results(
        protocol_spec_path=PROTOCOL_PATH,
        offline_authorization_path=authorization,
        joint_freeze_audit_path=joint,
        launch_manifest_path=launch,
        result_root=result_root,
        expected_sumo_version="1.22.0",
        bootstrap_replicates=100,
    )

    assert payload["validity_gate"]["passed"] is True
    assert payload["validity_gate"]["valid_rollout_count"] == 56
    assert payload["validity_gate"]["collision_incident_count"] == 16
    assert payload["collision_audit"]["method_vs_baselines"]["fixed_time"][
        "all_paired_rollouts_noninferior"
    ] is True
    assert payload["macro_summary"][METHOD_POLICY][PRIMARY_METRIC] == 10.0
    assert (
        payload["method_comparisons"]["fixed_time"][PRIMARY_METRIC][
            "oriented_improvement"
        ]
        == 3.0
    )


def test_closed_loop_aggregate_rejects_tampered_tripinfo(tmp_path):
    authorization, joint, launch, result_root = _synthetic_matrix(tmp_path)
    tripinfo = next(result_root.glob("**/tripinfo.xml"))
    tripinfo.write_text("<tampered/>\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tripinfo evidence mismatch"):
        aggregate_closed_loop_results(
            protocol_spec_path=PROTOCOL_PATH,
            offline_authorization_path=authorization,
            joint_freeze_audit_path=joint,
            launch_manifest_path=launch,
            result_root=result_root,
            expected_sumo_version="1.22.0",
            bootstrap_replicates=10,
        )
