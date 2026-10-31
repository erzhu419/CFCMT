from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    STRICT_SAFETY_MONITORING_V3,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    SUMO_EXECUTION_PROTOCOL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDITOR_PATH = PROJECT_ROOT / "scripts/cluster/audit_tsc_safety_survey.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_tsc_safety_survey", AUDITOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(*, collision_incidents: int = 0, teleports: int = 0):
    collision_events = collision_incidents * 2
    return {
        "ok": True,
        "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
        "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
        "collision_events": collision_events,
        "collision_event_steps": collision_incidents,
        "collision_incidents": collision_incidents,
        "collision_rate": float(collision_events) / 10.0,
        "collision_incident_rate": float(collision_incidents) / 10.0,
        "emergency_stops": 0,
        "starting_teleports": teleports,
        "ending_teleports": 0,
        "mean_system_vehicles_per_controlled_lane": 2.0,
    }


def _fixture(tmp_path: Path, *, teleports: int = 0):
    manifest_path = tmp_path / "manifest.json"
    spec_path = tmp_path / "spec.json"
    results_root = tmp_path / "results"
    scenarios = {
        "node001": ("network_a",),
        "node002": ("network_b",),
    }
    policies = ("fixed_program", "phase_pressure")
    seeds = (17, 23)
    manifest = {
        "scenarios": [
            {"scenario": "network_a", "city_group": "city_a"},
            {"scenario": "network_b", "city_group": "city_b"},
        ]
    }
    spec = {
        "manifest": str(manifest_path),
        "expected_sumo_version": "1.22.0",
        "counterfactual": {
            "control_interval_sec": 10,
            "warmup_sec": 60.0,
        },
        "safety_survey": {
            "duration_sec": 600.0,
            "policies": list(policies),
            "seeds": list(seeds),
        },
        "cache_assignments": {
            node: list(values) for node, values in scenarios.items()
        },
    }
    _write_json(manifest_path, manifest)
    _write_json(spec_path, spec)

    for node, node_scenarios in scenarios.items():
        rows = []
        for scenario in node_scenarios:
            for policy in policies:
                for seed in seeds:
                    rows.append(
                        {
                            "scenario": scenario,
                            "city_group": "city_a"
                            if scenario == "network_a"
                            else "city_b",
                            "policy": policy,
                            "seed": seed,
                            "metrics": _metrics(
                                collision_incidents=int(policy == "fixed_program"),
                                teleports=teleports,
                            ),
                        }
                    )
        aggregate = {
            scenario: {
                policy: {
                    metric: sum(
                        float(row["metrics"][metric])
                        for row in rows
                        if row["scenario"] == scenario
                        and row["policy"] == policy
                    )
                    / len(seeds)
                    for metric in AUDITOR.AUDITED_METRICS
                }
                for policy in policies
            }
            for scenario in node_scenarios
        }
        collision_rollouts = sum(
            int(row["metrics"]["collision_incidents"]) > 0 for row in rows
        )
        payload = {
            "experiment": "traffic_signal_benchmark_aware_safety_survey",
            "runtime": {
                "libsumo_version": "1.22.0",
                "source_tree_sha256": "source-hash",
                "determinism_environment": dict(
                    AUDITOR.EXPECTED_DETERMINISM_ENVIRONMENT
                ),
            },
            "setting": {
                "scenarios": list(node_scenarios),
                "policies": list(policies),
                "seeds": list(seeds),
                "duration_sec": 600.0,
                "control_interval_sec": 10,
                "warmup_sec": 60.0,
                "libsumo_version": "1.22.0",
                "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
                "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
                "collision_interpretation": (
                    "raw_benchmark_events_audited_not_blanket_controller_failure"
                ),
            },
            "expected_row_count": len(rows),
            "row_count": len(rows),
            "raw_collision_rollout_count": collision_rollouts,
            "raw_collision_rollout_fraction": collision_rollouts / len(rows),
            "passed": teleports == 0,
            "errors": [] if teleports == 0 else ["teleport event"],
            "aggregate": aggregate,
            "rows": rows,
        }
        _write_json(results_root / node / "safety_survey.json", payload)
    return results_root, spec_path


def test_auditor_reconstructs_complete_rollout_matrix(tmp_path: Path) -> None:
    results_root, spec_path = _fixture(tmp_path)

    summary = AUDITOR.audit_safety_survey(
        results_root=results_root,
        spec_path=spec_path,
        expected_source_sha256="source-hash",
    )

    assert summary["passed"] is True
    assert summary["node_count"] == 2
    assert summary["scenario_count"] == 2
    assert summary["rollout_count"] == 8
    assert summary["raw_collision_rollout_count"] == 4
    assert summary["policy_summary"]["fixed_program"][
        "collision_incidents"
    ] == 4


def test_auditor_rejects_teleport_even_when_shard_is_relabelled_passed(
    tmp_path: Path,
) -> None:
    results_root, spec_path = _fixture(tmp_path, teleports=1)
    for path in results_root.glob("**/safety_survey.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["passed"] = True
        payload["errors"] = []
        _write_json(path, payload)

    with pytest.raises(RuntimeError, match="teleport event"):
        AUDITOR.audit_safety_survey(
            results_root=results_root,
            spec_path=spec_path,
            expected_source_sha256="source-hash",
        )


def test_auditor_rejects_missing_node_partition(tmp_path: Path) -> None:
    results_root, spec_path = _fixture(tmp_path)
    (results_root / "node002" / "safety_survey.json").unlink()

    with pytest.raises(RuntimeError, match="node partition mismatch"):
        AUDITOR.audit_safety_survey(
            results_root=results_root,
            spec_path=spec_path,
            expected_source_sha256="source-hash",
        )
