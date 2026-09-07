#!/usr/bin/env python3
"""Audit estimand-aligned development rollouts and apply the frozen advance gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (  # noqa: E402
    ALIGNED_AUDIT_DECISION,
    ALIGNED_PROTOCOL,
    DEVELOPMENT_RESULT_PROTOCOL,
    METHOD,
)
from scripts.cluster.launch_tsc_external_estimand_aligned_rollouts import (  # noqa: E402
    LAUNCH_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v46r42-oof-validated-support-development-audit-v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _tripinfo_semantic_sha256(path: Path) -> str:
    root = ET.parse(path).getroot()
    return hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest()


def audit_estimand_aligned_development(
    *,
    results_root: Path,
    launch_manifest: Path,
    aligned_protocol_path: Path,
    aligned_joint_audit_path: Path,
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    protocol = _read_json(aligned_protocol_path)
    joint = _read_json(aligned_joint_audit_path)
    development = dict(protocol["development"])
    scenarios_by_city = {
        str(city): tuple(str(value) for value in scenarios)
        for city, scenarios in protocol["prospective_confirmation_reservation"][
            "city_scenarios"
        ].items()
    }
    scenario_city = {
        scenario: city
        for city, scenarios in scenarios_by_city.items()
        for scenario in scenarios
    }
    seeds = tuple(int(value) for value in development["closed_loop_seeds"])
    policies = tuple(str(value) for value in development["policies"])
    expected = {
        (scenario, seed, policy)
        for scenario in scenario_city
        for seed in seeds
        for policy in policies
    }
    spec_by_key = {}
    for spec in launch.get("specs", ()):
        result_parts = Path(str(spec["result_dir"])).parts
        key = (
            result_parts[-3],
            int(result_parts[-2].removeprefix("seed_")),
            result_parts[-1],
        )
        spec_by_key[key] = spec
    launch_gate = {
        "launch_protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "aligned_protocol_matches": protocol.get("protocol") == ALIGNED_PROTOCOL,
        "analysis_stage_development": launch.get("analysis_stage") == "development",
        "submitted": bool(launch.get("submitted", False)),
        "joint_audit_passed": joint.get("status") == "PASS",
        "joint_decision_matches": joint.get("decision") == ALIGNED_AUDIT_DECISION,
        "joint_hash_matches_protocol": _sha256(aligned_joint_audit_path)
        == str(protocol["estimand_aligned_joint_audit"]["sha256"]),
        "matrix_exact": set(spec_by_key) == expected
        and len(spec_by_key) == len(launch.get("specs", ())) == len(expected),
        "prospective_seeds_absent": not (
            {seed for _, seed, _ in spec_by_key}
            & {
                int(value)
                for value in protocol["prospective_confirmation_reservation"][
                    "closed_loop_seeds"
                ]
            }
        ),
    }
    launch_gate["passed"] = all(launch_gate.values())

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for scenario, seed, policy in sorted(expected):
        root = Path(results_root) / scenario / f"seed_{seed}" / policy
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        if not result_path.is_file() or not tripinfo_path.is_file():
            errors.append(f"missing result artifact: {scenario}/{seed}/{policy}")
            continue
        result = _read_json(result_path)
        metrics = dict(result.get("metrics", {}))
        spec = spec_by_key[(scenario, seed, policy)]
        tripinfo_sha = _sha256(tripinfo_path)
        row_gate = {
            "result_protocol_matches": result.get("protocol")
            == DEVELOPMENT_RESULT_PROTOCOL,
            "analysis_stage_matches": result.get("analysis_stage") == "development",
            "identity_matches": (
                result.get("scenario"),
                int(result.get("seed", -1)),
                result.get("policy"),
            )
            == (scenario, seed, policy),
            "city_matches": result.get("city") == scenario_city[scenario],
            "physical_host_matches": result.get("hostname") == spec.get("require_node"),
            "source_tree_matches_snapshot": result.get("runtime", {}).get(
                "source_tree_sha256"
            )
            == launch.get("source_tree_sha256"),
            "aligned_protocol_hash_matches": result.get(
                "estimand_aligned_protocol_sha256"
            )
            == _sha256(aligned_protocol_path),
            "aligned_audit_hash_matches": result.get(
                "estimand_aligned_joint_audit_sha256"
            )
            == _sha256(aligned_joint_audit_path),
            "tripinfo_hash_matches": result.get("tripinfo_evidence", {}).get("sha256")
            == tripinfo_sha,
            "tripinfo_size_matches": int(
                result.get("tripinfo_evidence", {}).get("size_bytes", -1)
            )
            == tripinfo_path.stat().st_size,
            "sumo_succeeded": bool(metrics.get("ok", False)),
        }
        if policy == METHOD:
            city_spec = protocol["city_artifacts"][scenario_city[scenario]]
            row_gate.update(
                {
                    "certificate_hash_matches": result.get(
                        "estimand_aligned_certificate_sha256"
                    )
                    == str(city_spec["certificate"]["sha256"]),
                    "model_hash_matches": result.get(
                        "estimand_aligned_model_sha256"
                    )
                    == str(city_spec["model"]["sha256"]),
                    "estimand_alignment_passed": bool(
                        result.get("estimand_alignment", {}).get("passed", False)
                    ),
                }
            )
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"result gate failed: {scenario}/{seed}/{policy}")
        rows.append(
            {
                "city": scenario_city[scenario],
                "scenario": scenario,
                "seed": seed,
                "policy": policy,
                "hostname": result.get("hostname"),
                "result_path": str(result_path.resolve()),
                "result_sha256": _sha256(result_path),
                "tripinfo_sha256": tripinfo_sha,
                "tripinfo_semantic_sha256": _tripinfo_semantic_sha256(
                    tripinfo_path
                ),
                "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
                "mean_travel_time": float(metrics["mean_tripinfo_duration"]),
                "executed_overrides": int(
                    metrics.get("guard_audit", {}).get(
                        "executed_overrides",
                        metrics.get("guard_audit", {}).get("accepted_overrides", 0),
                    )
                ),
                "safety": {
                    "starting_teleports": int(metrics.get("starting_teleports", -1)),
                    "ending_teleports": int(metrics.get("ending_teleports", -1)),
                    "collision_events": int(metrics.get("collision_events", -1)),
                    "collision_incidents": int(metrics.get("collision_incidents", -1)),
                },
                "gate": row_gate,
                "metrics": metrics,
            }
        )

    keyed = {(row["scenario"], row["seed"], row["policy"]): row for row in rows}
    paired_rows = []
    for scenario in scenario_city:
        for seed in seeds:
            method = keyed.get((scenario, seed, METHOD))
            phase = keyed.get((scenario, seed, "phase_pressure"))
            if method is None or phase is None:
                continue
            method_wait = float(method["mean_waiting_time"])
            phase_wait = float(phase["mean_waiting_time"])
            paired_rows.append(
                {
                    "city": scenario_city[scenario],
                    "scenario": scenario,
                    "seed": seed,
                    "method_mean_waiting_time": method_wait,
                    "phase_mean_waiting_time": phase_wait,
                    "relative_delta": (method_wait - phase_wait) / max(phase_wait, 1e-12),
                    "executed_overrides": method["executed_overrides"],
                    "method_safety": method["safety"],
                    "phase_safety": phase["safety"],
                    "metrics_exact_match": method["metrics"] == phase["metrics"],
                    "tripinfo_semantic_exact_match": method[
                        "tripinfo_semantic_sha256"
                    ]
                    == phase["tripinfo_semantic_sha256"],
                }
            )

    city_summary = {}
    for city in scenarios_by_city:
        selected = [row for row in paired_rows if row["city"] == city]
        method_mean = float(np.mean([row["method_mean_waiting_time"] for row in selected]))
        phase_mean = float(np.mean([row["phase_mean_waiting_time"] for row in selected]))
        city_summary[city] = {
            "paired_units": len(selected),
            "method_mean_waiting_time": method_mean,
            "phase_mean_waiting_time": phase_mean,
            "relative_delta": (method_mean - phase_mean) / max(phase_mean, 1e-12),
            "worst_paired_relative_delta": max(row["relative_delta"] for row in selected),
            "executed_overrides": sum(row["executed_overrides"] for row in selected),
            "minimum_executed_overrides_per_seed": min(
                (row["executed_overrides"] for row in selected), default=0
            ),
            "all_metrics_exact_match": all(row["metrics_exact_match"] for row in selected),
            "all_tripinfo_semantic_exact_match": all(
                row["tripinfo_semantic_exact_match"] for row in selected
            ),
        }

    threshold = dict(development["advance_gate"])
    la = city_summary.get("los_angeles", {})
    jinan = city_summary.get("jinan", {})
    integrity_gate = {
        "launch_gate_passed": bool(launch_gate["passed"]),
        "all_expected_results_present": len(rows) == len(expected),
        "all_result_gates_passed": bool(rows)
        and all(row["gate"]["passed"] for row in rows),
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    advance_gate = {
        "los_angeles_city_mean_noninferior": float(la.get("relative_delta", np.inf))
        <= float(
            threshold[
                "los_angeles_city_mean_relative_regression_vs_phase_pressure_maximum"
            ]
        ),
        "los_angeles_worst_seed_within_limit": float(
            la.get("worst_paired_relative_delta", np.inf)
        )
        <= float(
            threshold[
                "los_angeles_worst_seed_relative_regression_vs_phase_pressure_maximum"
            ]
        ),
        "los_angeles_intervened_each_seed": int(
            la.get("minimum_executed_overrides_per_seed", 0)
        )
        >= int(threshold["los_angeles_minimum_executed_interventions_per_seed"]),
        "jinan_exact_phase_fallback": bool(jinan.get("all_metrics_exact_match", False))
        and bool(jinan.get("all_tripinfo_semantic_exact_match", False)),
        "method_teleports_noninferior_to_paired_phase": all(
            int(row["method_safety"]["ending_teleports"])
            <= int(row["phase_safety"]["ending_teleports"])
            for row in paired_rows
        ),
        "method_collision_incidents_noninferior_to_paired_phase": all(
            int(row["method_safety"]["collision_incidents"])
            <= int(row["phase_safety"]["collision_incidents"])
            for row in paired_rows
        ),
        "result_and_tripinfo_hash_integrity": bool(integrity_gate["passed"]),
    }
    advance_gate["passed"] = all(advance_gate.values())
    decision = (
        "authorize_prospective_confirmation_analysis_freeze"
        if advance_gate["passed"]
        else "reject_prospective_confirmation_and_return_to_method_redesign"
    )
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": decision,
        "claim_boundary": (
            "seeds 8081/9091 were previously inspected and therefore provide "
            "method-redesign falsification evidence only"
        ),
        "launch_manifest_sha256": _sha256(launch_manifest),
        "aligned_protocol_sha256": _sha256(aligned_protocol_path),
        "aligned_joint_audit_sha256": _sha256(aligned_joint_audit_path),
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "launch_gate": launch_gate,
        "rows": [
            {key: value for key, value in row.items() if key != "metrics"}
            for row in rows
        ],
        "paired_rows": paired_rows,
        "city_summary": city_summary,
        "integrity_gate": integrity_gate,
        "advance_gate": advance_gate,
        "prospective_confirmation_seeds_remain_unrun": list(
            protocol["prospective_confirmation_reservation"]["closed_loop_seeds"]
        ),
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--aligned-protocol", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite development audit: {args.out}")
    payload = audit_estimand_aligned_development(
        results_root=args.results_root,
        launch_manifest=args.launch_manifest,
        aligned_protocol_path=args.aligned_protocol,
        aligned_joint_audit_path=args.aligned_joint_audit,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "advance_gate_passed": payload["advance_gate"]["passed"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
