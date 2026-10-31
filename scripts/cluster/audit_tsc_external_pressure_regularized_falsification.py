#!/usr/bin/env python3
"""Audit the frozen v47 contaminated falsification matrix.

This audit establishes evidence integrity and applies the predeclared rejection
gate.  Seeds 8081/9091 are contaminated development evidence and cannot support
an efficacy claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation import (  # noqa: E402
    FALSIFICATION_RESULT_PROTOCOL,
    JOINT_AUDIT_DECISION,
    JOINT_AUDIT_PROTOCOL,
    METHOD,
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_pressure_regularized_rollouts import (  # noqa: E402
    LAUNCH_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v47r43-pressure-regularized-falsification-audit-v1"
REJECTION_DECISION = (
    "reject_v47_and_preserve_robustness_and_prospective_seeds_for_successor_method"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _tripinfo_semantic_sha256(path: Path) -> str:
    root = ET.parse(path).getroot()
    return hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest()


def _spec_key(spec: Mapping[str, Any]) -> tuple[str, int, str]:
    parts = Path(str(spec["result_dir"])).parts
    return parts[-3], int(parts[-2].removeprefix("seed_")), parts[-1]


def _safety(metrics: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: int(metrics.get(key, -1))
        for key in (
            "starting_teleports",
            "ending_teleports",
            "collision_events",
            "collision_incidents",
        )
    }


def audit_pressure_regularized_falsification(
    *,
    results_root: Path,
    launch_manifest: Path,
    pressure_protocol_path: Path,
    pressure_joint_audit_path: Path,
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    protocol = _read_json(pressure_protocol_path)
    joint = _read_json(pressure_joint_audit_path)
    falsification = dict(protocol["contaminated_falsification"])
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
    seeds = tuple(int(value) for value in falsification["closed_loop_seeds"])
    policies = tuple(str(value) for value in falsification["policies"])
    expected = {
        (scenario, seed, policy)
        for scenario in scenario_city
        for seed in seeds
        for policy in policies
    }
    specs = tuple(launch.get("specs", ()))
    spec_by_key = {_spec_key(spec): spec for spec in specs}
    submitted = json.loads(str(launch.get("scheduler_stdout", "{}")) or "{}")
    submitted_by_signature = {
        str(item["signature"]): item for item in submitted.get("submitted", ())
    }
    robustness_seeds = {
        int(value)
        for value in protocol["untouched_robustness_development"][
            "closed_loop_seeds"
        ]
    }
    prospective_seeds = {
        int(value)
        for value in protocol["prospective_confirmation_reservation"][
            "closed_loop_seeds"
        ]
    }
    launch_seeds = {key[1] for key in spec_by_key}
    launch_gate = {
        "launch_protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "result_protocol_matches": launch.get("result_protocol")
        == FALSIFICATION_RESULT_PROTOCOL,
        "pressure_protocol_matches": protocol.get("protocol") == PROTOCOL,
        "joint_audit_protocol_matches": joint.get("protocol")
        == JOINT_AUDIT_PROTOCOL,
        "joint_audit_passed": joint.get("status") == "PASS",
        "joint_decision_matches": joint.get("decision") == JOINT_AUDIT_DECISION,
        "joint_hash_matches_protocol": _sha256(pressure_joint_audit_path)
        == str(protocol["pressure_regularized_joint_audit"]["sha256"]),
        "submitted": bool(launch.get("submitted", False)),
        "scheduler_returncode_zero": int(launch.get("scheduler_returncode", -1))
        == 0,
        "matrix_exact": set(spec_by_key) == expected
        and len(spec_by_key) == len(specs) == len(expected),
        "scheduler_submission_exact": len(submitted_by_signature) == len(expected)
        and set(submitted_by_signature)
        == {str(spec["signature"]) for spec in specs},
        "robustness_seeds_absent": not (launch_seeds & robustness_seeds),
        "prospective_seeds_absent": not (launch_seeds & prospective_seeds),
    }
    launch_gate["passed"] = all(launch_gate.values())

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    protocol_sha = _sha256(pressure_protocol_path)
    joint_sha = _sha256(pressure_joint_audit_path)
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
        submitted_spec = submitted_by_signature.get(str(spec["signature"]), {})
        tripinfo_sha = _sha256(tripinfo_path)
        row_gate = {
            "result_protocol_matches": result.get("protocol")
            == FALSIFICATION_RESULT_PROTOCOL,
            "analysis_stage_matches": result.get("analysis_stage")
            == "falsification",
            "identity_matches": (
                result.get("scenario"),
                int(result.get("seed", -1)),
                result.get("policy"),
            )
            == (scenario, seed, policy),
            "city_matches": result.get("city") == scenario_city[scenario],
            "physical_host_matches": result.get("hostname")
            == spec.get("require_node")
            == submitted_spec.get("require_node"),
            "source_tree_matches_snapshot": result.get("runtime", {}).get(
                "source_tree_sha256"
            )
            == launch.get("source_tree_sha256"),
            "pressure_protocol_hash_matches": result.get(
                "pressure_regularized_protocol_sha256"
            )
            == protocol_sha,
            "pressure_joint_hash_matches": result.get(
                "pressure_regularized_joint_audit_sha256"
            )
            == joint_sha,
            "tripinfo_hash_matches": result.get("tripinfo_evidence", {}).get(
                "sha256"
            )
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
                        "pressure_regularized_certificate_sha256"
                    )
                    == str(city_spec["certificate"]["sha256"]),
                    "model_hash_matches": result.get(
                        "pressure_regularized_model_sha256"
                    )
                    == str(city_spec["model"]["sha256"]),
                    "regularizer_enabled": bool(
                        result.get("pressure_regularized_deployment", {})
                        .get("regularizer", {})
                        .get("enabled", False)
                    ),
                }
            )
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"result gate failed: {scenario}/{seed}/{policy}")
        guard = dict(metrics.get("guard_audit", {}))
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
                    guard.get("executed_overrides", guard.get("accepted_overrides", 0))
                ),
                "safety": _safety(metrics),
                "gate": row_gate,
            }
        )

    keyed = {(row["scenario"], row["seed"], row["policy"]): row for row in rows}
    paired_rows: list[dict[str, Any]] = []
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
                    "relative_delta": (method_wait - phase_wait)
                    / max(phase_wait, 1e-12),
                    "executed_overrides": int(method["executed_overrides"]),
                    "method_safety": method["safety"],
                    "phase_safety": phase["safety"],
                }
            )

    city_summary: dict[str, Any] = {}
    for city in scenarios_by_city:
        selected = [row for row in paired_rows if row["city"] == city]
        method_mean = statistics.fmean(
            float(row["method_mean_waiting_time"]) for row in selected
        )
        phase_mean = statistics.fmean(
            float(row["phase_mean_waiting_time"]) for row in selected
        )
        city_summary[city] = {
            "paired_units": len(selected),
            "method_mean_waiting_time": method_mean,
            "phase_mean_waiting_time": phase_mean,
            "relative_delta": (method_mean - phase_mean) / max(phase_mean, 1e-12),
            "worst_paired_relative_delta": max(
                float(row["relative_delta"]) for row in selected
            ),
            "improved_pair_fraction": statistics.fmean(
                float(row["relative_delta"] <= 0.0) for row in selected
            ),
            "executed_overrides": sum(
                int(row["executed_overrides"]) for row in selected
            ),
            "minimum_executed_overrides_per_pair": min(
                int(row["executed_overrides"]) for row in selected
            ),
        }

    integrity_gate = {
        "launch_gate_passed": bool(launch_gate["passed"]),
        "all_expected_results_present": len(rows) == len(expected),
        "all_result_gates_passed": bool(rows)
        and all(row["gate"]["passed"] for row in rows),
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    threshold = dict(falsification["advance_gate"])
    efficacy_gate = {
        "each_city_mean_noninferior": all(
            float(summary["relative_delta"])
            <= float(
                threshold[
                    "each_city_mean_relative_regression_vs_phase_pressure_maximum"
                ]
            )
            for summary in city_summary.values()
        ),
        "worst_pair_within_limit": max(
            float(row["relative_delta"]) for row in paired_rows
        )
        <= float(
            threshold[
                "worst_scenario_seed_relative_regression_vs_phase_pressure_maximum"
            ]
        ),
        "intervened_each_active_city_seed": all(
            int(summary["minimum_executed_overrides_per_pair"])
            >= int(
                threshold[
                    "minimum_executed_interventions_per_active_city_seed"
                ]
            )
            for summary in city_summary.values()
        ),
        "teleports_noninferior": all(
            int(row["method_safety"]["ending_teleports"])
            <= int(row["phase_safety"]["ending_teleports"])
            for row in paired_rows
        ),
        "collision_incidents_noninferior": all(
            int(row["method_safety"]["collision_incidents"])
            <= int(row["phase_safety"]["collision_incidents"])
            for row in paired_rows
        ),
        "result_and_tripinfo_hash_integrity": bool(integrity_gate["passed"]),
    }
    efficacy_gate["passed"] = all(efficacy_gate.values())
    robustness_paths = [
        str(path.resolve())
        for seed in sorted(robustness_seeds)
        for path in Path(results_root).rglob(f"seed_{seed}")
    ]
    prospective_paths = [
        str(path.resolve())
        for seed in sorted(prospective_seeds)
        for path in Path(results_root).rglob(f"seed_{seed}")
    ]
    reservation_gate = {
        "robustness_results_absent": not robustness_paths,
        "prospective_results_absent": not prospective_paths,
    }
    reservation_gate["passed"] = all(reservation_gate.values())
    decision = (
        "authorize_v47_untouched_robustness_development"
        if efficacy_gate["passed"]
        else REJECTION_DECISION
    )
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS"
            if integrity_gate["passed"] and reservation_gate["passed"]
            else "FAIL"
        ),
        "decision": decision,
        "claim_boundary": (
            "Seeds 8081/9091 are contaminated method-redesign evidence only. "
            "A failed efficacy gate rejects v47 and does not license any claim."
        ),
        "launch_manifest_sha256": _sha256(launch_manifest),
        "pressure_protocol_sha256": protocol_sha,
        "pressure_joint_audit_sha256": joint_sha,
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "launch_gate": launch_gate,
        "rows": rows,
        "paired_rows": paired_rows,
        "city_summary": city_summary,
        "integrity_gate": integrity_gate,
        "efficacy_gate": efficacy_gate,
        "reservation_gate": reservation_gate,
        "untouched_robustness_seeds_remain_unrun": sorted(robustness_seeds),
        "prospective_confirmation_seeds_remain_unrun": sorted(prospective_seeds),
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--pressure-protocol", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite falsification audit: {args.out}")
    payload = audit_pressure_regularized_falsification(
        results_root=args.results_root,
        launch_manifest=args.launch_manifest,
        pressure_protocol_path=args.pressure_protocol,
        pressure_joint_audit_path=args.pressure_joint_audit,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "efficacy_gate_passed": payload["efficacy_gate"]["passed"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
