#!/usr/bin/env python3
"""Audit the contaminated v46 execution-semantics factorial diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
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
    EXECUTION_DIAGNOSTIC_RESULT_PROTOCOL,
    EXECUTION_DIAGNOSTICS,
    EXECUTION_VARIANTS,
    METHOD,
)
from scripts.cluster.launch_tsc_external_estimand_aligned_rollouts import (  # noqa: E402
    LAUNCH_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v46r42-external-execution-factorial-audit-v1"
ANALYSIS_STATUS = "post_failure_contaminated_execution_factorial_not_confirmation"
BASE_METHOD = METHOD
BASELINE = "phase_pressure"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _tripinfo_semantic_sha256(path: Path) -> str:
    root = ET.parse(path).getroot()
    return hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest()


def _safety(metrics: Mapping[str, Any]) -> dict[str, int]:
    return {
        "starting_teleports": int(metrics.get("starting_teleports", -1)),
        "ending_teleports": int(metrics.get("ending_teleports", -1)),
        "collision_events": int(metrics.get("collision_events", -1)),
        "collision_incidents": int(metrics.get("collision_incidents", -1)),
    }


def _executed_overrides(metrics: Mapping[str, Any]) -> int:
    audit = dict(metrics.get("guard_audit", {}))
    return int(audit.get("executed_overrides", audit.get("accepted_overrides", 0)))


def _load_reference(
    *, base_results_root: Path, scenario: str, seed: int, policy: str
) -> dict[str, Any]:
    root = Path(base_results_root) / scenario / f"seed_{seed}" / policy
    result_path = root / "result.json"
    tripinfo_path = root / "tripinfo.xml"
    if not result_path.is_file() or not tripinfo_path.is_file():
        raise FileNotFoundError(f"missing base result: {scenario}/{seed}/{policy}")
    result = _read_json(result_path)
    metrics = dict(result["metrics"])
    return {
        "policy": policy,
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "tripinfo_sha256": _sha256(tripinfo_path),
        "tripinfo_semantic_sha256": _tripinfo_semantic_sha256(tripinfo_path),
        "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
        "mean_travel_time": float(metrics["mean_tripinfo_duration"]),
        "executed_overrides": _executed_overrides(metrics),
        "safety": _safety(metrics),
    }


def audit_execution_factorial(
    *,
    results_root: Path,
    base_results_root: Path,
    launch_manifest: Path,
    aligned_protocol_path: Path,
    aligned_joint_audit_path: Path,
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    protocol = _read_json(aligned_protocol_path)
    joint = _read_json(aligned_joint_audit_path)
    seeds = tuple(int(value) for value in protocol["development"]["closed_loop_seeds"])
    scenarios = tuple(
        str(value)
        for value in protocol["prospective_confirmation_reservation"][
            "city_scenarios"
        ]["los_angeles"]
    )
    expected = {
        (scenario, seed, variant)
        for scenario in scenarios
        for seed in seeds
        for variant in EXECUTION_VARIANTS
    }
    spec_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for spec in launch.get("specs", ()):
        parts = Path(str(spec["result_dir"])).parts
        policy = str(parts[-1])
        prefix = f"{METHOD}_"
        variant = policy.removeprefix(prefix)
        key = (str(parts[-3]), int(parts[-2].removeprefix("seed_")), variant)
        spec_by_key[key] = dict(spec)

    launch_gate = {
        "launch_protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "result_protocol_matches": launch.get("result_protocol")
        == EXECUTION_DIAGNOSTIC_RESULT_PROTOCOL,
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
    for scenario, seed, variant in sorted(expected):
        policy = f"{METHOD}_{variant}"
        root = Path(results_root) / scenario / f"seed_{seed}" / policy
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        if not result_path.is_file() or not tripinfo_path.is_file():
            errors.append(f"missing result artifact: {scenario}/{seed}/{variant}")
            continue
        result = _read_json(result_path)
        metrics = dict(result.get("metrics", {}))
        diagnostic = dict(result.get("execution_diagnostic", {}))
        post_hoc = dict(result.get("post_hoc_diagnostic", {}))
        runtime_override = dict(result.get("method_runtime_override", {}))
        provenance = dict(runtime_override.get("provenance", {}))
        expected_spec = EXECUTION_DIAGNOSTICS[variant]
        spec = spec_by_key[(scenario, seed, variant)]
        tripinfo_sha = _sha256(tripinfo_path)
        row_gate = {
            "result_protocol_matches": result.get("protocol")
            == EXECUTION_DIAGNOSTIC_RESULT_PROTOCOL,
            "identity_matches": (
                result.get("scenario"),
                int(result.get("seed", -1)),
                result.get("policy"),
            )
            == (scenario, seed, policy),
            "city_matches": result.get("city") == "los_angeles",
            "physical_host_matches": result.get("hostname")
            == spec.get("require_node"),
            "source_tree_matches_snapshot": result.get("runtime", {}).get(
                "source_tree_sha256"
            )
            == launch.get("source_tree_sha256"),
            "sumo_version_matches": result.get("sumo_version") == "1.22.0",
            "aligned_protocol_hash_matches": result.get(
                "estimand_aligned_protocol_sha256"
            )
            == _sha256(aligned_protocol_path),
            "aligned_audit_hash_matches": result.get(
                "estimand_aligned_joint_audit_sha256"
            )
            == _sha256(aligned_joint_audit_path),
            "certificate_hash_matches": result.get(
                "estimand_aligned_certificate_sha256"
            )
            == str(protocol["city_artifacts"]["los_angeles"]["certificate"]["sha256"]),
            "model_hash_matches": result.get("estimand_aligned_model_sha256")
            == str(protocol["city_artifacts"]["los_angeles"]["model"]["sha256"]),
            "same_frozen_model_declared": bool(
                diagnostic.get("same_frozen_model_as_v46", False)
            ),
            "not_prospective": diagnostic.get("prospective_evidence") is False,
            "variant_matches": diagnostic.get("variant") == variant
            and provenance.get("execution_variant") == variant,
            "execution_semantics_match": (
                bool(diagnostic.get("guarded"))
                == bool(expected_spec["guarded"])
                and diagnostic.get("coordination_mode")
                == expected_spec["coordination_mode"]
                and diagnostic.get("cooldown_intervals_override")
                == expected_spec["cooldown_intervals_override"]
            ),
            "analysis_status_matches": post_hoc.get("analysis_status")
            == ANALYSIS_STATUS,
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
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"result gate failed: {scenario}/{seed}/{variant}")
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "variant": variant,
                "policy": policy,
                "hostname": result.get("hostname"),
                "result_path": str(result_path.resolve()),
                "result_sha256": _sha256(result_path),
                "tripinfo_sha256": tripinfo_sha,
                "tripinfo_semantic_sha256": _tripinfo_semantic_sha256(
                    tripinfo_path
                ),
                "mean_waiting_time": float(
                    metrics["mean_tripinfo_waiting_time"]
                ),
                "mean_travel_time": float(metrics["mean_tripinfo_duration"]),
                "executed_overrides": _executed_overrides(metrics),
                "proposed_overrides": int(
                    metrics.get("guard_audit", {}).get("proposed_overrides", 0)
                ),
                "safety": _safety(metrics),
                "gate": row_gate,
            }
        )

    references: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        references[seed] = {
            policy: _load_reference(
                base_results_root=base_results_root,
                scenario=scenarios[0],
                seed=seed,
                policy=policy,
            )
            for policy in (BASELINE, BASE_METHOD)
        }

    paired_rows = []
    for row in rows:
        phase = references[int(row["seed"])][BASELINE]
        phase_wait = float(phase["mean_waiting_time"])
        paired_rows.append(
            {
                **{key: value for key, value in row.items() if key != "gate"},
                "phase_mean_waiting_time": phase_wait,
                "relative_delta_vs_phase": (
                    float(row["mean_waiting_time"]) - phase_wait
                )
                / max(phase_wait, 1e-12),
                "teleports_noninferior": int(row["safety"]["ending_teleports"])
                <= int(phase["safety"]["ending_teleports"]),
                "collision_incidents_noninferior": int(
                    row["safety"]["collision_incidents"]
                )
                <= int(phase["safety"]["collision_incidents"]),
            }
        )

    summary = {}
    for variant in EXECUTION_VARIANTS:
        selected = [row for row in paired_rows if row["variant"] == variant]
        waits = np.asarray([row["mean_waiting_time"] for row in selected], dtype=float)
        phases = np.asarray(
            [row["phase_mean_waiting_time"] for row in selected], dtype=float
        )
        relative = np.asarray(
            [row["relative_delta_vs_phase"] for row in selected], dtype=float
        )
        summary[variant] = {
            "paired_units": len(selected),
            "mean_waiting_time": float(np.mean(waits)),
            "phase_mean_waiting_time": float(np.mean(phases)),
            "relative_delta_of_means": float(
                (np.mean(waits) - np.mean(phases)) / max(np.mean(phases), 1e-12)
            ),
            "mean_paired_relative_delta": float(np.mean(relative)),
            "worst_paired_relative_delta": float(np.max(relative)),
            "best_paired_relative_delta": float(np.min(relative)),
            "executed_overrides": int(
                sum(int(row["executed_overrides"]) for row in selected)
            ),
            "minimum_executed_overrides_per_seed": int(
                min((int(row["executed_overrides"]) for row in selected), default=0)
            ),
            "teleports_noninferior_all": all(
                bool(row["teleports_noninferior"]) for row in selected
            ),
            "collision_incidents_noninferior_all": all(
                bool(row["collision_incidents_noninferior"]) for row in selected
            ),
        }
        summary[variant]["development_gate_passed"] = bool(
            summary[variant]["relative_delta_of_means"] <= 0.0
            and summary[variant]["worst_paired_relative_delta"] <= 0.02
            and summary[variant]["minimum_executed_overrides_per_seed"] >= 1
            and summary[variant]["teleports_noninferior_all"]
            and summary[variant]["collision_incidents_noninferior_all"]
        )

    integrity_gate = {
        "launch_gate_passed": bool(launch_gate["passed"]),
        "all_expected_results_present": len(rows) == len(expected),
        "all_result_gates_passed": bool(rows)
        and all(bool(row["gate"]["passed"]) for row in rows),
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    passing = [
        variant
        for variant in EXECUTION_VARIANTS
        if bool(summary[variant]["development_gate_passed"])
    ]
    best_mean_variant = min(
        EXECUTION_VARIANTS,
        key=lambda variant: float(summary[variant]["relative_delta_of_means"]),
    )
    decision = (
        "execution_variant_available_for_robustness_development"
        if passing
        else "no_execution_variant_passes_robust_gate_continue_method_redesign"
    )
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": decision,
        "claim_boundary": (
            "seeds 8081/9091 are contaminated development evidence; execution "
            "variants may diagnose and redesign the method but cannot confirm efficacy"
        ),
        "launch_manifest_sha256": _sha256(launch_manifest),
        "aligned_protocol_sha256": _sha256(aligned_protocol_path),
        "aligned_joint_audit_sha256": _sha256(aligned_joint_audit_path),
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "launch_gate": launch_gate,
        "rows": rows,
        "references": references,
        "paired_rows": paired_rows,
        "variant_summary": summary,
        "best_mean_variant": best_mean_variant,
        "variants_passing_development_gate": passing,
        "integrity_gate": integrity_gate,
        "prospective_confirmation_seeds_remain_unrun": list(
            protocol["prospective_confirmation_reservation"]["closed_loop_seeds"]
        ),
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--base-results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--aligned-protocol", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite execution audit: {args.out}")
    payload = audit_execution_factorial(
        results_root=args.results_root,
        base_results_root=args.base_results_root,
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
                "best_mean_variant": payload["best_mean_variant"],
                "passing_variants": payload["variants_passing_development_gate"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
