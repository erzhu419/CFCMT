#!/usr/bin/env python3
"""Audit v48 long-horizon adaptation rollouts and freeze city selectors."""

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
from cf_h2o.eval.traffic_signal_external_policy_horizon_adaptation import (  # noqa: E402
    FAILED_V47_DECISION,
    PHASE_POLICY,
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
    policy_candidates,
)
from scripts.cluster.launch_tsc_external_policy_horizon_search import (  # noqa: E402
    LAUNCH_PROTOCOL,
    NODES,
)
from scripts.cluster.run_tsc_external_policy_horizon_search_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v48r44-policy-horizon-adaptation-selection-audit-v1"
ADVANCE_DECISION = "authorize_frozen_v48_selector_for_untouched_robustness"
REJECTION_DECISION = "reject_v48_before_untouched_robustness"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _tripinfo_semantic_sha256(path: Path) -> str:
    root = ET.parse(path).getroot()
    return hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest()


def summarize_candidate(
    *,
    city: str,
    candidate: Mapping[str, Any],
    paired_rows: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    selection_rule: Mapping[str, Any],
) -> dict[str, Any]:
    key = str(candidate["key"])
    selected = [
        row
        for row in paired_rows
        if row["city"] == city and row["candidate"] == key
    ]
    expected_count = len(scenarios) * len(seeds)
    if len(selected) != expected_count:
        raise ValueError(f"candidate pairing incomplete for {city}/{key}")
    seed_deltas = {}
    for seed in seeds:
        seed_rows = [row for row in selected if int(row["seed"]) == int(seed)]
        method_mean = statistics.fmean(
            float(row["method_mean_waiting_time"]) for row in seed_rows
        )
        phase_mean = statistics.fmean(
            float(row["phase_mean_waiting_time"]) for row in seed_rows
        )
        seed_deltas[str(seed)] = (method_mean - phase_mean) / max(phase_mean, 1e-12)
    values = tuple(float(value) for value in seed_deltas.values())
    mean_delta = statistics.fmean(values)
    std_delta = statistics.pstdev(values)
    worst_seed = max(values)
    active = str(candidate["family"]) != PHASE_POLICY
    safety_ok = all(
        int(row["method_safety"]["ending_teleports"])
        <= int(row["phase_safety"]["ending_teleports"])
        and int(row["method_safety"]["collision_incidents"])
        <= int(row["phase_safety"]["collision_incidents"])
        for row in selected
    )
    minimum_interventions = min(
        int(row["executed_overrides"]) for row in selected
    )
    feasible = bool(
        active
        and mean_delta
        <= -float(selection_rule["minimum_city_mean_relative_improvement"])
        and worst_seed
        <= float(
            selection_rule["maximum_seed_level_city_mean_relative_regression"]
        )
        and safety_ok
        and minimum_interventions
        >= int(selection_rule["minimum_executed_interventions_per_scenario_seed"])
    )
    return {
        **dict(candidate),
        "paired_units": len(selected),
        "seed_level_city_mean_relative_delta": seed_deltas,
        "city_mean_relative_delta": mean_delta,
        "seed_delta_std": std_delta,
        "worst_seed_relative_delta": worst_seed,
        "robust_score": mean_delta + 0.5 * std_delta + max(worst_seed, 0.0),
        "total_executed_overrides": sum(
            int(row["executed_overrides"]) for row in selected
        ),
        "minimum_executed_overrides_per_scenario_seed": minimum_interventions,
        "safety_noninferior": safety_ok,
        "feasible": feasible,
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    blend = row.get("blend_weight")
    return (
        float(row["robust_score"]),
        float(row["city_mean_relative_delta"]),
        int(row["total_executed_overrides"]),
        float(blend) if blend is not None else float("inf"),
        -int(row["cooldown_intervals"]),
        str(row["key"]),
    )


def audit_policy_horizon_adaptation(
    *,
    results_root: Path,
    launch_manifest: Path,
    policy_protocol_path: Path,
    failed_v47_audit_path: Path,
    pressure_joint_audit_path: Path,
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    protocol = _read_json(policy_protocol_path)
    failed = _read_json(failed_v47_audit_path)
    joint = _read_json(pressure_joint_audit_path)
    identities = all_rollout_identities(protocol)
    expected = {(scenario, seed, candidate) for _, scenario, seed, candidate in identities}
    identity_index = {
        (scenario, seed, candidate): index
        for index, (_, scenario, seed, candidate) in enumerate(identities)
    }
    specs = tuple(launch.get("specs", ()))
    spec_by_shard = {
        int(str(spec["signature"]).rsplit("-", 1)[-1]): spec for spec in specs
    }
    launch_gate = {
        "launch_protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "result_protocol_matches": launch.get("result_protocol") == RESULT_PROTOCOL,
        "policy_protocol_matches": protocol.get("protocol") == PROTOCOL,
        "failed_v47_evidence_matches": failed.get("status") == "PASS"
        and failed.get("decision") == FAILED_V47_DECISION,
        "failed_v47_hash_matches": _sha256(failed_v47_audit_path)
        == str(protocol["failed_v47_falsification_audit"]["sha256"]),
        "joint_hash_matches": _sha256(pressure_joint_audit_path)
        == str(protocol["model_artifacts"]["pressure_joint_audit_sha256"]),
        "submitted": bool(launch.get("submitted", False)),
        "scheduler_returncode_zero": int(launch.get("scheduler_returncode", -1))
        == 0,
        "six_unique_node_shards": set(spec_by_shard) == set(range(len(NODES)))
        and {str(spec["require_node"]) for spec in specs} == set(NODES),
        "matrix_size_matches": int(launch.get("matrix_size", -1)) == len(expected),
    }
    launch_gate["passed"] = all(launch_gate.values())

    shard_gate_rows = []
    shard_row_identities: set[tuple[str, int, str]] = set()
    for shard_index in range(len(NODES)):
        path = Path(results_root) / "_shards" / f"shard_{shard_index}.json"
        if not path.is_file():
            shard_gate_rows.append(
                {"shard_index": shard_index, "path": str(path), "passed": False}
            )
            continue
        shard = _read_json(path)
        spec = spec_by_shard[shard_index]
        rows = tuple(shard.get("rows", ()))
        observed = {
            (str(row["scenario"]), int(row["seed"]), str(row["candidate"]))
            for row in rows
        }
        assigned = {
            key
            for key, index in identity_index.items()
            if index % len(NODES) == shard_index
        }
        passed = bool(
            shard.get("protocol") == SHARD_PROTOCOL
            and shard.get("passed") is True
            and shard.get("hostname") == spec.get("require_node")
            and int(shard.get("shard_count", -1)) == len(NODES)
            and int(shard.get("task_count", -1)) == len(assigned)
            and observed == assigned
        )
        shard_gate_rows.append(
            {
                "shard_index": shard_index,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "hostname": shard.get("hostname"),
                "task_count": len(rows),
                "passed": passed,
            }
        )
        shard_row_identities.update(observed)

    errors: list[str] = []
    rows = []
    policy_sha = _sha256(policy_protocol_path)
    failed_sha = _sha256(failed_v47_audit_path)
    joint_sha = _sha256(pressure_joint_audit_path)
    for city, scenario, seed, candidate in identities:
        root = Path(results_root) / scenario / f"seed_{seed}" / candidate
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        if not result_path.is_file() or not tripinfo_path.is_file():
            errors.append(f"missing v48 result: {scenario}/{seed}/{candidate}")
            continue
        result = _read_json(result_path)
        metrics = dict(result.get("metrics", {}))
        shard_index = identity_index[(scenario, seed, candidate)] % len(NODES)
        expected_host = str(spec_by_shard[shard_index]["require_node"])
        tripinfo_sha = _sha256(tripinfo_path)
        row_gate = {
            "protocol_matches": result.get("protocol") == RESULT_PROTOCOL,
            "stage_matches": result.get("analysis_stage")
            == "policy_horizon_adaptation",
            "identity_matches": (
                result.get("city"),
                result.get("scenario"),
                int(result.get("seed", -1)),
                result.get("policy"),
            )
            == (city, scenario, seed, candidate),
            "physical_host_matches_shard": result.get("hostname") == expected_host,
            "source_tree_matches_snapshot": result.get("runtime", {}).get(
                "source_tree_sha256"
            )
            == launch.get("source_tree_sha256"),
            "policy_protocol_hash_matches": result.get(
                "policy_horizon_protocol_sha256"
            )
            == policy_sha,
            "failed_v47_hash_matches": result.get(
                "failed_v47_falsification_audit_sha256"
            )
            == failed_sha,
            "joint_hash_matches": result.get(
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
            "selection_eligible": bool(result.get("selection_eligible_evidence")),
            "not_prospective": result.get("prospective_evidence") is False,
        }
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"v48 row gate failed: {scenario}/{seed}/{candidate}")
        rows.append(
            {
                "city": city,
                "scenario": scenario,
                "seed": seed,
                "candidate": candidate,
                "candidate_spec": result["policy_horizon_candidate"],
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
                    key: int(metrics.get(key, -1))
                    for key in (
                        "starting_teleports",
                        "ending_teleports",
                        "collision_events",
                        "collision_incidents",
                    )
                },
                "gate": row_gate,
            }
        )

    keyed = {(row["scenario"], row["seed"], row["candidate"]): row for row in rows}
    paired_rows = []
    for city, scenario, seed, candidate in identities:
        if candidate == PHASE_POLICY:
            continue
        method = keyed.get((scenario, seed, candidate))
        phase = keyed.get((scenario, seed, PHASE_POLICY))
        if method is None or phase is None:
            continue
        paired_rows.append(
            {
                "city": city,
                "scenario": scenario,
                "seed": seed,
                "candidate": candidate,
                "method_mean_waiting_time": method["mean_waiting_time"],
                "phase_mean_waiting_time": phase["mean_waiting_time"],
                "relative_delta": (
                    float(method["mean_waiting_time"])
                    - float(phase["mean_waiting_time"])
                )
                / max(float(phase["mean_waiting_time"]), 1e-12),
                "executed_overrides": method["executed_overrides"],
                "method_safety": method["safety"],
                "phase_safety": phase["safety"],
            }
        )

    adaptation = dict(protocol["policy_horizon_adaptation"])
    seeds = tuple(int(value) for value in adaptation["seeds"])
    selection_rule = dict(adaptation["selection_rule"])
    city_selection = {}
    for city, scenarios in protocol["city_scenarios"].items():
        candidates = [
            candidate.__dict__
            for candidate in policy_candidates(protocol, str(city))
            if candidate.key != PHASE_POLICY
        ]
        grid = [
            summarize_candidate(
                city=str(city),
                candidate=candidate,
                paired_rows=paired_rows,
                scenarios=tuple(str(value) for value in scenarios),
                seeds=seeds,
                selection_rule=selection_rule,
            )
            for candidate in candidates
        ]
        feasible = [row for row in grid if bool(row["feasible"])]
        selected = min(feasible, key=_selection_key) if feasible else {
            "key": PHASE_POLICY,
            "family": PHASE_POLICY,
            "blend_weight": None,
            "risk_multiplier": None,
            "min_context_trust": None,
            "cooldown_intervals": 0,
            "feasible": True,
            "fallback": True,
        }
        city_selection[str(city)] = {
            "decision": (
                "deploy_active_cfcmt_candidate"
                if feasible
                else "deploy_phase_pressure_fallback"
            ),
            "selected": selected,
            "feasible_active_candidate_count": len(feasible),
            "grid": grid,
        }

    robustness_seeds = {
        int(value)
        for value in protocol["untouched_robustness_development"]["seeds"]
    }
    prospective_seeds = {
        int(value)
        for value in protocol["prospective_confirmation_reservation"]["seeds"]
    }
    forbidden_paths = [
        str(path.resolve())
        for seed in sorted(robustness_seeds | prospective_seeds)
        for path in Path(results_root).rglob(f"seed_{seed}")
    ]
    integrity_gate = {
        "launch_gate_passed": bool(launch_gate["passed"]),
        "all_shards_passed": len(shard_gate_rows) == len(NODES)
        and all(row["passed"] for row in shard_gate_rows),
        "shard_rows_cover_matrix": shard_row_identities == expected,
        "all_expected_results_present": len(rows) == len(expected),
        "all_result_gates_passed": bool(rows)
        and all(row["gate"]["passed"] for row in rows),
        "all_pairs_present": len(paired_rows)
        == len(expected)
        - sum(len(values) for values in protocol["city_scenarios"].values())
        * len(seeds),
        "forbidden_seed_results_absent": not forbidden_paths,
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    active_cities = [
        city
        for city, value in city_selection.items()
        if value["decision"] == "deploy_active_cfcmt_candidate"
    ]
    advance_gate = {
        "integrity_passed": bool(integrity_gate["passed"]),
        "at_least_one_city_selects_active_cfcmt": bool(active_cities),
        "every_city_has_frozen_deployment": len(city_selection)
        == len(protocol["city_scenarios"]),
    }
    advance_gate["passed"] = all(advance_gate.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": ADVANCE_DECISION if advance_gate["passed"] else REJECTION_DECISION,
        "claim_boundary": (
            "This artifact selects deployment policies from target adaptation "
            "seeds 5057/6067 only. It is not robustness or prospective evidence."
        ),
        "launch_manifest_sha256": _sha256(launch_manifest),
        "policy_protocol_sha256": policy_sha,
        "failed_v47_audit_sha256": failed_sha,
        "pressure_joint_audit_sha256": joint_sha,
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "launch_gate": launch_gate,
        "shards": shard_gate_rows,
        "rows": rows,
        "paired_rows": paired_rows,
        "city_selection": city_selection,
        "active_cities": active_cities,
        "integrity_gate": integrity_gate,
        "advance_gate": advance_gate,
        "untouched_robustness_seeds_remain_unrun": sorted(robustness_seeds),
        "prospective_confirmation_seeds_remain_unrun": sorted(prospective_seeds),
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--policy-protocol", type=Path, required=True)
    parser.add_argument("--failed-v47-audit", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v48 selection audit: {args.out}")
    payload = audit_policy_horizon_adaptation(
        results_root=args.results_root,
        launch_manifest=args.launch_manifest,
        policy_protocol_path=args.policy_protocol,
        failed_v47_audit_path=args.failed_v47_audit,
        pressure_joint_audit_path=args.pressure_joint_audit,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "active_cities": payload["active_cities"],
                "selected": {
                    city: value["selected"]["key"]
                    for city, value in payload["city_selection"].items()
                },
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
