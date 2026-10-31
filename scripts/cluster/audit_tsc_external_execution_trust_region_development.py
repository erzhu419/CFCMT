#!/usr/bin/env python3
"""Audit v51 development and freeze or reject its deployment selector."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_execution_trust_region_development import (  # noqa: E402
    FAILED_V50_DECISION,
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
    trust_region_candidates,
)
from cf_h2o.eval.traffic_signal_external_policy_horizon_adaptation import (  # noqa: E402
    PHASE_POLICY,
    policy_candidates,
)
from scripts.cluster.audit_tsc_external_policy_horizon_expanded_development import (  # noqa: E402
    AUDIT_PROTOCOL as V50_AUDIT_PROTOCOL,
    REJECTION_DECISION as V50_REJECTION_DECISION,
    select_city,
)
from scripts.cluster.launch_tsc_external_execution_trust_region_development import (  # noqa: E402
    LAUNCH_PROTOCOL,
    NODES,
)
from scripts.cluster.run_tsc_external_execution_trust_region_development_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v51r47-execution-trust-region-development-selection-audit-v1"
ADVANCE_DECISION = "authorize_frozen_v51_selector_for_new_untouched_validation"
REJECTION_DECISION = "reject_v51_and_preserve_new_validation_and_prospective_seeds"
PREVIEW_DECISION = "preview_only_no_validation_authorization"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _phase_references(
    *,
    failed_v50: Mapping[str, Any],
    city: str,
    scenario: str,
    seeds: Sequence[int],
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    expected_seeds = {int(value) for value in seeds}
    references: dict[int, dict[str, Any]] = {}
    errors = []
    for row in failed_v50.get("paired_rows", ()):
        if str(row.get("city")) != city or str(row.get("scenario")) != scenario:
            continue
        seed = int(row["seed"])
        if seed not in expected_seeds:
            continue
        candidate = {
            "mean_waiting_time": float(row["phase_mean_waiting_time"]),
            "safety": dict(row["phase_safety"]),
        }
        if seed in references and references[seed] != candidate:
            errors.append(f"inconsistent v50 phase reference for seed {seed}")
        references[seed] = candidate
    missing = sorted(expected_seeds.difference(references))
    if missing:
        errors.append(f"missing v50 phase references: {missing}")
    return references, errors


def _feasibility_components(
    summary: Mapping[str, Any], selection_rule: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "mean_improvement": float(summary["city_mean_relative_delta"])
        <= -float(selection_rule["minimum_city_mean_relative_improvement"]),
        "bootstrap_upper_bound": float(
            summary["bootstrap_mean_relative_delta"]["upper"]
        )
        <= float(
            selection_rule["maximum_bootstrap_95pct_upper_mean_relative_delta"]
        ),
        "worst_seed_regression": float(summary["worst_seed_relative_delta"])
        <= float(selection_rule["maximum_seed_level_city_mean_relative_regression"]),
        "improved_seed_fraction": float(summary["improved_seed_fraction"])
        >= float(selection_rule["minimum_improved_seed_fraction"]),
        "intervened_seed_fraction": float(summary["intervened_seed_fraction"])
        >= float(selection_rule["minimum_intervened_seed_fraction"]),
        "safety_noninferiority": bool(summary["safety_noninferior"]),
    }


def audit_execution_trust_region_development(
    *,
    results_root: Path,
    launch_manifest: Path,
    trust_protocol_path: Path,
    expanded_protocol_path: Path,
    adaptation_protocol_path: Path,
    robustness_protocol_path: Path,
    pressure_protocol_path: Path,
    failed_v50_audit_path: Path,
    pressure_joint_audit_path: Path,
    preview: bool,
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    protocol = _read_json(trust_protocol_path)
    expanded = _read_json(expanded_protocol_path)
    adaptation = _read_json(adaptation_protocol_path)
    robustness = _read_json(robustness_protocol_path)
    pressure = _read_json(pressure_protocol_path)
    failed_v50 = _read_json(failed_v50_audit_path)
    identities = all_rollout_identities(protocol, adaptation)
    expected = {
        (str(scenario), int(seed), str(candidate))
        for _, scenario, seed, candidate in identities
    }
    identity_index = {
        (str(scenario), int(seed), str(candidate)): index
        for index, (_, scenario, seed, candidate) in enumerate(identities)
    }
    hashes = {
        "trust_protocol": _sha256(trust_protocol_path),
        "expanded_protocol": _sha256(expanded_protocol_path),
        "adaptation_protocol": _sha256(adaptation_protocol_path),
        "robustness_protocol": _sha256(robustness_protocol_path),
        "pressure_protocol": _sha256(pressure_protocol_path),
        "failed_v50_audit": _sha256(failed_v50_audit_path),
        "pressure_joint_audit": _sha256(pressure_joint_audit_path),
    }
    specs = tuple(launch.get("specs", ()))
    spec_by_shard = {
        int(str(spec["signature"]).rsplit("-", 1)[-1]): spec for spec in specs
    }
    validation_seeds = {
        int(value) for value in protocol["new_untouched_validation"]["seeds"]
    }
    prospective_seeds = {
        int(value)
        for value in protocol["prospective_confirmation_reservation"]["seeds"]
    }
    command_text = "\n".join(str(spec.get("cmd", "")) for spec in specs)
    launch_hashes = dict(launch.get("input_hashes", {}))
    launch_gate = {
        "launch_protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "result_protocol_matches": launch.get("result_protocol") == RESULT_PROTOCOL,
        "trust_protocol_matches": protocol.get("protocol") == PROTOCOL,
        "trust_protocol_hash_matches": launch_hashes.get("trust_protocol")
        == hashes["trust_protocol"],
        "failed_v50_audit_matches": failed_v50.get("protocol")
        == V50_AUDIT_PROTOCOL
        and failed_v50.get("status") == "PASS"
        and failed_v50.get("decision")
        in {FAILED_V50_DECISION, V50_REJECTION_DECISION}
        and hashes["failed_v50_audit"]
        == str(protocol["failed_v50_selection_audit"]["sha256"])
        == str(launch_hashes.get("failed_v50_audit")),
        "expanded_protocol_hash_matches": hashes["expanded_protocol"]
        == str(protocol["parent_expanded_protocol"]["sha256"])
        == str(launch_hashes.get("expanded_protocol")),
        "adaptation_protocol_hash_matches": hashes["adaptation_protocol"]
        == str(protocol["candidate_grid_protocol"]["sha256"])
        == str(launch_hashes.get("adaptation_protocol")),
        "robustness_protocol_hash_matches": hashes["robustness_protocol"]
        == str(expanded["parent_robustness_protocol"]["sha256"])
        == str(launch_hashes.get("robustness_protocol")),
        "pressure_protocol_hash_matches": hashes["pressure_protocol"]
        == str(adaptation["parent_pressure_protocol"]["sha256"])
        == str(launch_hashes.get("pressure_protocol")),
        "pressure_joint_hash_matches": hashes["pressure_joint_audit"]
        == str(adaptation["model_artifacts"]["pressure_joint_audit_sha256"])
        == str(launch_hashes.get("pressure_joint_audit")),
        "submitted": bool(launch.get("submitted", False)),
        "scheduler_returncode_zero": int(launch.get("scheduler_returncode", -1))
        == 0,
        "six_unique_node_shards": set(spec_by_shard) == set(range(len(NODES)))
        and {str(spec["require_node"]) for spec in specs} == set(NODES),
        "matrix_size_matches": int(launch.get("matrix_size", -1)) == len(expected),
        "sealed_seed_literals_absent_from_commands": not any(
            str(seed) in command_text for seed in validation_seeds | prospective_seeds
        ),
    }
    launch_gate["passed"] = all(launch_gate.values())

    shard_summaries = []
    shard_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for shard_index in range(len(NODES)):
        path = Path(results_root) / "_shards" / f"shard_{shard_index}.json"
        if not path.is_file():
            shard_summaries.append({"shard_index": shard_index, "passed": False})
            continue
        shard = _read_json(path)
        assigned = {
            key
            for key, index in identity_index.items()
            if index % len(NODES) == shard_index
        }
        rows = tuple(shard.get("rows", ()))
        observed = {
            (str(row["scenario"]), int(row["seed"]), str(row["candidate"]))
            for row in rows
        }
        passed = bool(
            shard.get("protocol") == SHARD_PROTOCOL
            and shard.get("passed") is True
            and shard.get("hostname") == spec_by_shard[shard_index]["require_node"]
            and int(shard.get("shard_count", -1)) == len(NODES)
            and int(shard.get("task_count", -1)) == len(assigned)
            and shard.get("trust_protocol_sha256") == hashes["trust_protocol"]
            and shard.get("expanded_protocol_sha256") == hashes["expanded_protocol"]
            and shard.get("adaptation_protocol_sha256")
            == hashes["adaptation_protocol"]
            and shard.get("robustness_protocol_sha256")
            == hashes["robustness_protocol"]
            and shard.get("failed_v50_audit_sha256") == hashes["failed_v50_audit"]
            and shard.get("pressure_joint_audit_sha256")
            == hashes["pressure_joint_audit"]
            and observed == assigned
        )
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "hostname": shard.get("hostname"),
                "task_count": len(rows),
                "elapsed_sec": shard.get("elapsed_sec"),
                "passed": passed,
            }
        )
        for row in rows:
            key = (str(row["scenario"]), int(row["seed"]), str(row["candidate"]))
            if key in shard_rows:
                raise ValueError(f"duplicate v51 shard identity: {key}")
            shard_rows[key] = dict(row)

    grid = dict(protocol["development_grid"])
    city = str(grid["city"])
    scenario = str(grid["scenario"])
    seeds = tuple(int(value) for value in grid["seeds"])
    phase_rows, errors = _phase_references(
        failed_v50=failed_v50,
        city=city,
        scenario=scenario,
        seeds=seeds,
    )
    base_candidates = {
        candidate.key: candidate for candidate in policy_candidates(adaptation, city)
    }
    candidate_specs = {}
    for candidate in trust_region_candidates(protocol, adaptation):
        candidate_specs[candidate.key] = {
            **asdict(candidate),
            "execution_trust_region": asdict(candidate.execution_trust_region),
            "base_candidate": asdict(base_candidates[candidate.base_candidate_key]),
        }

    result_rows = []
    paired_rows = []
    guard_totals: dict[str, Counter[str]] = {
        key: Counter() for key in candidate_specs
    }
    for _, identity_scenario, seed, candidate in identities:
        root = (
            Path(results_root)
            / identity_scenario
            / f"seed_{seed}"
            / candidate
        )
        result_path = root / "result.json"
        shard_row = shard_rows.get((identity_scenario, seed, candidate), {})
        if not result_path.is_file():
            errors.append(f"missing v51 result: {identity_scenario}/{seed}/{candidate}")
            continue
        result = _read_json(result_path)
        metrics = dict(result.get("metrics", {}))
        guard = dict(metrics.get("guard_audit", {}))
        for key, value in guard.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                guard_totals[candidate][key] += value
        result_sha = _sha256(result_path)
        tripinfo_evidence = dict(result.get("tripinfo_evidence", {}))
        expected_host = str(
            spec_by_shard[
                identity_index[(identity_scenario, seed, candidate)] % len(NODES)
            ]["require_node"]
        )
        expected_relative = (
            f"/{identity_scenario}/seed_{seed}/{candidate}/result.json"
        )
        row_gate = {
            "protocol_matches": result.get("protocol") == RESULT_PROTOCOL,
            "stage_matches": result.get("analysis_stage")
            == "execution_trust_region_development",
            "identity_matches": (
                result.get("city"),
                result.get("scenario"),
                int(result.get("seed", -1)),
                result.get("policy"),
            )
            == (city, identity_scenario, seed, candidate),
            "candidate_spec_matches": result.get(
                "execution_trust_region_candidate"
            )
            == candidate_specs[candidate],
            "physical_host_matches_shard": result.get("hostname") == expected_host,
            "source_tree_matches_snapshot": result.get("runtime", {}).get(
                "source_tree_sha256"
            )
            == launch.get("source_tree_sha256"),
            "trust_protocol_hash_matches": result.get(
                "execution_trust_region_protocol_sha256"
            )
            == hashes["trust_protocol"],
            "expanded_protocol_hash_matches": result.get(
                "parent_expanded_protocol_sha256"
            )
            == hashes["expanded_protocol"],
            "candidate_grid_hash_matches": result.get(
                "candidate_grid_protocol_sha256"
            )
            == hashes["adaptation_protocol"],
            "failed_v50_hash_matches": result.get(
                "failed_v50_selection_audit_sha256"
            )
            == hashes["failed_v50_audit"],
            "joint_hash_matches": result.get(
                "pressure_regularized_joint_audit_sha256"
            )
            == hashes["pressure_joint_audit"],
            "certificate_hash_matches": result.get(
                "pressure_regularized_certificate_sha256"
            )
            == pressure["city_artifacts"][city]["certificate"]["sha256"],
            "model_hash_matches": result.get("pressure_regularized_model_sha256")
            == pressure["city_artifacts"][city]["model"]["sha256"],
            "result_hash_matches_shard": shard_row.get("result_sha256")
            == result_sha,
            "tripinfo_hash_matches_shard": shard_row.get("tripinfo_sha256")
            == tripinfo_evidence.get("sha256"),
            "remote_result_path_matches": str(shard_row.get("result_path", "")).endswith(
                expected_relative
            ),
            "sumo_succeeded": bool(metrics.get("ok", False)),
            "selection_eligible": bool(result.get("selection_eligible_evidence")),
            "not_untouched_validation": result.get("untouched_validation_evidence")
            is False,
            "not_prospective": result.get("prospective_evidence") is False,
        }
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"v51 row gate failed: {identity_scenario}/{seed}/{candidate}")
        safety = {
            key: int(metrics.get(key, -1))
            for key in (
                "starting_teleports",
                "ending_teleports",
                "collision_events",
                "collision_incidents",
            )
        }
        overrides = int(
            guard.get("executed_overrides", guard.get("accepted_overrides", 0))
        )
        row = {
            "city": city,
            "scenario": identity_scenario,
            "seed": seed,
            "candidate": candidate,
            "candidate_spec": candidate_specs[candidate],
            "hostname": result.get("hostname"),
            "result_path": str(result_path.resolve()),
            "result_sha256": result_sha,
            "remote_tripinfo_path": shard_row.get("tripinfo_path"),
            "tripinfo_sha256": tripinfo_evidence.get("sha256"),
            "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
            "mean_travel_time": float(metrics["mean_tripinfo_duration"]),
            "executed_overrides": overrides,
            "safety": safety,
            "guard_audit": guard,
            "gate": row_gate,
        }
        result_rows.append(row)
        phase = phase_rows.get(seed)
        if phase is not None:
            paired_rows.append(
                {
                    "city": city,
                    "scenario": identity_scenario,
                    "seed": seed,
                    "candidate": candidate,
                    "method_mean_waiting_time": row["mean_waiting_time"],
                    "phase_mean_waiting_time": phase["mean_waiting_time"],
                    "relative_delta": (
                        row["mean_waiting_time"] - phase["mean_waiting_time"]
                    )
                    / max(phase["mean_waiting_time"], 1e-12),
                    "executed_overrides": overrides,
                    "method_safety": safety,
                    "phase_safety": phase["safety"],
                    "method_evidence_source": "v51_remote_tripinfo_hash_bound_result",
                    "phase_evidence_source": "hash_frozen_v50_phase_reference",
                }
            )

    candidates = [asdict(value) for value in trust_region_candidates(protocol, adaptation)]
    selection_rule = dict(grid["selection_rule"])
    bootstrap = dict(grid["bootstrap"])
    la_selection = select_city(
        city=city,
        candidates=candidates,
        paired_rows=paired_rows,
        scenarios=(scenario,),
        seeds=seeds,
        selection_rule=selection_rule,
        bootstrap=bootstrap,
    )
    failed_gate_counts: Counter[str] = Counter()
    for summary in la_selection["grid"]:
        components = _feasibility_components(summary, selection_rule)
        summary["feasibility_components"] = components
        summary["failed_feasibility_components"] = [
            key for key, passed in components.items() if not passed
        ]
        for key, passed in components.items():
            if not passed:
                failed_gate_counts[key] += 1
    city_selection = {
        city: la_selection,
        "jinan": {
            "decision": "deploy_phase_pressure_fallback",
            "selected": {
                "key": PHASE_POLICY,
                "family": PHASE_POLICY,
                "fallback": True,
                "feasible": True,
            },
            "feasible_active_candidate_count": 0,
            "grid": [],
            "frozen_without_v51_rollouts": True,
        },
    }
    active_cities = (
        [city]
        if la_selection["decision"] == "deploy_active_cfcmt_candidate"
        else []
    )

    selected_tripinfo = []
    selected_tripinfo_ok = True
    if active_cities:
        selected_key = str(la_selection["selected"]["key"])
        rows_by_seed = {
            int(row["seed"]): row
            for row in result_rows
            if row["candidate"] == selected_key
        }
        for seed in seeds:
            row = rows_by_seed[seed]
            path = (
                Path(results_root)
                / scenario
                / f"seed_{seed}"
                / selected_key
                / "tripinfo.xml"
            )
            exists = path.is_file()
            digest = _sha256(path) if exists else None
            passed = exists and digest == row["tripinfo_sha256"]
            selected_tripinfo_ok = selected_tripinfo_ok and passed
            selected_tripinfo.append(
                {
                    "city": city,
                    "scenario": scenario,
                    "seed": seed,
                    "candidate": selected_key,
                    "path": str(path.resolve()),
                    "sha256": digest,
                    "expected_sha256": row["tripinfo_sha256"],
                    "passed": passed,
                }
            )

    forbidden_paths = [
        str(path.resolve())
        for seed in sorted(validation_seeds | prospective_seeds)
        for path in Path(results_root).rglob(f"seed_{seed}")
    ]
    observed_result_paths = set(Path(results_root).rglob("result.json"))
    core_integrity_gate = {
        "launch_gate_passed": bool(launch_gate["passed"]),
        "all_shards_passed": len(shard_summaries) == len(NODES)
        and all(row["passed"] for row in shard_summaries),
        "shard_rows_cover_matrix": set(shard_rows) == expected,
        "all_results_present_without_extras": len(observed_result_paths)
        == len(expected)
        == len(result_rows),
        "all_result_gates_passed": bool(result_rows)
        and all(row["gate"]["passed"] for row in result_rows),
        "all_phase_references_present": len(phase_rows) == len(seeds),
        "all_pairs_present": len(paired_rows) == len(expected),
        "sealed_seed_results_absent": not forbidden_paths,
        "no_errors": not errors,
    }
    core_integrity_gate["passed"] = all(core_integrity_gate.values())
    integrity_gate = {
        **core_integrity_gate,
        "selected_tripinfo_locally_verified": selected_tripinfo_ok,
    }
    integrity_gate["passed"] = bool(core_integrity_gate["passed"]) and (
        preview or selected_tripinfo_ok
    )
    advance_gate = {
        "final_not_preview": not preview,
        "integrity_passed": bool(integrity_gate["passed"]),
        "los_angeles_selects_active_cfcmt": bool(active_cities),
        "jinan_has_frozen_fallback": city_selection["jinan"]["decision"]
        == "deploy_phase_pressure_fallback",
    }
    advance_gate["passed"] = all(advance_gate.values())
    if preview:
        status = "PREVIEW"
        decision = PREVIEW_DECISION
    else:
        status = "PASS" if integrity_gate["passed"] else "FAIL"
        decision = ADVANCE_DECISION if advance_gate["passed"] else REJECTION_DECISION

    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": decision,
        "claim_boundary": (
            "Post-v50 method development over the same ten disclosed development "
            "seeds. It is neither untouched validation nor prospective confirmation."
        ),
        "preview": preview,
        "launch_manifest_sha256": _sha256(launch_manifest),
        **{f"{key}_sha256": value for key, value in hashes.items()},
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "launch_gate": launch_gate,
        "shards": shard_summaries,
        "result_rows": result_rows,
        "paired_rows": paired_rows,
        "phase_references": {str(key): value for key, value in phase_rows.items()},
        "city_selection": city_selection,
        "active_cities": active_cities,
        "candidate_guard_totals": {
            key: dict(sorted(counter.items())) for key, counter in guard_totals.items()
        },
        "failed_feasibility_gate_counts": dict(sorted(failed_gate_counts.items())),
        "selected_tripinfo": selected_tripinfo,
        "core_integrity_gate": core_integrity_gate,
        "integrity_gate": integrity_gate,
        "advance_gate": advance_gate,
        "new_untouched_validation_seeds_remain_unrun": sorted(validation_seeds),
        "prospective_confirmation_seeds_remain_unrun": sorted(prospective_seeds),
        "forbidden_paths": forbidden_paths,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--trust-protocol", type=Path, required=True)
    parser.add_argument("--expanded-protocol", type=Path, required=True)
    parser.add_argument("--adaptation-protocol", type=Path, required=True)
    parser.add_argument("--robustness-protocol", type=Path, required=True)
    parser.add_argument("--pressure-protocol", type=Path, required=True)
    parser.add_argument("--failed-v50-audit", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v51 selection audit: {args.out}")
    payload = audit_execution_trust_region_development(
        results_root=args.results_root,
        launch_manifest=args.launch_manifest,
        trust_protocol_path=args.trust_protocol,
        expanded_protocol_path=args.expanded_protocol,
        adaptation_protocol_path=args.adaptation_protocol,
        robustness_protocol_path=args.robustness_protocol,
        pressure_protocol_path=args.pressure_protocol,
        failed_v50_audit_path=args.failed_v50_audit,
        pressure_joint_audit_path=args.pressure_joint_audit,
        preview=bool(args.preview),
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
                "core_integrity": payload["core_integrity_gate"]["passed"],
                "selected_tripinfo_verified": payload["integrity_gate"][
                    "selected_tripinfo_locally_verified"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    if args.preview:
        return 0 if payload["core_integrity_gate"]["passed"] else 2
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
