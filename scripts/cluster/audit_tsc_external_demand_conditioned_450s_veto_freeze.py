#!/usr/bin/env python3
"""Audit v70 OOF partitions and frozen demand-conditioned veto artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_external_demand_conditioned_veto_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION,
    RESULT_PROTOCOL,
    TRAINING_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from cf_h2o.traffic_signal.demand_conditioned_target_veto import (  # noqa: E402
    ARTIFACT_PROTOCOL,
    DemandConditionedTargetVeto,
)


AUDIT_PROTOCOL = "tsc-v70r66-demand-conditioned-450s-veto-joint-audit-v1"
AUDIT_DECISION = AUTHORIZATION_DECISION


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _file_count(path: Path) -> int:
    return sum(1 for item in Path(path).rglob("*") if item.is_file()) if path.exists() else 0


def audit_freeze(
    *,
    training_protocol_path: Path,
    training_result_path: Path,
    artifact_root: Path,
    future_closed_loop_root: Path,
) -> dict[str, Any]:
    protocol = _read(training_protocol_path)
    result = _read(training_result_path)
    training = protocol["training"]
    expected_cities = set(training["city_scenarios"])
    seeds = {int(value) for value in training["seeds"]}
    source_gate = {
        path: Path(path).is_file() and _sha256(Path(path)) == expected
        for path, expected in protocol["runtime_executable_sources"].items()
    }
    top_gate = {
        "training_protocol_matches": protocol.get("protocol") == TRAINING_PROTOCOL,
        "training_result_passed": (
            result.get("protocol") == RESULT_PROTOCOL
            and result.get("status") == "PASS"
            and result.get("decision") == AUDIT_DECISION
            and bool(result.get("integrity_gate", {}).get("passed", False))
        ),
        "protocol_hash_bound": result.get("training_protocol_sha256")
        == _sha256(training_protocol_path),
        "source_hashes_match": bool(source_gate) and all(source_gate.values()),
        "cities_exact": set(result.get("city_results", {})) == expected_cities
        and set(result.get("city_artifacts", {})) == expected_cities,
        "artifact_file_count_exact": _file_count(artifact_root)
        == 2 * len(expected_cities),
        "closed_loop_results_absent": _file_count(future_closed_loop_root) == 0,
        "sealed_seed_sets_disjoint": not (
            seeds & {int(value) for value in protocol["validation"]["closed_loop_seeds"]}
        )
        and not (
            seeds
            & {
                int(value)
                for value in protocol["prospective_confirmation"]["closed_loop_seeds"]
            }
        ),
    }
    city_rows = {}
    errors = []
    for city in sorted(expected_cities):
        declared = result["city_artifacts"][city]
        certificate_path = Path(artifact_root) / city / "freeze.json"
        model_path = Path(artifact_root) / city / "model.pkl"
        certificate = _read(certificate_path) if certificate_path.is_file() else {}
        artifact = pickle.loads(model_path.read_bytes()) if model_path.is_file() else {}
        scenarios = {str(value) for value in training["city_scenarios"][city]}
        folds = certificate.get("oof_folds", ())
        fold_predictions = [
            row for fold in folds for row in fold.get("predictions", ())
        ]
        active = bool(certificate.get("final_config", {}).get("enabled", False))
        selected = certificate.get("gate_selection", {}).get("selected")
        scenario_vetoes = artifact.get("scenario_vetoes", {})
        city_gate = {
            "hashes_match": (
                certificate_path.is_file()
                and model_path.is_file()
                and _sha256(certificate_path) == declared["certificate"]["sha256"]
                and _sha256(model_path) == declared["model"]["sha256"]
            ),
            "certificate_contract": (
                certificate.get("protocol") == RESULT_PROTOCOL
                and certificate.get("city") == city
                and set(certificate.get("scenarios", ())) == scenarios
                and {int(value) for value in certificate.get("seeds", ())} == seeds
                and certificate.get("combined_feature_names")
                == training["combined_feature_names"]
                and certificate.get("demand_feature_names")
                == training["demand_feature_names"]
            ),
            "oof_partition_exact": (
                len(folds) == len(seeds)
                and {int(fold.get("heldout_seed", -1)) for fold in folds} == seeds
                and all(
                    int(fold["heldout_seed"])
                    not in {int(value) for value in fold.get("training_seeds", ())}
                    for fold in folds
                )
                and len(fold_predictions)
                == int(certificate.get("proposal_training_row_count", -1))
                and len({str(row.get("group_id")) for row in fold_predictions})
                == len(fold_predictions)
            ),
            "selection_contract": (
                active == bool(declared["active"])
                and ((selected is not None and bool(selected.get("feasible", False))) if active else selected is None)
            ),
            "artifact_contract": (
                artifact.get("protocol") == ARTIFACT_PROTOCOL
                and artifact.get("city") == city
                and artifact.get("zero_shot") is False
                and artifact.get("scenario_id_is_model_feature") is False
                and artifact.get("veto_role")
                == "demand_conditioned_veto_only_never_action_originator"
                and set(scenario_vetoes) == scenarios
                and all(
                    isinstance(veto, DemandConditionedTargetVeto)
                    and int(veto.config.rollout_value_horizon_sec) == 450
                    and bool(veto.config.enabled) == active
                    for veto in scenario_vetoes.values()
                )
            ),
            "demand_profiles_bound": all(
                scenario_vetoes[scenario].demand_profile.input_sha256
                == protocol["demand_schedule_context"]["profiles"][scenario][
                    "input_sha256"
                ]
                for scenario in scenarios
            )
            if set(scenario_vetoes) == scenarios
            else False,
        }
        city_gate["passed"] = all(city_gate.values())
        if not city_gate["passed"]:
            errors.append(f"v70 city gate failed: {city}")
        city_rows[city] = {
            "active": active,
            "scenarios": sorted(scenarios),
            "model": declared["model"],
            "certificate": declared["certificate"],
            "proposal_training_row_count": certificate.get(
                "proposal_training_row_count"
            ),
            "selected_gate": selected,
            "gate": city_gate,
        }
    active_cities = sorted(city for city, row in city_rows.items() if row["active"])
    integrity_gate = {
        **top_gate,
        "all_city_gates_passed": bool(city_rows)
        and all(row["gate"]["passed"] for row in city_rows.values()),
        "at_least_one_active_city": bool(active_cities),
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": AUDIT_DECISION if integrity_gate["passed"] else "reject",
        "claim_boundary": (
            "OOF and artifact integrity only. This audit authorizes development-seed "
            "closed-loop evaluation, not validation."
        ),
        "training_protocol_sha256": _sha256(training_protocol_path),
        "training_result_sha256": _sha256(training_result_path),
        "source_gate": source_gate,
        "active_cities": active_cities,
        "cities": city_rows,
        "integrity_gate": integrity_gate,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--future-closed-loop-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v70 joint audit: {args.out}")
    payload = audit_freeze(
        training_protocol_path=args.training_protocol,
        training_result_path=args.training_result,
        artifact_root=args.artifact_root,
        future_closed_loop_root=args.future_closed_loop_root,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "active_cities": payload["active_cities"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
