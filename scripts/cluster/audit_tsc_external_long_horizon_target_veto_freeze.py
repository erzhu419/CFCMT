#!/usr/bin/env python3
"""Jointly audit the v66 OOF target-veto model artifacts."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION as TRAINING_DECISION,
    MODEL_PROTOCOL,
    RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.long_horizon_target_veto import (  # noqa: E402
    LongHorizonTargetVeto,
)
from scripts.cluster.audit_tsc_external_long_horizon_target_veto_cache import (  # noqa: E402
    AUDIT_DECISION as CACHE_AUDIT_DECISION,
    AUDIT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_training import (  # noqa: E402
    PROTOCOL as TRAINING_PROTOCOL,
)
from scripts.cluster.run_tsc_external_long_horizon_target_veto_training import (  # noqa: E402
    RUN_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v66r62-external-v9-long-horizon-target-veto-joint-audit-v1"
AUDIT_DECISION = "authorize_frozen_target_veto_closed_loop_development_only"


def audit_target_veto_freeze(
    *,
    training_result_path: Path,
    training_protocol_path: Path,
    cache_audit_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    result = json.loads(Path(training_result_path).read_text(encoding="utf-8"))
    training = json.loads(Path(training_protocol_path).read_text(encoding="utf-8"))
    cache_audit = json.loads(Path(cache_audit_path).read_text(encoding="utf-8"))
    errors = []
    top_gate = {
        "result_protocol_matches": result.get("protocol") == RESULT_PROTOCOL,
        "run_protocol_matches": result.get("run_protocol") == RUN_PROTOCOL,
        "training_status_passed": result.get("status") == "PASS"
        and result.get("decision") == TRAINING_DECISION
        and bool(result.get("integrity_gate", {}).get("passed", False)),
        "training_protocol_matches": training.get("protocol") == TRAINING_PROTOCOL
        and result.get("executable_training_freeze", {}).get(
            "training_protocol_sha256"
        )
        == _sha256(training_protocol_path),
        "executable_source_gate_passed": bool(
            result.get("executable_training_freeze", {})
            .get("gate", {})
            .get("passed", False)
        ),
        "cache_audit_matches": cache_audit.get("protocol") == CACHE_AUDIT_PROTOCOL
        and cache_audit.get("status") == "PASS"
        and cache_audit.get("decision") == CACHE_AUDIT_DECISION
        and result.get("cache_audit_sha256") == _sha256(cache_audit_path),
        "cache_identity_matches": result.get("cache_sha256")
        == cache_audit.get("cache_sha256")
        and int(result.get("cache_file_count", -1)) == 384,
    }
    city_rows = {}
    declared = dict(result.get("city_artifacts", {}))
    expected_cities = set(result.get("city_results", {}))
    for city in sorted(expected_cities):
        declared_city = dict(declared.get(city, {}))
        model_path = Path(artifact_root) / city / "model.pkl"
        certificate_path = Path(artifact_root) / city / "freeze.json"
        try:
            certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
            artifact = pickle.loads(model_path.read_bytes())
        except Exception as exc:
            errors.append(f"cannot load {city} target-veto artifact: {exc}")
            continue
        oof_folds = tuple(certificate.get("oof_folds", ()))
        oof_records = tuple(certificate.get("oof_records", ()))
        groups = [str(row.get("group_id")) for row in oof_records]
        selected = certificate.get("gate_selection", {}).get("selected")
        grid = tuple(certificate.get("gate_selection", {}).get("grid", ()))
        target_veto = artifact.get("target_veto")
        city_gate = {
            "declared_hashes_match": _sha256(model_path)
            == declared_city.get("model", {}).get("sha256")
            and _sha256(certificate_path)
            == declared_city.get("certificate", {}).get("sha256"),
            "certificate_matches": certificate.get("protocol") == RESULT_PROTOCOL
            and certificate.get("city") == city,
            "model_contract_matches": artifact.get("protocol") == MODEL_PROTOCOL
            and artifact.get("city") == city
            and isinstance(target_veto, LongHorizonTargetVeto),
            "few_shot_claim_matches": artifact.get("classification")
            == "target_simulator_labeled_few_shot_adaptation"
            and artifact.get("zero_shot") is False,
            "veto_only_contract_matches": artifact.get("veto_role")
            == "veto_only_never_action_originator",
            "six_oof_folds": len(oof_folds) == 6
            and len({int(row.get("heldout_seed", -1)) for row in oof_folds}) == 6,
            "oof_group_partition_exact": bool(groups)
            and len(groups) == len(set(groups))
            and len(groups) == int(certificate.get("action_group_count", -1)),
            "gate_grid_exact": len(grid) == 20
            and len({str(row.get("key")) for row in grid}) == 20,
            "selected_gate_consistent": (
                selected is not None
                and bool(declared_city.get("active"))
                and bool(artifact["target_veto"].config.enabled)
                and artifact.get("selected_gate", {}).get("selected", {}).get("key")
                == selected.get("key")
            )
            or (
                selected is None
                and not bool(declared_city.get("active"))
                and not bool(artifact["target_veto"].config.enabled)
            ),
        }
        city_gate["passed"] = all(city_gate.values())
        if not city_gate["passed"]:
            errors.append(f"target-veto city gate failed: {city}")
        city_rows[city] = {
            "model": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
            "certificate": {
                "path": str(certificate_path.resolve()),
                "sha256": _sha256(certificate_path),
            },
            "active": bool(declared_city.get("active")),
            "selected_gate": selected,
            "oof_group_count": len(groups),
            "gate": city_gate,
        }
    integrity_gate = {
        "top_gate_passed": all(top_gate.values()),
        "all_expected_cities_present": set(city_rows) == expected_cities
        and expected_cities == {"jinan", "los_angeles"},
        "all_city_gates_passed": bool(city_rows)
        and all(row["gate"]["passed"] for row in city_rows.values()),
        "at_least_one_active_city": any(row["active"] for row in city_rows.values()),
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": AUDIT_DECISION if integrity_gate["passed"] else "reject",
        "claim_boundary": (
            "OOF model identity and veto-only integrity. Closed-loop efficacy is "
            "not evaluated and validation/prospective seeds remain sealed."
        ),
        "training_result_sha256": _sha256(training_result_path),
        "training_protocol_sha256": _sha256(training_protocol_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "top_gate": top_gate,
        "cities": city_rows,
        "integrity_gate": integrity_gate,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite target-veto joint audit: {args.out}")
    payload = audit_target_veto_freeze(
        training_result_path=args.training_result,
        training_protocol_path=args.training_protocol,
        cache_audit_path=args.cache_audit,
        artifact_root=args.artifact_root,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
