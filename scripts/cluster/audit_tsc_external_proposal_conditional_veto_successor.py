#!/usr/bin/env python3
"""Jointly audit the v67 proposal-conditional successor artifacts."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_external_proposal_conditional_veto_freeze import (  # noqa: E402
    ARTIFACT_PROTOCOL,
    AUTHORIZATION_DECISION as TRAINING_DECISION,
    RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (  # noqa: E402
    MODEL_PROTOCOL,
    ProposalConditionalRegressor,
    ProposalConditionalTargetVeto,
)
from scripts.cluster.freeze_tsc_external_v9_proposal_conditional_veto_successor import (  # noqa: E402
    PROTOCOL as SUCCESSOR_PROTOCOL,
)
from scripts.cluster.run_tsc_external_proposal_conditional_veto_successor import (  # noqa: E402
    RUN_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v67r63-proposal-conditional-veto-successor-joint-audit-v1"
AUDIT_DECISION = "authorize_frozen_proposal_conditional_closed_loop_development_only"


def audit_successor(
    *,
    training_result_path: Path,
    successor_protocol_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    result = json.loads(Path(training_result_path).read_text(encoding="utf-8"))
    protocol = json.loads(Path(successor_protocol_path).read_text(encoding="utf-8"))
    errors = []
    top_gate = {
        "result_protocol_matches": result.get("protocol") == RESULT_PROTOCOL,
        "run_protocol_matches": result.get("run_protocol") == RUN_PROTOCOL,
        "training_passed": result.get("status") == "PASS"
        and result.get("decision") == TRAINING_DECISION
        and bool(result.get("integrity_gate", {}).get("passed", False)),
        "successor_protocol_matches": protocol.get("protocol")
        == SUCCESSOR_PROTOCOL
        and result.get("successor_protocol_sha256")
        == _sha256(successor_protocol_path),
        "executable_freeze_passed": bool(
            result.get("executable_freeze", {}).get("gate", {}).get("passed", False)
        )
        and result.get("executable_freeze", {}).get("successor_protocol_sha256")
        == _sha256(successor_protocol_path),
        "failed_v66_identity_matches": result.get("failed_v66_result_sha256")
        == protocol["failed_v66_result"]["sha256"],
        "cache_audit_identity_matches": result.get("cache_audit_sha256")
        == protocol["cache_audit"]["sha256"],
        "post_failure_disclosure_preserved": protocol.get(
            "scientific_disclosure", {}
        ).get("successor_oof_is_exploratory_method_selection_not_confirmation")
        is True,
    }
    declared = dict(result.get("city_artifacts", {}))
    expected_cities = set(protocol["training"]["city_scenarios"])
    city_rows = {}
    for city in sorted(expected_cities):
        model_path = Path(artifact_root) / city / "model.pkl"
        certificate_path = Path(artifact_root) / city / "freeze.json"
        try:
            certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
            artifact = pickle.loads(model_path.read_bytes())
        except Exception as exc:
            errors.append(f"cannot load successor artifact {city}: {exc}")
            continue
        target_veto = artifact.get("proposal_conditional_veto")
        folds = tuple(certificate.get("oof_folds", ()))
        predictions = tuple(certificate.get("oof_predictions", ()))
        groups = [str(row.get("group_id")) for row in predictions]
        proposal_seeds = {
            int(row.get("simulator_seed", -1)) for row in predictions
        }
        selected = certificate.get("gate_selection", {}).get("selected")
        active = bool(declared.get(city, {}).get("active"))
        config = dict(certificate.get("final_config", {}))
        city_gate = {
            "declared_hashes_match": _sha256(model_path)
            == declared.get(city, {}).get("model", {}).get("sha256")
            and _sha256(certificate_path)
            == declared.get(city, {}).get("certificate", {}).get("sha256"),
            "certificate_contract_matches": certificate.get("protocol")
            == RESULT_PROTOCOL
            and certificate.get("city") == city,
            "artifact_contract_matches": artifact.get("protocol")
            == ARTIFACT_PROTOCOL
            and artifact.get("model_protocol") == MODEL_PROTOCOL
            and artifact.get("city") == city
            and isinstance(artifact.get("model"), ProposalConditionalRegressor)
            and isinstance(target_veto, ProposalConditionalTargetVeto),
            "few_shot_veto_only_claim_matches": artifact.get("classification")
            == "target_simulator_labeled_few_shot_adaptation"
            and artifact.get("zero_shot") is False
            and artifact.get("veto_role")
            == "proposal_conditional_veto_only_never_action_originator",
            "six_seed_folds_exact": len(folds) == 6
            and len({int(row.get("heldout_seed", -1)) for row in folds}) == 6
            and all(
                {int(value) for value in row.get("training_seeds", ())}
                == proposal_seeds - {int(row.get("heldout_seed", -1))}
                for row in folds
            ),
            "oof_proposal_partition_exact": bool(groups)
            and len(groups) == len(set(groups))
            and len(groups)
            == int(certificate.get("proposal_training_row_count", -1)),
            "five_candidate_grid_exact": len(
                certificate.get("gate_selection", {}).get("grid", ())
            )
            == 5,
            "selected_config_consistent": (
                selected is not None
                and active
                and bool(config.get("enabled"))
                and bool(target_veto.config.enabled)
                and float(config.get("acceptance_quantile"))
                == float(selected.get("acceptance_quantile"))
                == float(target_veto.config.acceptance_quantile)
            )
            or (
                selected is None
                and not active
                and not bool(config.get("enabled"))
                and not bool(target_veto.config.enabled)
            ),
            "feature_contract_matches": list(artifact["model"].feature_names)
            == list(protocol["training"]["features"]),
        }
        city_gate["passed"] = all(city_gate.values())
        if not city_gate["passed"]:
            errors.append(f"successor city gate failed: {city}")
        city_rows[city] = {
            "model": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
            "certificate": {
                "path": str(certificate_path.resolve()),
                "sha256": _sha256(certificate_path),
            },
            "active": active,
            "selected_gate": selected,
            "proposal_training_row_count": certificate.get(
                "proposal_training_row_count"
            ),
            "gate": city_gate,
        }
    integrity_gate = {
        "top_gate_passed": all(top_gate.values()),
        "all_expected_cities_present": set(city_rows) == expected_cities,
        "all_city_gates_passed": bool(city_rows)
        and all(row["gate"]["passed"] for row in city_rows.values()),
        "at_least_one_active_city": any(row["active"] for row in city_rows.values()),
        "at_least_one_safe_fallback_city": any(
            not row["active"] for row in city_rows.values()
        ),
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": AUDIT_DECISION if integrity_gate["passed"] else "reject",
        "claim_boundary": (
            "Artifact and OOF partition integrity after outcome-informed successor "
            "development. No closed-loop efficacy is claimed."
        ),
        "training_result_sha256": _sha256(training_result_path),
        "successor_protocol_sha256": _sha256(successor_protocol_path),
        "top_gate": top_gate,
        "cities": city_rows,
        "integrity_gate": integrity_gate,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--successor-protocol", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite successor audit: {args.out}")
    payload = audit_successor(
        training_result_path=args.training_result,
        successor_protocol_path=args.successor_protocol,
        artifact_root=args.artifact_root,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "active_cities": [
                    city for city, row in payload["cities"].items() if row["active"]
                ],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
