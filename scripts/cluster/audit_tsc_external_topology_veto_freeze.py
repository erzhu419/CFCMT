#!/usr/bin/env python3
"""Audit v77 topology-only artifacts and runtime scoring contracts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import socket
import sys
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_topology_veto_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION,
    FREEZE_PROTOCOL,
    RESULT_PROTOCOL,
    WINNER_VARIANT,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (  # noqa: E402
    TARGET_SUPPORT_FEATURES,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset  # noqa: E402
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (  # noqa: E402
    ProposalConditionalRegressor,
)
from cf_h2o.traffic_signal.topology_target_veto import (  # noqa: E402
    ARTIFACT_PROTOCOL,
    VETO_PROTOCOL,
    TopologyTargetVeto,
)


AUDIT_PROTOCOL = "tsc-v77r73-topology-veto-joint-freeze-audit-v1"
AUDIT_DECISION = "authorize_topology_veto_closed_loop_development_only"


def _runtime_probe(veto: TopologyTargetVeto) -> dict[str, Any]:
    tls_id = sorted(veto.topology_context.tls_features)[0]
    features = np.zeros((2, len(TARGET_SUPPORT_FEATURES)), dtype=float)
    dataset = MechanismDataset(
        feature_names=tuple(TARGET_SUPPORT_FEATURES),
        features=features,
        context_names=("probe",),
        context=np.zeros((2, 1), dtype=float),
        priors={"cost": np.zeros(2, dtype=float)},
        targets={"cost": np.zeros(2, dtype=float)},
        domains=np.asarray(["probe", "probe"]),
        metadata={
            "action_group_ids": ["probe", "probe"],
            "is_reference": [True, False],
            "row_tls": [tls_id, tls_id],
        },
    )
    result = veto.evaluate(dataset, candidate_index=1, reference_index=0)
    return {
        "passed": (
            result["protocol"] == VETO_PROTOCOL
            and np.isfinite(float(result["predicted_delta"]))
            and result["runtime_tls_id"] == tls_id
            and (
                bool(veto.config.enabled)
                or (
                    result["eligible"] is False
                    and result["rejection"] == "disabled"
                )
            )
        ),
        "tls_id": tls_id,
        "predicted_delta": float(result["predicted_delta"]),
        "eligible": bool(result["eligible"]),
        "rejection": str(result["rejection"]),
    }


def audit(
    *, protocol_path: Path, result_path: Path, artifact_root: Path
) -> dict[str, Any]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    cache_protocol = json.loads(
        Path(protocol["cache_protocol"]["path"]).read_text(encoding="utf-8")
    )
    cities = {}
    for city, expected_scenarios in sorted(cache_protocol["city_scenarios"].items()):
        evidence = result["city_artifacts"][city]
        model_path = Path(artifact_root) / city / "model.pkl"
        certificate_path = Path(artifact_root) / city / "freeze.json"
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        artifact = pickle.loads(model_path.read_bytes())
        vetoes = artifact.get("scenario_vetoes", {})
        probes = {
            scenario: _runtime_probe(vetoes[scenario])
            for scenario in sorted(vetoes)
        }
        expected_active = (
            protocol["model"]["city_decisions"][city]["decision"]
            == "freeze_active_veto"
        )
        gates = {
            "model_hash": _sha256(model_path) == evidence["model"]["sha256"],
            "certificate_hash": _sha256(certificate_path)
            == evidence["certificate"]["sha256"],
            "certificate_protocol": certificate.get("protocol") == RESULT_PROTOCOL,
            "artifact_protocol": artifact.get("protocol") == ARTIFACT_PROTOCOL,
            "artifact_city": artifact.get("city") == city,
            "artifact_variant": artifact.get("variant") == WINNER_VARIANT,
            "artifact_model_type": isinstance(
                artifact.get("model"), ProposalConditionalRegressor
            ),
            "feature_contract": tuple(artifact["model"].feature_names)
            == tuple(protocol["model"]["variant_spec"]["feature_names"]),
            "scenario_set": set(vetoes) == set(expected_scenarios),
            "veto_types": all(
                isinstance(veto, TopologyTargetVeto) for veto in vetoes.values()
            ),
            "activation_exact": bool(artifact["config"].enabled)
            == expected_active
            == bool(evidence["active"]),
            "no_scenario_id_feature": artifact.get("scenario_id_is_model_feature")
            is False,
            "veto_only_role": artifact.get("veto_role")
            == "static_topology_veto_only_never_action_originator",
            "runtime_probes": bool(probes)
            and all(bool(probe["passed"]) for probe in probes.values()),
        }
        gates["passed"] = all(gates.values())
        cities[city] = {
            "status": "PASS" if gates["passed"] else "FAIL",
            "active": expected_active,
            "scenarios": list(expected_scenarios),
            "model": evidence["model"],
            "certificate": evidence["certificate"],
            "runtime_probes": probes,
            "integrity_gate": gates,
        }
    gates = {
        "protocol_identity": protocol.get("protocol") == FREEZE_PROTOCOL,
        "result_identity": result.get("protocol") == RESULT_PROTOCOL,
        "result_passed": result.get("status") == "PASS",
        "result_decision": result.get("decision") == AUTHORIZATION_DECISION,
        "result_binds_protocol": result.get("freeze_protocol_sha256")
        == _sha256(protocol_path),
        "result_integrity": result.get("integrity_gate", {}).get("passed") is True,
        "source_gate": bool(result.get("source_gate"))
        and all(bool(value) for value in result["source_gate"].values()),
        "all_city_audits": bool(cities)
        and all(row["status"] == "PASS" for row in cities.values()),
        "active_city_exact": result.get("active_cities") == ["jinan"],
        "los_angeles_disabled": cities.get("los_angeles", {}).get("active") is False,
        "sealed_absent_at_freeze": all(
            int(value) == 0
            for value in protocol["future_file_counts_at_freeze"].values()
        ),
    }
    gates["passed"] = all(gates.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "status": "PASS" if gates["passed"] else "FAIL",
        "decision": AUDIT_DECISION if gates["passed"] else "reject",
        "freeze_protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
        },
        "freeze_result": {
            "path": str(result_path),
            "sha256": _sha256(result_path),
        },
        "artifact_root": str(Path(artifact_root).resolve()),
        "cities": cities,
        "integrity_gate": gates,
        "claim_boundary": (
            "Artifact and runtime-contract audit only. No closed-loop efficacy or "
            "sealed-seed result has been observed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v77 audit: {args.out}")
    payload = audit(
        protocol_path=args.protocol,
        result_path=args.result,
        artifact_root=args.artifact_root,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "cities": {
                    city: {"status": row["status"], "active": row["active"]}
                    for city, row in payload["cities"].items()
                },
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
