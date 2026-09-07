#!/usr/bin/env python3
"""Freeze the v9 pressure artifacts and inherited deployment candidate grid."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (  # noqa: E402
    V9_ALIGNED_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation import (  # noqa: E402
    V9_JOINT_AUDIT_DECISION,
    V9_JOINT_AUDIT_PROTOCOL,
    V9_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_freeze import (  # noqa: E402
    V9_RESULT_PROTOCOL,
)


CITIES = ("los_angeles", "jinan")
POLICY_GRID_PROTOCOL = "tsc-v48r44-external-policy-horizon-adaptation-v1"
GLOBAL_GRID_PROTOCOL = (
    "tsc-v52r48-external-estimand-aligned-global-cooldown-development-v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def build_protocol(
    *,
    parent_protocol_path: Path,
    pressure_audit_path: Path,
    freeze_root: Path,
    policy_grid_path: Path,
    global_grid_path: Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    parent = _read_json(parent_protocol_path)
    audit = _read_json(pressure_audit_path)
    policy_grid = _read_json(policy_grid_path)
    global_grid = _read_json(global_grid_path)
    if parent.get("protocol") != V9_ALIGNED_PROTOCOL:
        raise ValueError("v9 pressure parent protocol changed")
    if (
        audit.get("protocol") != V9_JOINT_AUDIT_PROTOCOL
        or audit.get("status") != "PASS"
        or audit.get("decision") != V9_JOINT_AUDIT_DECISION
        or audit.get("generation") != "v9"
        or not bool(audit.get("integrity_gate", {}).get("passed", False))
    ):
        raise ValueError("v9 pressure audit does not authorize freezing")
    if (
        policy_grid.get("protocol") != POLICY_GRID_PROTOCOL
        or global_grid.get("protocol") != GLOBAL_GRID_PROTOCOL
        or int(global_grid["development_grid"]["candidate_count"]) != 24
    ):
        raise ValueError("inherited v52 deployment candidate grid changed")

    city_artifacts: dict[str, Any] = {}
    for city in CITIES:
        audit_city = dict(audit.get("cities", {}).get(city, {}))
        if not bool(audit_city.get("gate", {}).get("passed", False)):
            raise ValueError(f"v9 pressure city audit failed: {city}")
        certificate_path = Path(freeze_root) / city / "freeze.json"
        model_path = Path(freeze_root) / city / "model.pkl"
        certificate = _read_json(certificate_path)
        certificate_sha = _sha256(certificate_path)
        model_sha = _sha256(model_path)
        if (
            certificate.get("protocol") != V9_RESULT_PROTOCOL
            or certificate.get("city") != city
            or certificate_sha
            != str(audit_city.get("certificate", {}).get("sha256", ""))
            or model_sha != str(audit_city.get("model", {}).get("sha256", ""))
            or not bool(certificate.get("freeze_gate", {}).get("passed", False))
        ):
            raise ValueError(f"v9 pressure artifact identity changed: {city}")
        city_artifacts[city] = {
            "certificate": {
                "path": _reference(certificate_path),
                "sha256": certificate_sha,
            },
            "model": {"path": _reference(model_path), "sha256": model_sha},
            "regularizer": certificate.get("deployment", {}).get("regularizer"),
            "selected_candidate": certificate.get("selected_candidate"),
            "deployment_policy": certificate.get("deployment", {}).get("policy"),
        }

    development = dict(parent["development"])
    validation = dict(parent["validation_reservation"])
    prospective = dict(parent["prospective_confirmation_reservation"])
    policy_horizon = dict(policy_grid["policy_horizon_adaptation"])
    if tuple(policy_horizon["seeds"]) != tuple(
        development["adaptation_coupled_diagnostic_seeds"]
    ):
        raise ValueError("v52 base grid no longer matches adaptation diagnostics")
    policy_horizon["selection_role"] = (
        "candidate definitions only; v9 selection cannot use adaptation-coupled seeds"
    )
    return {
        "protocol": V9_PROTOCOL,
        "frozen_at_utc": str(frozen_at_utc),
        "claim_boundary": (
            "Pressure artifacts and the numeric v52 deployment grid are frozen before "
            "any repaired-network closed-loop efficacy outcome. The two adaptation "
            "seeds cannot enter the v9 controller selector."
        ),
        "parent_protocol": {
            "path": _reference(parent_protocol_path),
            "sha256": _sha256(parent_protocol_path),
        },
        "pressure_regularized_joint_audit": {
            "path": _reference(pressure_audit_path),
            "sha256": _sha256(pressure_audit_path),
            "decision": V9_JOINT_AUDIT_DECISION,
        },
        "city_artifacts": city_artifacts,
        "city_scenarios": prospective["city_scenarios"],
        "model_artifacts": {
            "pressure_joint_audit_sha256": _sha256(pressure_audit_path),
            "freeze_root": _reference(freeze_root),
            "reuse_contract": (
                "reuse v56 models and label-free support without refitting from "
                "offline seed 9277 or closed-loop outcomes"
            ),
        },
        "inherited_candidate_grid": {
            "policy_horizon_protocol": {
                "path": _reference(policy_grid_path),
                "sha256": _sha256(policy_grid_path),
            },
            "global_cooldown_protocol": {
                "path": _reference(global_grid_path),
                "sha256": _sha256(global_grid_path),
            },
            "base_candidates": list(
                global_grid["development_grid"]["base_candidates"]
            ),
            "execution_trust_regions": list(
                global_grid["development_grid"]["execution_trust_regions"]
            ),
            "candidate_count": 24,
            "numeric_grid_must_not_change_after_v9_closed_loop_starts": True,
        },
        "policy_horizon_adaptation": policy_horizon,
        "development": development,
        "validation_reservation": validation,
        "prospective_confirmation_reservation": prospective,
        "deployment_contract": {
            "fallback_policy": "phase_pressure",
            "coordination_mode": "sparse",
            "one_executed_residual_then_network_prior_only": True,
            "per_city_selection": True,
            "equal_scenario_weight_within_city": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--pressure-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--policy-grid", type=Path, required=True)
    parser.add_argument("--global-grid", type=Path, required=True)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite pressure protocol: {args.out}")
    payload = build_protocol(
        parent_protocol_path=args.parent_protocol,
        pressure_audit_path=args.pressure_audit,
        freeze_root=args.freeze_root,
        policy_grid_path=args.policy_grid,
        global_grid_path=args.global_grid,
        frozen_at_utc=args.frozen_at_utc or datetime.now(timezone.utc).isoformat(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"protocol": payload["protocol"], "sha256": _sha256(args.out), "out": str(args.out)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
