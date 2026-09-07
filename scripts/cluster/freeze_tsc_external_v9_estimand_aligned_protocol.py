#!/usr/bin/env python3
"""Freeze the v9 estimand-aligned development protocol from audited artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (  # noqa: E402
    METHOD,
    V9_ALIGNED_AUDIT_DECISION,
    V9_ALIGNED_AUDIT_PROTOCOL,
    V9_ALIGNED_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_freeze import (  # noqa: E402
    V9_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (  # noqa: E402
    V9_PROTOCOL as V9_FULL_BUDGET_PROTOCOL,
)


CITIES = ("los_angeles", "jinan")
BASELINE_POLICIES = (
    "phase_pressure",
    "max_pressure",
    "fixed_time",
    "rigid_anchor_mpc",
    "simulator_only_mpc",
    "h2oplus_style_dense_residual_mpc",
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
    aligned_audit_path: Path,
    freeze_root: Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    parent = _read_json(parent_protocol_path)
    audit = _read_json(aligned_audit_path)
    if parent.get("protocol") != V9_FULL_BUDGET_PROTOCOL:
        raise ValueError("v9 aligned protocol parent changed")
    if (
        audit.get("protocol") != V9_ALIGNED_AUDIT_PROTOCOL
        or audit.get("status") != "PASS"
        or audit.get("decision") != V9_ALIGNED_AUDIT_DECISION
        or audit.get("generation") != "v9"
        or not bool(audit.get("gate", {}).get("passed", False))
    ):
        raise ValueError("v9 aligned artifact audit does not authorize freezing")

    target = dict(parent["target_protocol"])
    adaptation = tuple(int(value) for value in target["adaptation_seeds"])
    development = tuple(
        int(value) for value in target["closed_loop_development_seeds"]
    )
    validation = tuple(
        int(value) for value in target["closed_loop_validation_seeds"]
    )
    prospective = tuple(
        int(value) for value in target["closed_loop_prospective_seeds"]
    )
    adaptation_diagnostics = tuple(
        seed for seed in development if seed in set(adaptation)
    )
    generalization_development = tuple(
        seed for seed in development if seed not in set(adaptation)
    )
    if (
        set(adaptation_diagnostics) != set(adaptation)
        or not generalization_development
        or set(generalization_development)
        & (set(adaptation) | set(validation) | set(prospective))
        or set(validation) & set(prospective)
    ):
        raise ValueError("v9 development seed-role partition changed")

    city_artifacts: dict[str, Any] = {}
    for city in CITIES:
        audit_city = dict(audit.get("cities", {}).get(city, {}))
        if not bool(audit_city.get("gate", {}).get("passed", False)):
            raise ValueError(f"v9 aligned city audit failed: {city}")
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
            raise ValueError(f"v9 aligned artifact identity changed: {city}")
        support = dict(certificate.get("deployment", {}).get("target_support", {}))
        city_artifacts[city] = {
            "certificate": {
                "path": _reference(certificate_path),
                "sha256": certificate_sha,
            },
            "model": {"path": _reference(model_path), "sha256": model_sha},
            "deployment_decision": certificate.get("deployment", {}).get("policy"),
            "selected_candidate": certificate.get("selected_candidate"),
            "validated_oof_prototypes": int(support.get("prototype_count", 0)),
        }

    city_scenarios = {
        str(city): list(scenarios)
        for city, scenarios in target["external_city_scenarios"].items()
    }
    return {
        "protocol": V9_ALIGNED_PROTOCOL,
        "frozen_at_utc": str(frozen_at_utc),
        "claim_boundary": (
            "The repaired-network estimand-aligned artifacts use only adaptation "
            "counterfactual records. Seeds shared with adaptation are diagnostic-only; "
            "only the eight disjoint development seeds may select a controller."
        ),
        "parent_protocol": {
            "path": _reference(parent_protocol_path),
            "sha256": _sha256(parent_protocol_path),
        },
        "estimand_aligned_joint_audit": {
            "path": _reference(aligned_audit_path),
            "sha256": _sha256(aligned_audit_path),
            "decision": V9_ALIGNED_AUDIT_DECISION,
        },
        "city_artifacts": city_artifacts,
        "estimand_contract": {
            "counterfactual_behavior_policy": "phase_pressure",
            "counterfactual_rollout_policy": "phase_pressure",
            "contrast_reference_policy": "phase_pressure",
            "deployment_fallback_policy": "phase_pressure",
            "candidate_selection": "two-fold leave-one-adaptation-seed-out",
            "guard_selection": (
                "fold-max finite-sample conformal safety margin with fold and scenario gates"
            ),
            "deployment_support": (
                "deterministic topology-invariant support fitted only to OOF actions "
                "accepted by the frozen guard"
            ),
            "minimum_operational_total_vehicles": 1.0,
            "activation": "after the 60-second counterfactual warm-up",
        },
        "development": {
            "closed_loop_seeds": list(development),
            "adaptation_coupled_diagnostic_seeds": list(adaptation_diagnostics),
            "generalization_selection_seeds": list(generalization_development),
            "status": "repaired_network_frozen_development_replay",
            "permitted_use": (
                "candidate selection on generalization_selection_seeds only; "
                "adaptation-coupled seeds are descriptive diagnostics"
            ),
            "policies": [METHOD, "phase_pressure"],
            "primary_metric": "mean_tripinfo_waiting_time",
            "paired_unit": "scenario_seed",
        },
        "validation_reservation": {
            "closed_loop_seeds": list(validation),
            "must_not_run_until_development_selector_and_analysis_hash_are_frozen": True,
            "city_scenarios": city_scenarios,
        },
        "prospective_confirmation_reservation": {
            "closed_loop_seeds": list(prospective),
            "must_not_run_until_validation_passes_and_prospective_analysis_is_frozen": True,
            "city_scenarios": city_scenarios,
            "candidate_policies_for_later_freeze": [METHOD, *BASELINE_POLICIES],
        },
        "deployment": {
            "policy": METHOD,
            "coordination_mode": "sparse",
            "cooldown": "prediction_horizon_minus_one_control_interval",
            "failed_city_gate_falls_back_exactly_to_phase_pressure": True,
            "v9_network_repair_role": (
                "integrity correction after v52 efficacy inspection; all v9 efficacy "
                "must be reported separately from v8"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--aligned-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite aligned protocol: {args.out}")
    frozen_at = args.frozen_at_utc or datetime.now(timezone.utc).isoformat()
    payload = build_protocol(
        parent_protocol_path=args.parent_protocol,
        aligned_audit_path=args.aligned_audit,
        freeze_root=args.freeze_root,
        frozen_at_utc=frozen_at,
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
