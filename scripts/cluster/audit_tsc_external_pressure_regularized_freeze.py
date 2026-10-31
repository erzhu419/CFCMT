#!/usr/bin/env python3
"""Jointly audit the two-city v47 pressure-regularized freeze."""

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
    ADAPTATION_SEEDS,
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_freeze import (  # noqa: E402
    ANCHOR_POLICY,
    DEPLOYMENT_POLICY,
    MODEL_PROTOCOL,
    RESULT_PROTOCOL,
    V9_MODEL_PROTOCOL,
    V9_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (  # noqa: E402
    PriorRegularizationConfig,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (  # noqa: E402
    OfflineScreeningModels,
)
from cf_h2o.traffic_signal.target_action_support import (  # noqa: E402
    TargetActionSupport,
)
from scripts.cluster.launch_tsc_external_pressure_regularized_freeze import (  # noqa: E402
    ASSIGNMENTS,
    LAUNCH_PROTOCOL,
    V9_ASSIGNMENTS,
    V9_LAUNCH_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v47r43-external-oof-pressure-regularized-joint-audit-v1"
AUDIT_DECISION = "authorize_pressure_regularized_contaminated_development_only"
V9_AUDIT_PROTOCOL = (
    "tsc-v56r52-external-v9-oof-pressure-regularized-joint-audit-v1"
)
V9_AUDIT_DECISION = "authorize_v9_pressure_regularized_frozen_development_replay"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit_pressure_regularized_freeze(
    *, freeze_root: Path, launch_manifest: Path, generation: str = "v8"
) -> dict[str, Any]:
    if generation not in {"v8", "v9"}:
        raise ValueError(f"unsupported pressure-freeze audit generation: {generation}")
    assignments = ASSIGNMENTS if generation == "v8" else V9_ASSIGNMENTS
    expected_launch_protocol = (
        LAUNCH_PROTOCOL if generation == "v8" else V9_LAUNCH_PROTOCOL
    )
    expected_result_protocol = (
        RESULT_PROTOCOL if generation == "v8" else V9_RESULT_PROTOCOL
    )
    expected_model_protocol = (
        MODEL_PROTOCOL if generation == "v8" else V9_MODEL_PROTOCOL
    )
    launch = _read_json(launch_manifest)
    expected_cities = {city for _, city in assignments}
    specs = {Path(str(row["result_dir"])).name: row for row in launch.get("specs", ())}
    launch_gate = {
        "protocol_matches": launch.get("protocol") == expected_launch_protocol,
        "result_protocol_matches": launch.get("result_protocol")
        == expected_result_protocol,
        "submitted": bool(launch.get("submitted", False)),
        "task_count_matches": int(launch.get("task_count", -1))
        == len(expected_cities),
        "city_matrix_exact": set(specs) == expected_cities,
    }
    launch_gate["passed"] = all(launch_gate.values())

    cities = {}
    errors = []
    for city in sorted(expected_cities):
        root = Path(freeze_root) / city
        certificate_path = root / "freeze.json"
        model_path = root / "model.pkl"
        if not certificate_path.is_file() or not model_path.is_file():
            errors.append(f"missing pressure-regularized artifact: {city}")
            continue
        certificate = _read_json(certificate_path)
        model = pickle.loads(model_path.read_bytes())
        regularizer = model.get("regularizer")
        support = model.get("target_support")
        models = model.get("models")
        spec = specs[city]
        model_sha = _sha256(model_path)
        certificate_sha = _sha256(certificate_path)
        prospective = {
            int(value)
            for value in certificate.get("excluded_seed_roles", {}).get(
                "prospective_confirmation", ()
            )
        }
        development = {
            int(value)
            for value in certificate.get("excluded_seed_roles", {}).get(
                "closed_loop_development", ()
            )
        }
        adaptation_diagnostics = {
            int(value)
            for value in certificate.get("excluded_seed_roles", {}).get(
                "closed_loop_adaptation_diagnostics", ()
            )
        }
        generalization_development = {
            int(value)
            for value in certificate.get("excluded_seed_roles", {}).get(
                "closed_loop_generalization_development", ()
            )
        }
        validation = {
            int(value)
            for value in certificate.get("excluded_seed_roles", {}).get(
                "closed_loop_validation", ()
            )
        }
        selected_groups = tuple(str(value) for value in model.get("selected_group_ids", ()))
        row_gate = {
            "certificate_protocol_matches": certificate.get("protocol")
            == expected_result_protocol,
            "model_protocol_matches": model.get("protocol")
            == expected_model_protocol,
            "city_matches": certificate.get("city") == model.get("city") == city,
            "physical_host_matches": certificate.get("hostname")
            == spec.get("require_node"),
            "source_tree_matches_snapshot": certificate.get("runtime", {}).get(
                "source_tree_sha256"
            )
            == launch.get("source_tree_sha256"),
            "model_hash_matches_certificate": model_sha
            == certificate.get("model_artifact", {}).get("sha256"),
            "model_size_matches_certificate": model_path.stat().st_size
            == int(certificate.get("model_artifact", {}).get("size_bytes", -1)),
            "freeze_gate_passed": bool(
                certificate.get("freeze_gate", {}).get("passed", False)
            ),
            "anchor_is_phase_pressure": model.get("anchor_policy") == ANCHOR_POLICY
            and certificate.get("anchor_policy") == ANCHOR_POLICY,
            "regularizer_type_valid": isinstance(
                regularizer, PriorRegularizationConfig
            ),
            "regularizer_enabled": bool(getattr(regularizer, "enabled", False)),
            "deployment_policy_matches": certificate.get("deployment", {}).get(
                "policy"
            )
            == DEPLOYMENT_POLICY,
            "direct_coordination_frozen": certificate.get("deployment", {}).get(
                "coordination_mode"
            )
            == "direct",
            "support_type_valid": isinstance(support, TargetActionSupport),
            "support_is_label_free": bool(getattr(support, "label_free", False)),
            "models_type_valid": isinstance(models, OfflineScreeningModels),
            "parent_certificate_hash_matches": model.get(
                "parent_certificate_sha256"
            )
            == certificate.get("parent_certificate_sha256"),
            "parent_model_hash_matches": model.get("parent_model_sha256")
            == certificate.get("parent_model_sha256"),
            "selection_passed": bool(
                certificate.get("regularizer_selection", {})
                .get("selected", {})
                .get("feasible", False)
            ),
            "development_and_prospective_roles_disjoint": bool(
                development and prospective and development.isdisjoint(prospective)
            ),
            "validation_roles_disjoint": bool(
                generation == "v8"
                or (
                    validation
                    and validation.isdisjoint(development | prospective)
                )
            ),
            "excluded_seeds_absent_from_group_ids": not any(
                f"seed{seed}:" in group
                for seed in (
                    development | prospective
                    if generation == "v8"
                    else generalization_development | validation | prospective
                )
                for group in selected_groups
            ),
            "adaptation_diagnostic_overlap_declared": bool(
                generation == "v8"
                or (
                    adaptation_diagnostics == set(ADAPTATION_SEEDS)
                    and development - adaptation_diagnostics
                    == generalization_development
                    and all(
                        any(f"seed{seed}:" in group for group in selected_groups)
                        for seed in adaptation_diagnostics
                    )
                )
            ),
        }
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"pressure-regularized artifact gate failed: {city}")
        cities[city] = {
            "certificate": {
                "path": str(certificate_path.resolve()),
                "sha256": certificate_sha,
                "size_bytes": certificate_path.stat().st_size,
            },
            "model": {
                "path": str(model_path.resolve()),
                "sha256": model_sha,
                "size_bytes": model_path.stat().st_size,
            },
            "selected_candidate": certificate.get("selected_candidate"),
            "regularizer": certificate.get("deployment", {}).get("regularizer"),
            "target_support": certificate.get("deployment", {}).get(
                "target_support"
            ),
            "selection_summary": certificate.get("regularizer_selection", {}).get(
                "selected"
            ),
            "gate": row_gate,
        }

    integrity_gate = {
        "launch_gate_passed": bool(launch_gate["passed"]),
        "all_cities_present": set(cities) == expected_cities,
        "all_city_gates_passed": bool(cities)
        and all(row["gate"]["passed"] for row in cities.values()),
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    return {
        "protocol": AUDIT_PROTOCOL if generation == "v8" else V9_AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": (
            AUDIT_DECISION if generation == "v8" else V9_AUDIT_DECISION
        )
        if integrity_gate["passed"]
        else "reject",
        "claim_boundary": (
            "this audit authorizes only the frozen development replay; it does not "
            "authorize validation, prospective confirmation, or an effectiveness claim"
        ),
        "generation": generation,
        "launch_manifest_sha256": _sha256(launch_manifest),
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "launch_gate": launch_gate,
        "cities": cities,
        "integrity_gate": integrity_gate,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--generation", choices=("v8", "v9"), default="v8")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v47 joint audit: {args.out}")
    payload = audit_pressure_regularized_freeze(
        freeze_root=args.freeze_root,
        launch_manifest=args.launch_manifest,
        generation=args.generation,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
