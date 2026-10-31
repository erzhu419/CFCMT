#!/usr/bin/env python3
"""Jointly audit the two-city v60 hierarchical successor freeze."""

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
from cf_h2o.eval.traffic_signal_external_hierarchical_guard_freeze import (  # noqa: E402
    DEPLOYMENT_POLICY,
    MODEL_PROTOCOL,
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (  # noqa: E402
    ContrastGuardConfig,
    PriorRegularizationConfig,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (  # noqa: E402
    OfflineScreeningModels,
)
from cf_h2o.traffic_signal.target_action_support import (  # noqa: E402
    HierarchicalTargetActionSupport,
    TargetActionSupport,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    ASSIGNMENTS,
    LAUNCH_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v60r56-external-v9-hierarchical-guard-joint-audit-v1"
AUDIT_DECISION = "authorize_hierarchical_fresh_offline_seed_collection"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit_hierarchical_guard_freeze(
    *, freeze_root: Path, launch_manifest: Path
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    expected_cities = {city for _, city in ASSIGNMENTS}
    specs = {Path(str(row["result_dir"])).name: row for row in launch.get("specs", ())}
    future_seeds = tuple(
        int(value) for value in launch.get("future_evaluation_seeds", ())
    )
    launch_gate = {
        "protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "result_protocol_matches": launch.get("result_protocol")
        == RESULT_PROTOCOL,
        "submitted": bool(launch.get("submitted", False)),
        "task_count_matches": int(launch.get("task_count", -1))
        == len(expected_cities),
        "city_matrix_exact": set(specs) == expected_cities,
        "future_seed_count_at_least_two": len(future_seeds) >= 2,
        "future_seeds_unique": len(future_seeds) == len(set(future_seeds)),
    }
    launch_gate["passed"] = all(launch_gate.values())
    cities: dict[str, Any] = {}
    errors = []
    for city in sorted(expected_cities):
        root = Path(freeze_root) / city
        certificate_path = root / "freeze.json"
        model_path = root / "model.pkl"
        if not certificate_path.is_file() or not model_path.is_file():
            errors.append(f"missing hierarchical artifact: {city}")
            continue
        certificate = _read_json(certificate_path)
        model = pickle.loads(model_path.read_bytes())
        models = model.get("models")
        regularizer = model.get("regularizer")
        guard = model.get("guard")
        support = model.get("target_support")
        pressure_support = model.get("pressure_target_support")
        conformal_support = model.get("conformal_target_support")
        spec = specs[city]
        selected_groups = tuple(
            str(value) for value in model.get("selected_group_ids", ())
        )
        observed_group_seeds = {
            int(part[4:])
            for group in selected_groups
            for part in group.split(":")
            if part.startswith("seed")
        }
        row_gate = {
            "certificate_protocol_matches": certificate.get("protocol")
            == RESULT_PROTOCOL,
            "model_protocol_matches": model.get("protocol") == MODEL_PROTOCOL,
            "city_matches": certificate.get("city") == model.get("city") == city,
            "physical_host_matches": certificate.get("hostname")
            == spec.get("require_node"),
            "source_tree_matches_snapshot": certificate.get("runtime", {}).get(
                "source_tree_sha256"
            )
            == launch.get("source_tree_sha256"),
            "model_hash_matches_certificate": _sha256(model_path)
            == certificate.get("model_artifact", {}).get("sha256"),
            "model_size_matches_certificate": model_path.stat().st_size
            == int(certificate.get("model_artifact", {}).get("size_bytes", -1)),
            "freeze_gate_passed": bool(
                certificate.get("freeze_gate", {}).get("passed", False)
            ),
            "models_type_valid": isinstance(models, OfflineScreeningModels),
            "regularizer_type_and_enabled": isinstance(
                regularizer, PriorRegularizationConfig
            )
            and regularizer.enabled,
            "guard_type_valid": isinstance(guard, ContrastGuardConfig),
            "hierarchical_support_type_valid": isinstance(
                support, HierarchicalTargetActionSupport
            ),
            "component_support_types_valid": isinstance(
                pressure_support, TargetActionSupport
            )
            and isinstance(conformal_support, TargetActionSupport),
            "support_components_match_wrapper": bool(
                isinstance(support, HierarchicalTargetActionSupport)
                and support.pressure_support is pressure_support
                and support.conformal_support is conformal_support
                and support.conformal_guard_enabled == guard.enabled
            ),
            "deployment_policy_matches": certificate.get("deployment", {}).get(
                "policy"
            )
            == DEPLOYMENT_POLICY,
            "future_seed_list_matches": tuple(
                certificate.get("future_evaluation", {}).get("seeds", ())
            )
            == future_seeds,
            "future_cache_empty_at_freeze": int(
                certificate.get("future_evaluation", {}).get(
                    "cache_file_count_at_freeze", -1
                )
            )
            == 0,
            "adaptation_groups_only": observed_group_seeds == set(ADAPTATION_SEEDS),
            "diagnostic_and_future_labels_absent_from_model": not (
                observed_group_seeds & ({9277} | set(future_seeds))
            ),
            "parent_hashes_match_launch": certificate.get(
                "aligned_joint_audit_sha256"
            )
            == launch.get("input_hashes", {}).get("aligned_joint_audit")
            and certificate.get("pressure_joint_audit_sha256")
            == launch.get("input_hashes", {}).get("pressure_joint_audit")
            and certificate.get("diagnostic_sha256")
            == launch.get("input_hashes", {}).get("diagnostic"),
        }
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"hierarchical artifact gate failed: {city}")
        cities[city] = {
            "certificate": {
                "path": str(certificate_path.resolve()),
                "sha256": _sha256(certificate_path),
                "size_bytes": certificate_path.stat().st_size,
            },
            "model": {
                "path": str(model_path.resolve()),
                "sha256": _sha256(model_path),
                "size_bytes": model_path.stat().st_size,
            },
            "selected_candidate": model.get("selected_candidate"),
            "regularizer": certificate.get("deployment", {}).get("regularizer"),
            "conformal_guard": certificate.get("deployment", {}).get(
                "conformal_guard"
            ),
            "target_support": certificate.get("deployment", {}).get(
                "target_support"
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
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": AUDIT_DECISION if integrity_gate["passed"] else "reject",
        "claim_boundary": (
            "This audit authorizes only collection and action-level evaluation on "
            "the predeclared fresh offline seeds. It does not authorize closed-loop "
            "development, validation, prospective confirmation, or efficacy claims."
        ),
        "launch_manifest_sha256": _sha256(launch_manifest),
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "future_evaluation_seeds": list(future_seeds),
        "future_cache_file_count_at_freeze": 0,
        "launch_gate": launch_gate,
        "cities": cities,
        "integrity_gate": integrity_gate,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite hierarchical audit: {args.out}")
    payload = audit_hierarchical_guard_freeze(
        freeze_root=args.freeze_root, launch_manifest=args.launch_manifest
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
