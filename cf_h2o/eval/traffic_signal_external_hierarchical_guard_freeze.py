"""Freeze a no-refit pressure/conformal successor after the 9277 diagnosis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_bytes,
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (
    V9_ALIGNED_AUDIT_DECISION,
    V9_ALIGNED_AUDIT_PROTOCOL,
    V9_ALIGNED_PROTOCOL,
    _load_aligned_artifacts,
    _read_json,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_guard_diagnostic import (
    HIERARCHICAL,
    RESULT_PROTOCOL as DIAGNOSTIC_PROTOCOL,
    SUCCESSOR_DECISION,
)
from cf_h2o.eval.traffic_signal_external_pressure_regularized_confirmation import (
    V9_JOINT_AUDIT_DECISION as V9_PRESSURE_AUDIT_DECISION,
    V9_JOINT_AUDIT_PROTOCOL as V9_PRESSURE_AUDIT_PROTOCOL,
    V9_PROTOCOL as V9_PRESSURE_PROTOCOL,
    _load_artifacts as _load_pressure_artifacts,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ContrastGuardConfig,
    PriorRegularizationConfig,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
)
from cf_h2o.traffic_signal.target_action_support import (
    HierarchicalTargetActionSupport,
    TargetActionSupport,
)


RESULT_PROTOCOL = "tsc-v60r56-external-v9-hierarchical-guard-freeze-v1"
MODEL_PROTOCOL = "cfcmt-external-v9-hierarchical-pressure-conformal-model-v1"
DEPLOYMENT_POLICY = "cfcmt_hierarchical_guard_mpc"
RUNTIME_POLICY_SUFFIX = "_hierarchical_guard"


def _object_sha256(value: object) -> str:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()


def _future_cache_file_count(path: Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(item.is_file() for item in root.rglob("*"))


def _guard_payload(guard: ContrastGuardConfig) -> dict[str, Any]:
    return {
        "enabled": bool(guard.enabled),
        "risk_multiplier": float(guard.risk_multiplier),
        "min_context_trust": float(guard.min_context_trust),
        "margin": float(guard.margin),
        "max_relative_rule_gap": float(guard.max_relative_rule_gap),
    }


def _regularizer_payload(
    regularizer: PriorRegularizationConfig,
) -> dict[str, Any]:
    return {
        "enabled": bool(regularizer.enabled),
        "blend_weight": float(regularizer.blend_weight),
        "risk_multiplier": float(regularizer.risk_multiplier),
        "min_context_trust": float(regularizer.min_context_trust),
    }


def _write_model(path: Path, payload: Mapping[str, Any], *, city: str) -> dict[str, Any]:
    if Path(path).exists():
        raise FileExistsError(f"refusing to overwrite hierarchical model: {path}")
    _atomic_bytes(path, pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL))
    loaded = pickle.loads(Path(path).read_bytes())
    if (
        loaded.get("protocol") != MODEL_PROTOCOL
        or loaded.get("city") != city
        or not isinstance(loaded.get("models"), OfflineScreeningModels)
        or not isinstance(loaded.get("regularizer"), PriorRegularizationConfig)
        or not isinstance(loaded.get("guard"), ContrastGuardConfig)
        or not isinstance(
            loaded.get("target_support"), HierarchicalTargetActionSupport
        )
    ):
        raise ValueError("hierarchical model round-trip failed")
    return {
        "protocol": MODEL_PROTOCOL,
        "path": str(Path(path).resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(Path(path).stat().st_size),
        "round_trip_passed": True,
    }


def run_hierarchical_guard_freeze(
    *,
    aligned_protocol_path: Path,
    aligned_joint_audit_path: Path,
    expected_aligned_joint_audit_sha256: str,
    aligned_freeze_root: Path,
    pressure_protocol_path: Path,
    pressure_joint_audit_path: Path,
    expected_pressure_joint_audit_sha256: str,
    pressure_freeze_root: Path,
    diagnostic_path: Path,
    expected_diagnostic_sha256: str,
    future_evaluation_cache_root: Path,
    future_evaluation_seeds: Sequence[int],
    expected_source_tree_sha256: str | None,
    city: str,
    model_out: Path,
) -> dict[str, Any]:
    aligned = _read_json(aligned_protocol_path)
    aligned_joint = _read_json(aligned_joint_audit_path)
    pressure = _read_json(pressure_protocol_path)
    pressure_joint = _read_json(pressure_joint_audit_path)
    diagnostic = _read_json(diagnostic_path)
    aligned_joint_sha = _sha256(aligned_joint_audit_path)
    pressure_joint_sha = _sha256(pressure_joint_audit_path)
    diagnostic_sha = _sha256(diagnostic_path)
    if (
        aligned.get("protocol") != V9_ALIGNED_PROTOCOL
        or aligned_joint_sha != str(expected_aligned_joint_audit_sha256)
        or aligned_joint.get("protocol") != V9_ALIGNED_AUDIT_PROTOCOL
        or aligned_joint.get("status") != "PASS"
        or aligned_joint.get("decision") != V9_ALIGNED_AUDIT_DECISION
        or pressure.get("protocol") != V9_PRESSURE_PROTOCOL
        or pressure_joint_sha != str(expected_pressure_joint_audit_sha256)
        or pressure_joint.get("protocol") != V9_PRESSURE_AUDIT_PROTOCOL
        or pressure_joint.get("status") != "PASS"
        or pressure_joint.get("decision") != V9_PRESSURE_AUDIT_DECISION
        or diagnostic_sha != str(expected_diagnostic_sha256)
        or diagnostic.get("protocol") != DIAGNOSTIC_PROTOCOL
        or diagnostic.get("decision") != SUCCESSOR_DECISION
        or not bool(diagnostic.get("diagnostic_gate", {}).get("passed", False))
    ):
        raise ValueError("hierarchical freeze evidence chain changed")
    if city not in diagnostic.get("city_results", {}):
        raise ValueError(f"unknown hierarchical freeze city: {city}")
    if not bool(
        diagnostic["city_results"][city]["variants"][HIERARCHICAL]["gate"][
            "passed"
        ]
    ):
        raise ValueError(f"hierarchical diagnostic did not pass for {city}")
    seeds = tuple(int(value) for value in future_evaluation_seeds)
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("hierarchical freeze requires unique future seeds")
    future_cache_count = _future_cache_file_count(future_evaluation_cache_root)
    if future_cache_count != 0:
        raise ValueError("future hierarchical evaluation cache is not empty")

    runtime = _runtime_metadata()
    if expected_source_tree_sha256 and runtime.get("source_tree_sha256") != str(
        expected_source_tree_sha256
    ):
        raise ValueError("hierarchical freeze source-tree SHA-256 mismatch")
    aligned_certificate, aligned_model = _load_aligned_artifacts(
        protocol=aligned,
        joint=aligned_joint,
        aligned_freeze_root=aligned_freeze_root,
        city=city,
    )
    pressure_certificate, pressure_model = _load_pressure_artifacts(
        protocol=pressure,
        joint=pressure_joint,
        freeze_root=pressure_freeze_root,
        city=city,
    )
    aligned_models_sha = _object_sha256(aligned_model["models"])
    pressure_models_sha = _object_sha256(pressure_model["models"])
    if (
        aligned_models_sha != pressure_models_sha
        or aligned_model["selected_candidate"]
        != pressure_model["selected_candidate"]
    ):
        raise ValueError("hierarchical parent fitted model identity changed")
    guard = aligned_model.get("guard")
    regularizer = pressure_model.get("regularizer")
    conformal_support = aligned_model.get("target_support")
    pressure_support = pressure_model.get("target_support")
    if (
        not isinstance(guard, ContrastGuardConfig)
        or not isinstance(regularizer, PriorRegularizationConfig)
        or not regularizer.enabled
        or not isinstance(conformal_support, TargetActionSupport)
        or not isinstance(pressure_support, TargetActionSupport)
        or not pressure_support.label_free
        or (guard.enabled and conformal_support.label_free)
    ):
        raise ValueError("hierarchical parent gate/support contract failed")
    support = HierarchicalTargetActionSupport(
        pressure_support=pressure_support,
        conformal_support=conformal_support,
        conformal_guard_enabled=bool(guard.enabled),
    )
    model_payload = {
        "protocol": MODEL_PROTOCOL,
        "city": city,
        "scenarios": tuple(aligned_model["scenarios"]),
        "anchor_policy": str(aligned_model["anchor_policy"]),
        "selected_candidate": str(aligned_model["selected_candidate"]),
        "models": aligned_model["models"],
        "regularizer": regularizer,
        "guard": guard,
        "pressure_target_support": pressure_support,
        "conformal_target_support": conformal_support,
        "target_support": support,
        "selected_group_ids": tuple(aligned_model["selected_group_ids"]),
        "fold_assignment": dict(aligned_model["fold_assignment"]),
        "aligned_certificate_sha256": str(
            aligned["city_artifacts"][city]["certificate"]["sha256"]
        ),
        "aligned_model_sha256": str(
            aligned["city_artifacts"][city]["model"]["sha256"]
        ),
        "pressure_certificate_sha256": str(
            pressure["city_artifacts"][city]["certificate"]["sha256"]
        ),
        "pressure_model_sha256": str(
            pressure["city_artifacts"][city]["model"]["sha256"]
        ),
        "diagnostic_sha256": diagnostic_sha,
    }
    model_artifact = _write_model(model_out, model_payload, city=city)
    freeze_gate = {
        "aligned_parent_audit_verified": True,
        "pressure_parent_audit_verified": True,
        "post_failure_diagnostic_declared_and_passed": True,
        "common_fitted_model_identity_verified": aligned_models_sha
        == pressure_models_sha,
        "no_model_refit": True,
        "regularizer_inherited_exactly": _regularizer_payload(regularizer)
        == pressure_certificate["deployment"]["regularizer"],
        "guard_inherited_exactly": _guard_payload(guard)
        == aligned_certificate["deployment"]["guard"],
        "pressure_support_inherited_exactly": pressure_support.diagnostics()
        == pressure_certificate["deployment"]["target_support"],
        "conformal_support_inherited_exactly": conformal_support.diagnostics()
        == aligned_certificate["deployment"]["target_support"],
        "future_seed_cache_empty_at_freeze": future_cache_count == 0,
        "future_seed_count_at_least_two": len(seeds) >= 2,
        "model_artifact_round_trip": bool(model_artifact["round_trip_passed"]),
    }
    freeze_gate["passed"] = all(freeze_gate.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": runtime,
        "city": city,
        "claim_boundary": (
            "The successor structure was chosen after inspecting seed 9277. The "
            "artifact contains only adaptation-fitted v55/v56 parents and frozen "
            "gates; it contains no seed-9277 label or future-seed outcome."
        ),
        "selected_candidate": str(aligned_model["selected_candidate"]),
        "common_fitted_model_sha256": aligned_models_sha,
        "aligned_protocol_sha256": _sha256(aligned_protocol_path),
        "aligned_joint_audit_sha256": aligned_joint_sha,
        "pressure_protocol_sha256": _sha256(pressure_protocol_path),
        "pressure_joint_audit_sha256": pressure_joint_sha,
        "diagnostic_sha256": diagnostic_sha,
        "parent_artifacts": {
            "aligned_certificate_sha256": model_payload[
                "aligned_certificate_sha256"
            ],
            "aligned_model_sha256": model_payload["aligned_model_sha256"],
            "pressure_certificate_sha256": model_payload[
                "pressure_certificate_sha256"
            ],
            "pressure_model_sha256": model_payload["pressure_model_sha256"],
        },
        "future_evaluation": {
            "seeds": list(seeds),
            "cache_root": str(Path(future_evaluation_cache_root).resolve()),
            "cache_file_count_at_freeze": future_cache_count,
        },
        "deployment": {
            "policy": DEPLOYMENT_POLICY,
            "runtime_policy_suffix": RUNTIME_POLICY_SUFFIX,
            "fallback_policy": "phase_pressure",
            "regularizer": _regularizer_payload(regularizer),
            "conformal_guard": _guard_payload(guard),
            "target_support": support.diagnostics(),
            "coordination_mode_before_closed_loop_selection": "direct",
        },
        "model_artifact": model_artifact,
        "freeze_gate": freeze_gate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-protocol", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-aligned-joint-audit-sha256", required=True)
    parser.add_argument("--aligned-freeze-root", type=Path, required=True)
    parser.add_argument("--pressure-protocol", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-pressure-joint-audit-sha256", required=True)
    parser.add_argument("--pressure-freeze-root", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--expected-diagnostic-sha256", required=True)
    parser.add_argument("--future-evaluation-cache-root", type=Path, required=True)
    parser.add_argument("--future-evaluation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-source-tree-sha256")
    parser.add_argument("--city", required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.model_out.exists():
        raise FileExistsError("refusing to overwrite hierarchical freeze evidence")
    payload = run_hierarchical_guard_freeze(
        aligned_protocol_path=args.aligned_protocol,
        aligned_joint_audit_path=args.aligned_joint_audit,
        expected_aligned_joint_audit_sha256=args.expected_aligned_joint_audit_sha256,
        aligned_freeze_root=args.aligned_freeze_root,
        pressure_protocol_path=args.pressure_protocol,
        pressure_joint_audit_path=args.pressure_joint_audit,
        expected_pressure_joint_audit_sha256=args.expected_pressure_joint_audit_sha256,
        pressure_freeze_root=args.pressure_freeze_root,
        diagnostic_path=args.diagnostic,
        expected_diagnostic_sha256=args.expected_diagnostic_sha256,
        future_evaluation_cache_root=args.future_evaluation_cache_root,
        future_evaluation_seeds=args.future_evaluation_seeds,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        city=args.city,
        model_out=args.model_out,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "city": args.city,
                "gate_passed": payload["freeze_gate"]["passed"],
                "model_sha256": payload["model_artifact"]["sha256"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["freeze_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
