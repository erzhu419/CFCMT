#!/usr/bin/env python3
"""Freeze the v61 fresh-seed and closed-loop successor protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_hierarchical_guard_diagnostic import (  # noqa: E402
    RESULT_PROTOCOL as DIAGNOSTIC_PROTOCOL,
    SUCCESSOR_DECISION,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_guard_freeze import (  # noqa: E402
    DEPLOYMENT_POLICY,
    MODEL_PROTOCOL,
    RESULT_PROTOCOL as FREEZE_RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.target_action_support import (  # noqa: E402
    HierarchicalTargetActionSupport,
)
from scripts.cluster.audit_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    AUDIT_DECISION,
    AUDIT_PROTOCOL,
)
from scripts.cluster.audit_tsc_seed_novelty import (  # noqa: E402
    PROTOCOL as NOVELTY_PROTOCOL,
)


PROTOCOL = "tsc-v61r57-external-v9-hierarchical-guard-confirmation-v1"
CITIES = ("los_angeles", "jinan")
SEED_DERIVATION_NAMESPACE = "cfcmt-v9-hierarchical-successor-v1"
SEED_ROLE = "offline-confirmation"
SEED_MIN = 13001
SEED_RANGE = 76999
FRESH_SEED_COUNT = 6


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


def derive_fresh_seeds(diagnostic_sha256: str) -> tuple[int, ...]:
    base = f"{diagnostic_sha256}|{SEED_DERIVATION_NAMESPACE}"
    return tuple(
        SEED_MIN
        + (
            int(
                hashlib.sha256(
                    f"{base}|{SEED_ROLE}|{index}".encode("utf-8")
                ).hexdigest()[:16],
                16,
            )
            % SEED_RANGE
        )
        for index in range(FRESH_SEED_COUNT)
    )


def build_protocol(
    *,
    diagnostic_path: Path,
    novelty_audit_path: Path,
    joint_audit_path: Path,
    freeze_root: Path,
    pressure_protocol_path: Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    diagnostic = _read_json(diagnostic_path)
    novelty = _read_json(novelty_audit_path)
    audit = _read_json(joint_audit_path)
    pressure = _read_json(pressure_protocol_path)
    diagnostic_sha = _sha256(diagnostic_path)
    fresh_seeds = derive_fresh_seeds(diagnostic_sha)
    if (
        diagnostic.get("protocol") != DIAGNOSTIC_PROTOCOL
        or diagnostic.get("decision") != SUCCESSOR_DECISION
        or not bool(diagnostic.get("diagnostic_gate", {}).get("passed", False))
        or novelty.get("protocol") != NOVELTY_PROTOCOL
        or novelty.get("status") != "PASS"
        or tuple(novelty.get("seeds", ())) != fresh_seeds
        or int(novelty.get("hit_count", -1)) != 0
        or audit.get("protocol") != AUDIT_PROTOCOL
        or audit.get("status") != "PASS"
        or audit.get("decision") != AUDIT_DECISION
        or tuple(audit.get("future_evaluation_seeds", ())) != fresh_seeds
        or int(audit.get("future_cache_file_count_at_freeze", -1)) != 0
        or not bool(audit.get("integrity_gate", {}).get("passed", False))
    ):
        raise ValueError("hierarchical protocol authorization changed")
    development = dict(pressure["development"])
    validation = dict(pressure["validation_reservation"])
    prospective = dict(pressure["prospective_confirmation_reservation"])
    adaptation_diagnostics = {
        int(value)
        for value in development["adaptation_coupled_diagnostic_seeds"]
    }
    development_seeds = tuple(
        int(value) for value in development["generalization_selection_seeds"]
    )
    validation_seeds = tuple(
        int(value) for value in validation["closed_loop_seeds"]
    )
    prospective_seeds = tuple(
        int(value) for value in prospective["closed_loop_seeds"]
    )
    role_sets = [
        set(fresh_seeds),
        set(development_seeds),
        set(validation_seeds),
        set(prospective_seeds),
        adaptation_diagnostics,
        {9277},
    ]
    if any(
        left & right
        for index, left in enumerate(role_sets)
        for right in role_sets[index + 1 :]
    ):
        raise ValueError("hierarchical protocol seed roles overlap")

    city_artifacts: dict[str, Any] = {}
    for city in CITIES:
        audit_city = dict(audit["cities"][city])
        certificate_path = Path(freeze_root) / city / "freeze.json"
        model_path = Path(freeze_root) / city / "model.pkl"
        certificate = _read_json(certificate_path)
        model = pickle.loads(model_path.read_bytes())
        if (
            certificate.get("protocol") != FREEZE_RESULT_PROTOCOL
            or certificate.get("city") != city
            or _sha256(certificate_path)
            != str(audit_city["certificate"]["sha256"])
            or model.get("protocol") != MODEL_PROTOCOL
            or _sha256(model_path) != str(audit_city["model"]["sha256"])
            or not isinstance(
                model.get("target_support"), HierarchicalTargetActionSupport
            )
            or not bool(certificate.get("freeze_gate", {}).get("passed", False))
        ):
            raise ValueError(f"hierarchical city artifact changed: {city}")
        city_artifacts[city] = {
            "certificate": {
                "path": _reference(certificate_path),
                "sha256": _sha256(certificate_path),
            },
            "model": {
                "path": _reference(model_path),
                "sha256": _sha256(model_path),
            },
            "selected_candidate": model["selected_candidate"],
            "deployment": certificate["deployment"],
        }

    inherited_grid = dict(pressure["inherited_candidate_grid"])
    if int(inherited_grid["candidate_count"]) != 24:
        raise ValueError("hierarchical inherited execution grid changed")
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": str(frozen_at_utc),
        "claim_boundary": (
            "The controller structure was selected on the explicitly contaminated "
            "seed 9277. Models and all numeric gates are now immutable. Six hash-"
            "derived offline seeds, eight closed-loop development seeds, sixteen "
            "validation seeds, and three prospective seeds have disjoint roles."
        ),
        "diagnostic_parent": {
            "path": _reference(diagnostic_path),
            "sha256": diagnostic_sha,
            "role": "post-failure_design_diagnostic_not_confirmation",
        },
        "seed_novelty_audit": {
            "path": _reference(novelty_audit_path),
            "sha256": _sha256(novelty_audit_path),
            "status": "PASS",
        },
        "hierarchical_joint_audit": {
            "path": _reference(joint_audit_path),
            "sha256": _sha256(joint_audit_path),
            "decision": AUDIT_DECISION,
        },
        "city_artifacts": city_artifacts,
        "city_scenarios": pressure["city_scenarios"],
        "fresh_offline_confirmation": {
            "seeds": list(fresh_seeds),
            "seed_derivation": {
                "algorithm": "sha256-prefix64-modulo-v1",
                "input": (
                    "<diagnostic_sha256>|cfcmt-v9-hierarchical-successor-v1|"
                    "offline-confirmation|<index>"
                ),
                "diagnostic_sha256": diagnostic_sha,
                "indices": list(range(FRESH_SEED_COUNT)),
                "minimum": SEED_MIN,
                "modulus": SEED_RANGE,
            },
            "duration_sec": 600,
            "warmup_sec": 60,
            "control_interval_sec": 10,
            "counterfactual_horizon_intervals": 6,
            "collection_shards_per_scenario_seed": 16,
            "behavior_policy": "phase_pressure",
            "reporting_unit": "city",
            "scenario_weighting_within_city": "equal",
            "seed_weighting_within_city": "equal",
            "action_level_gate": {
                "minimum_equal_seed_scenario_mean_improvement": 0.001,
                "maximum_bootstrap_95pct_upper_delta": 0.01,
                "maximum_seed_scenario_mean_delta": 0.01,
                "minimum_nonpositive_seed_fraction": 0.6666666666666666,
                "minimum_override_fraction": 0.005,
                "maximum_override_fraction": 0.5,
                "bootstrap_unit": "counterfactual_action_group_within_seed_scenario",
                "bootstrap_replicates": 10000,
                "all_cities_must_pass": True,
            },
            "authorization_if_passed": (
                "closed_loop_development_grid_replay_only"
            ),
        },
        "closed_loop_development": {
            "seeds": list(development_seeds),
            "duration_sec": 3600,
            "primary_metric": "mean_tripinfo_waiting_time",
            "reporting_unit": "city",
            "scenario_weighting_within_city": "equal",
            "seed_weighting": "equal",
            "adaptation_coupled_seeds_excluded": sorted(adaptation_diagnostics),
            "candidate_grid": inherited_grid,
            "selection_rule": pressure["policy_horizon_adaptation"][
                "selection_rule"
            ],
            "fallback": "phase_pressure",
        },
        "closed_loop_validation": {
            **validation,
            "closed_loop_seeds": list(validation_seeds),
            "method_and_execution_candidate_must_be_hash_frozen_before_launch": True,
        },
        "prospective_confirmation": {
            **prospective,
            "closed_loop_seeds": list(prospective_seeds),
            "release_condition": "validation_gate_passed_and_analysis_hash_frozen",
        },
        "deployment_contract": {
            "method": DEPLOYMENT_POLICY,
            "runtime_policy_suffix": "_hierarchical_guard",
            "fallback_policy": "phase_pressure",
            "pressure_and_conformal_gates_applied_to_same_action": True,
            "no_target_refit_after_v60": True,
            "equal_city_reporting": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--novelty-audit", type=Path, required=True)
    parser.add_argument("--joint-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--pressure-protocol", type=Path, required=True)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite hierarchical protocol: {args.out}")
    payload = build_protocol(
        diagnostic_path=args.diagnostic,
        novelty_audit_path=args.novelty_audit,
        joint_audit_path=args.joint_audit,
        freeze_root=args.freeze_root,
        pressure_protocol_path=args.pressure_protocol,
        frozen_at_utc=args.frozen_at_utc or datetime.now(timezone.utc).isoformat(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
