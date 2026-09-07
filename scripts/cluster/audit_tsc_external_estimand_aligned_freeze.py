#!/usr/bin/env python3
"""Audit the estimand-aligned two-city freeze before any closed-loop rollout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (  # noqa: E402
    BASE_FAMILIES,
)
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_selection import (  # noqa: E402
    SELECTOR_SPECS,
    select_target_candidate,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    ADAPTATION_SEEDS,
    FROZEN_SELECTOR_KEY,
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_freeze import (  # noqa: E402
    ANCHOR_POLICY,
    MODEL_PROTOCOL,
    RESULT_PROTOCOL,
    V9_MODEL_PROTOCOL,
    V9_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (  # noqa: E402
    FOLD_COUNT,
    FOLD_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (  # noqa: E402
    MIN_OPERATIONAL_TOTAL_VEHICLES,
    accepted_oof_records,
    select_oof_trust_region,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import ContrastGuardConfig  # noqa: E402
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (  # noqa: E402
    OfflineScreeningModels,
)
from cf_h2o.traffic_signal.target_action_support import (  # noqa: E402
    TargetActionSupport,
)
from scripts.cluster.launch_tsc_external_estimand_aligned_freeze import (  # noqa: E402
    CITY_NODES,
    LAUNCH_PROTOCOL,
    V9_LAUNCH_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v46r42-external-oof-validated-support-joint-audit-v1"
DECISION = "authorize_oof_validated_support_contaminated_seed_development_only"
V9_DECISION = "authorize_v9_estimand_aligned_frozen_development_replay"
PROSPECTIVE_SEEDS = (10091, 11003, 12007)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _guard_payload(guard: ContrastGuardConfig) -> dict[str, Any]:
    return {
        "enabled": bool(guard.enabled),
        "risk_multiplier": float(guard.risk_multiplier),
        "min_context_trust": float(guard.min_context_trust),
        "margin": float(guard.margin),
        "max_relative_rule_gap": float(guard.max_relative_rule_gap),
    }


def _group_seed(group_id: str) -> int:
    matches = [part for part in str(group_id).split(":") if part.startswith("seed")]
    if len(matches) != 1:
        raise ValueError(f"group has no unique seed: {group_id}")
    return int(matches[0][4:])


def _model_gate(
    *,
    model_path: Path,
    certificate: Mapping[str, Any],
    city: str,
    expected_model_protocol: str = MODEL_PROTOCOL,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    digest = _sha256(model_path)
    payload = pickle.loads(Path(model_path).read_bytes())
    models = payload.get("models")
    support = payload.get("target_support")
    guard = payload.get("guard")
    embedded = dict(certificate.get("model_artifact", {}))
    selected_groups = tuple(str(value) for value in certificate["selected_group_ids"])
    gate = {
        "file_nonempty": model_path.is_file() and model_path.stat().st_size > 0,
        "sha256_matches_certificate": digest == str(embedded.get("sha256", "")),
        "size_matches_certificate": model_path.stat().st_size
        == int(embedded.get("size_bytes", -1)),
        "protocol_matches": payload.get("protocol") == expected_model_protocol,
        "city_matches": payload.get("city") == city,
        "scenarios_match": tuple(payload.get("scenarios", ()))
        == tuple(certificate.get("scenarios", ())),
        "anchor_matches": payload.get("anchor_policy") == ANCHOR_POLICY,
        "selected_candidate_matches": payload.get("selected_candidate")
        == certificate.get("selected_candidate"),
        "selected_groups_match": tuple(payload.get("selected_group_ids", ()))
        == selected_groups,
        "fold_assignment_matches": payload.get("fold_assignment")
        == certificate.get("fold_assignment", {}).get("assignments"),
        "fitted_models_type": isinstance(models, OfflineScreeningModels),
        "family_set_matches": isinstance(models, OfflineScreeningModels)
        and set(models.family_models) == set(BASE_FAMILIES),
        "prior_matches_anchor": isinstance(models, OfflineScreeningModels)
        and str(getattr(models.prior_spec, "key", "")) == ANCHOR_POLICY,
        "control_only_objectives": isinstance(models, OfflineScreeningModels)
        and set(models.objective_modes) == set(BASE_FAMILIES)
        and all(value == "control_only" for value in models.objective_modes.values()),
        "target_support_type": isinstance(support, TargetActionSupport),
        "target_support_matches": isinstance(support, TargetActionSupport)
        and support.diagnostics()
        == certificate.get("deployment", {}).get("target_support"),
        "guard_type": isinstance(guard, ContrastGuardConfig),
        "guard_matches": isinstance(guard, ContrastGuardConfig)
        and _guard_payload(guard)
        == certificate.get("deployment", {}).get("guard"),
    }
    gate["passed"] = all(gate.values())
    return (
        {
            "path": str(model_path.resolve()),
            "sha256": digest,
            "size_bytes": model_path.stat().st_size,
            "gate": gate,
        },
        payload,
    )


def audit_estimand_aligned_freeze(
    *,
    freeze_root: Path,
    launch_manifest: Path,
    parent_joint_audit: Path,
    generation: str = "v8",
) -> dict[str, Any]:
    if generation not in {"v8", "v9"}:
        raise ValueError(f"unsupported aligned-freeze audit generation: {generation}")
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
    parent = _read_json(parent_joint_audit)
    launch_nodes = {
        str(spec["signature"]).rsplit("/", 1)[-1]: str(spec["require_node"])
        for spec in launch.get("specs", ())
    }
    launch_gate = {
        "protocol_matches": launch.get("protocol") == expected_launch_protocol,
        "submitted": bool(launch.get("submitted", False)),
        "task_count_matches": int(launch.get("task_count", -1)) == len(CITY_NODES),
        "physical_node_assignment_matches": launch_nodes == CITY_NODES,
        "parent_audit_passed": parent.get("status") == "PASS",
        "v9_parent_protocol_matches": generation == "v8"
        or parent.get("protocol")
        == "tsc-v54r50-external-v9-full-budget-joint-freeze-audit-v1",
        "parent_audit_hash_matches": launch.get("parent_joint_audit_sha256")
        == _sha256(parent_joint_audit),
    }
    launch_gate["passed"] = all(launch_gate.values())
    if not launch_gate["passed"]:
        raise ValueError("estimand-aligned launch or parent audit changed")

    cities: dict[str, Any] = {}
    errors: list[str] = []
    for city, expected_node in CITY_NODES.items():
        city_root = Path(freeze_root) / city
        certificate_path = city_root / "freeze.json"
        model_path = city_root / "model.pkl"
        if not certificate_path.is_file() or not model_path.is_file():
            errors.append(f"missing estimand-aligned artifact for {city}")
            continue
        certificate = _read_json(certificate_path)
        model_audit, model_payload = _model_gate(
            model_path=model_path,
            certificate=certificate,
            city=city,
            expected_model_protocol=expected_model_protocol,
        )

        records = list(certificate.get("oof_action_records", ()))
        selected_groups = tuple(str(value) for value in certificate.get("selected_group_ids", ()))
        selected_group_set = set(selected_groups)
        assignments = {
            str(group): int(fold)
            for group, fold in certificate.get("fold_assignment", {})
            .get("assignments", {})
            .items()
        }
        record_groups = [str(row["group_id"]) for row in records]
        record_identities = {
            (int(row["fold_index"]), str(row["scenario"]), str(row["group_id"]))
            for row in records
        }
        observed_group_seeds = {_group_seed(group) for group in selected_groups}
        observed_validation_seeds = {int(row["validation_seed"]) for row in records}

        fold_checks = []
        for fold in sorted(certificate.get("fold_results", ()), key=lambda row: int(row["fold_index"])):
            fold_index = int(fold["fold_index"])
            training = set(str(value) for value in fold.get("training_group_ids", ()))
            validation = set(str(value) for value in fold.get("validation_group_ids", ()))
            assigned = {group for group, value in assignments.items() if value == fold_index}
            fold_checks.append(
                {
                    "fold_index": fold_index,
                    "validation_seed": int(fold["validation_seed"]),
                    "training_validation_disjoint": not (training & validation),
                    "training_validation_cover_all": training | validation
                    == selected_group_set,
                    "validation_matches_assignment": validation == assigned,
                    "validation_seed_blocked": {_group_seed(group) for group in validation}
                    == {int(fold["validation_seed"])},
                    "training_excludes_validation_seed": int(fold["validation_seed"])
                    not in {_group_seed(group) for group in training},
                    "prior_matches_anchor": fold.get("model_diagnostics", {}).get(
                        "prior_policy"
                    )
                    == ANCHOR_POLICY,
                    "target_support_label_free": bool(
                        fold.get("target_support", {}).get("label_free", False)
                    ),
                }
            )
            fold_checks[-1]["passed"] = all(
                value for key, value in fold_checks[-1].items() if key not in {"fold_index", "validation_seed"}
            )

        recomputed_selection = select_target_candidate(
            certificate["oof_candidates"],
            expected_fold_count=FOLD_COUNT,
            **SELECTOR_SPECS[FROZEN_SELECTOR_KEY],
        )
        recomputed_guard, recomputed_guard_selection = select_oof_trust_region(records)
        recomputed_validated_records = accepted_oof_records(
            records, recomputed_guard
        )
        recomputed_validated_groups = [
            str(row["group_id"]) for row in recomputed_validated_records
        ]
        recomputed_validated_folds = {
            int(row["fold_index"]) for row in recomputed_validated_records
        }
        excluded = dict(certificate.get("excluded_seed_roles", {}))
        legacy_excluded_seeds = {
            int(value)
            for role in (
                "diagnostic_only",
                "offline_evaluation",
                "closed_loop_development",
                "prospective_confirmation",
                "closed_loop_validation",
            )
            for value in excluded.get(role, ())
        }
        protected_seeds = {
            int(value)
            for role in (
                "diagnostic_only",
                "offline_evaluation",
                "prospective_confirmation",
                "closed_loop_validation",
                "closed_loop_generalization_development",
            )
            for value in excluded.get(role, ())
        }
        adaptation_diagnostics = {
            int(value)
            for value in excluded.get("closed_loop_adaptation_diagnostics", ())
        }
        declared_development = {
            int(value)
            for value in excluded.get("closed_loop_development", ())
        }
        expected_policy = (
            "cfcmt_oof_validated_support_guard"
            if recomputed_guard.enabled
            else ANCHOR_POLICY
        )
        estimand = dict(certificate.get("estimand_alignment", {}))
        city_gate = {
            "certificate_protocol_matches": certificate.get("protocol")
            == expected_result_protocol,
            "city_matches": certificate.get("city") == city,
            "physical_hostname_matches": certificate.get("hostname") == expected_node,
            "source_tree_matches_launch": certificate.get("runtime", {}).get(
                "source_tree_sha256"
            )
            == launch.get("source_tree_sha256"),
            "parent_audit_hash_matches": certificate.get("parent_joint_audit_sha256")
            == _sha256(parent_joint_audit),
            "estimand_anchor_matches_everywhere": certificate.get(
                "counterfactual_rollout_anchor"
            )
            == certificate.get("deployment_fallback_anchor")
            == estimand.get("cache_behavior_policy")
            == estimand.get("cache_counterfactual_rollout_policy")
            == estimand.get("contrast_reference_policy")
            == estimand.get("deployment_fallback_policy")
            == ANCHOR_POLICY
            and bool(estimand.get("passed", False)),
            "freeze_gate_passed": bool(certificate.get("freeze_gate", {}).get("passed")),
            "two_seed_blocked_folds": len(fold_checks) == FOLD_COUNT
            and all(row["passed"] for row in fold_checks),
            "fold_protocol_matches": certificate.get("fold_assignment", {}).get(
                "protocol"
            )
            == FOLD_PROTOCOL,
            "assignments_cover_groups_once": set(assignments) == selected_group_set
            and len(assignments) == len(selected_groups) == len(selected_group_set),
            "adaptation_seeds_only": observed_group_seeds == set(ADAPTATION_SEEDS)
            and observed_validation_seeds == set(ADAPTATION_SEEDS),
            "excluded_seeds_absent": bool(
                not (
                    observed_group_seeds
                    & (
                        legacy_excluded_seeds
                        if generation == "v8"
                        else protected_seeds
                    )
                )
            ),
            "adaptation_diagnostic_overlap_declared": bool(
                generation == "v8"
                or (
                    adaptation_diagnostics == set(ADAPTATION_SEEDS)
                    and observed_group_seeds & declared_development
                    == adaptation_diagnostics
                    and declared_development - adaptation_diagnostics
                    == {
                        int(value)
                        for value in excluded.get(
                            "closed_loop_generalization_development", ()
                        )
                    }
                )
            ),
            "prospective_seed_reservation_matches": tuple(
                int(value) for value in excluded.get("prospective_confirmation", ())
            )
            == PROSPECTIVE_SEEDS,
            "one_oof_record_per_group": len(records)
            == len(record_identities)
            == len(record_groups)
            == len(set(record_groups))
            == len(selected_groups)
            and set(record_groups) == selected_group_set,
            "oof_count_matches": int(certificate.get("oof_action_record_count", -1))
            == len(records),
            "selector_recomputed_exactly": recomputed_selection
            == certificate.get("target_selection")
            and recomputed_selection.get("selected_candidate")
            == certificate.get("selected_candidate")
            == model_payload.get("selected_candidate"),
            "guard_recomputed_exactly": recomputed_guard_selection
            == certificate.get("guard_selection")
            and _guard_payload(recomputed_guard)
            == certificate.get("deployment", {}).get("guard"),
            "validated_oof_groups_recomputed_exactly": recomputed_validated_groups
            == certificate.get("deployment", {}).get(
                "validated_oof_group_ids", []
            ),
            "validated_support_fold_coverage": bool(
                not recomputed_guard.enabled
                or recomputed_validated_folds == set(range(FOLD_COUNT))
            ),
            "validated_support_operational_floor": bool(
                not recomputed_guard.enabled
                or (
                    model_payload["target_support"].diagnostics().get(
                        "minimum_total_vehicles"
                    )
                    == MIN_OPERATIONAL_TOTAL_VEHICLES
                    and not model_payload["target_support"].diagnostics().get(
                        "label_free", True
                    )
                    and model_payload["target_support"].diagnostics().get(
                        "prototype_count"
                    )
                    == len(recomputed_validated_records)
                )
            ),
            "deployment_policy_matches_guard": certificate.get("deployment", {}).get(
                "policy"
            )
            == expected_policy,
            "full_refit_anchor_matches": certificate.get("deployment_refit", {})
            .get("model_diagnostics", {})
            .get("prior_policy")
            == ANCHOR_POLICY,
            "model_gate_passed": bool(model_audit["gate"]["passed"]),
        }
        city_gate["passed"] = all(city_gate.values())
        if not city_gate["passed"]:
            errors.append(f"estimand-aligned city gate failed: {city}")
        cities[city] = {
            "certificate": {
                "path": str(certificate_path.resolve()),
                "sha256": _sha256(certificate_path),
            },
            "model": model_audit,
            "hostname": certificate.get("hostname"),
            "selected_candidate": certificate.get("selected_candidate"),
            "guard_decision": certificate.get("guard_selection", {}).get("decision"),
            "deployment": certificate.get("deployment"),
            "selected_oof_diagnostics": certificate.get("guard_selection", {}).get(
                "selected"
            ),
            "oof_action_record_count": len(records),
            "fold_checks": fold_checks,
            "gate": city_gate,
        }

    global_gate = {
        "launch_gate_passed": bool(launch_gate["passed"]),
        "all_cities_present": set(cities) == set(CITY_NODES),
        "all_city_gates_passed": bool(cities)
        and all(row["gate"]["passed"] for row in cities.values()),
        "no_errors": not errors,
    }
    global_gate["passed"] = all(global_gate.values())
    return {
        "protocol": (
            AUDIT_PROTOCOL
            if generation == "v8"
            else "tsc-v55r51-external-v9-oof-validated-support-joint-audit-v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if global_gate["passed"] else "FAIL",
        "decision": (
            DECISION if generation == "v8" else V9_DECISION
        )
        if global_gate["passed"]
        else "reject_rollout_authorization",
        "claim_boundary": (
            "the estimand-aligned model and guard are frozen using adaptation seeds "
            "only; repaired-network development replay cannot confirm effectiveness"
        ),
        "launch_manifest_sha256": _sha256(launch_manifest),
        "parent_joint_audit_sha256": _sha256(parent_joint_audit),
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "prospective_confirmation_seeds_untouched": list(PROSPECTIVE_SEEDS),
        "generation": generation,
        "launch_gate": launch_gate,
        "cities": cities,
        "errors": errors,
        "gate": global_gate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--parent-joint-audit", type=Path, required=True)
    parser.add_argument("--generation", choices=("v8", "v9"), default="v8")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite estimand-aligned audit: {args.out}")
    payload = audit_estimand_aligned_freeze(
        freeze_root=args.freeze_root,
        launch_manifest=args.launch_manifest,
        parent_joint_audit=args.parent_joint_audit,
        generation=args.generation,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({"status": payload["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
