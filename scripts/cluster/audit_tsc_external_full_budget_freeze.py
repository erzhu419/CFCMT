#!/usr/bin/env python3
"""Audit the v43 full-budget method/baseline freeze before seed 8171."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import BASE_FAMILIES
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_selection import (
    SEED_BLOCKED_SELECTION_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_city_baseline_freeze import (
    BENCHMARK_FAMILIES,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    FROZEN_SELECTOR_KEY,
    _json_ready,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    BASELINE_FREEZE_PROTOCOL,
    BASELINE_MODEL_PROTOCOL,
    DIAGNOSTIC_ONLY_SEED,
    EVALUATION_SEED,
    FOLD_COUNT,
    FOLD_PROTOCOL,
    METHOD_FREEZE_PROTOCOL,
    METHOD_MODEL_PROTOCOL,
    PROTOCOL,
    REFIT_PROTOCOL,
    V9_BASELINE_FREEZE_PROTOCOL,
    V9_BASELINE_MODEL_PROTOCOL,
    V9_EVALUATION_SEED,
    V9_METHOD_FREEZE_PROTOCOL,
    V9_METHOD_MODEL_PROTOCOL,
    V9_PROTOCOL,
)

CITIES = ("los_angeles", "jinan")
DECISION = "authorize_external_seed8171_collection"
V9_DECISION = "authorize_external_v9_seed9277_collection"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_ready(value), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _model_audit(
    result: Mapping[str, Any],
    model_path: Path,
    *,
    expected_protocol: str,
    city: str,
    expected_families: Sequence[str],
    expected_groups: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    embedded = dict(result.get("model_artifact", {}))
    digest = _sha256(model_path)
    payload = pickle.loads(Path(model_path).read_bytes())
    models = tuple(payload.get("model_ensemble", ()))
    family_sets = [sorted(model.family_models) for model in models]
    groups = tuple(str(value) for value in expected_groups)
    gate = {
        "file_nonempty": Path(model_path).stat().st_size > 0,
        "sha256_matches_result": digest == str(embedded.get("sha256", "")),
        "protocol_matches": payload.get("protocol") == expected_protocol,
        "city_matches": payload.get("city") == city,
        "single_full_refit_model": len(models) == 1,
        "family_set_matches": all(
            set(values) == set(expected_families) for values in family_sets
        ),
        "selected_groups_match": tuple(payload.get("selected_group_ids", ()))
        == groups,
        "deployment_groups_match": tuple(
            payload.get("deployment_refit_group_ids", ())
        )
        == groups,
        "deployment_refit_protocol_matches": payload.get(
            "deployment_refit_protocol"
        )
        == REFIT_PROTOCOL,
    }
    gate["passed"] = all(gate.values())
    diagnostics_sha = (
        _canonical_sha256(models[0].diagnostics) if len(models) == 1 else None
    )
    return (
        {
            "path": str(Path(model_path).resolve()),
            "sha256": digest,
            "size_bytes": Path(model_path).stat().st_size,
            "payload_protocol": payload.get("protocol"),
            "family_sets": family_sets,
            "diagnostics_sha256": diagnostics_sha,
            "gate": gate,
        },
        payload,
    )


def audit_external_full_budget_freeze(
    *,
    freeze_root: Path,
    protocol_spec_path: Path,
    launch_manifest_path: Path,
    evaluation_cache_root: Path,
    generation: str = "v8",
) -> dict[str, Any]:
    if generation not in {"v8", "v9"}:
        raise ValueError(f"unsupported full-budget audit generation: {generation}")
    protocol = _load_json(protocol_spec_path)
    launch = _load_json(launch_manifest_path)
    expected_protocol = PROTOCOL if generation == "v8" else V9_PROTOCOL
    expected_method_freeze = (
        METHOD_FREEZE_PROTOCOL
        if generation == "v8"
        else V9_METHOD_FREEZE_PROTOCOL
    )
    expected_baseline_freeze = (
        BASELINE_FREEZE_PROTOCOL
        if generation == "v8"
        else V9_BASELINE_FREEZE_PROTOCOL
    )
    expected_method_model = (
        METHOD_MODEL_PROTOCOL if generation == "v8" else V9_METHOD_MODEL_PROTOCOL
    )
    expected_baseline_model = (
        BASELINE_MODEL_PROTOCOL
        if generation == "v8"
        else V9_BASELINE_MODEL_PROTOCOL
    )
    expected_evaluation_seed = (
        EVALUATION_SEED if generation == "v8" else V9_EVALUATION_SEED
    )
    if protocol.get("protocol") != expected_protocol:
        raise ValueError("full-budget protocol changed")
    expected_repair_evidence = None
    if generation == "v9":
        parent_spec = dict(protocol.get("parent_protocol", {}))
        parent_path = PROJECT_ROOT / str(parent_spec.get("path", ""))
        if _sha256(parent_path) != str(parent_spec.get("sha256", "")):
            raise ValueError("v9 repair parent protocol changed")
        parent = _load_json(parent_path)
        expected_repair_evidence = {
            "failed_v52_audit_sha256": parent[
                "post_efficacy_integrity_amendment"
            ]["failed_v52_audit"]["sha256"],
            "network_admission_audit_sha256": parent["network_admission"][
                "audit_sha256"
            ],
            "counterfactual_cache_audit_sha256": parent[
                "counterfactual_cache"
            ]["audit_sha256"],
        }
    expected_scenarios = {
        str(city): tuple(str(value) for value in scenarios)
        for city, scenarios in protocol["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    if set(expected_scenarios) != set(CITIES):
        raise ValueError("v43 full-budget city set changed")

    city_rows: dict[str, Any] = {}
    errors: list[str] = []
    source_cache_ids: set[str] = set()
    adaptation_cache_ids: set[str] = set()
    development_audit_ids: set[str] = set()
    protocol_ids: set[str] = set()
    source_tree_ids: set[str] = set()
    for city in CITIES:
        city_root = Path(freeze_root) / city
        method_result_path = city_root / "method_freeze.json"
        baseline_result_path = city_root / "baseline_freeze.json"
        method_model_path = city_root / "method_model.pkl"
        baseline_model_path = city_root / "baseline_model.pkl"
        paths = (
            method_result_path,
            baseline_result_path,
            method_model_path,
            baseline_model_path,
        )
        missing = [path for path in paths if not path.is_file() or path.stat().st_size <= 0]
        if missing:
            errors.extend(f"missing or empty freeze artifact: {path}" for path in missing)
            continue

        method = _load_json(method_result_path)
        baseline = _load_json(baseline_result_path)
        method_groups = tuple(str(value) for value in method["selected_group_ids"])
        baseline_groups = tuple(str(value) for value in baseline["selected_group_ids"])
        method_model, method_payload = _model_audit(
            method,
            method_model_path,
            expected_protocol=expected_method_model,
            city=city,
            expected_families=BASE_FAMILIES,
            expected_groups=method_groups,
        )
        baseline_model, baseline_payload = _model_audit(
            baseline,
            baseline_model_path,
            expected_protocol=expected_baseline_model,
            city=city,
            expected_families=BENCHMARK_FAMILIES,
            expected_groups=baseline_groups,
        )

        method_folds = dict(method["fold_assignment"])
        baseline_folds = dict(baseline["fold_assignment"])
        observed_seeds = {
            int(part[4:])
            for group in method_groups
            for part in group.split(":")
            if part.startswith("seed")
        }
        expected_group_count = sum(
            int(method["external_cache_audit"]["groups_by_scenario"][scenario])
            for scenario in expected_scenarios[city]
        )
        assignments = dict(method_folds.get("assignments", {}))
        seed_blocked = all(
            int(fold) == ADAPTATION_SEEDS.index(
                next(
                    seed
                    for seed in ADAPTATION_SEEDS
                    if f":seed{seed}:" in group
                )
            )
            for group, fold in assignments.items()
        )
        row_gate = {
            "method_protocol": method.get("protocol") == expected_method_freeze,
            "baseline_protocol": baseline.get("protocol")
            == expected_baseline_freeze,
            "method_freeze_gate": bool(method.get("freeze_gate", {}).get("passed")),
            "baseline_freeze_gate": bool(
                baseline.get("freeze_gate", {}).get("passed")
            ),
            "city_matches": method.get("city") == baseline.get("city") == city,
            "scenarios_match_protocol": tuple(method.get("scenarios", ()))
            == tuple(baseline.get("scenarios", ()))
            == expected_scenarios[city],
            "all_available_groups_used_once": len(method_groups)
            == len(set(method_groups))
            == expected_group_count
            == int(method["target_group_count"])
            == int(baseline["target_group_count"]),
            "method_baseline_groups_identical": method_groups == baseline_groups,
            "method_baseline_folds_identical": method_folds == baseline_folds,
            "seed_blocked_fold_protocol": method_folds.get("protocol")
            == FOLD_PROTOCOL,
            "two_complete_seed_folds": int(method_folds.get("fold_count", -1))
            == FOLD_COUNT
            and len(set(assignments.values())) == FOLD_COUNT,
            "every_group_assigned_once": set(assignments) == set(method_groups),
            "fold_assignment_matches_group_seed": seed_blocked,
            "adaptation_seeds_only": observed_seeds == set(ADAPTATION_SEEDS),
            "diagnostic_seed_absent": DIAGNOSTIC_ONLY_SEED not in observed_seeds,
            "evaluation_seed_absent": expected_evaluation_seed not in observed_seeds,
            "frozen_selector_matches": method.get("frozen_selector_key")
            == FROZEN_SELECTOR_KEY
            == method_payload.get("selector_key"),
            "seed_blocked_selector_result": method.get(
                "target_selection", {}
            ).get("protocol")
            == SEED_BLOCKED_SELECTION_PROTOCOL
            and int(method.get("target_selection", {}).get("fold_count", -1))
            == FOLD_COUNT,
            "selected_candidate_matches_model": method.get("selected_candidate")
            == method_payload.get("selected_candidate"),
            "method_model_passed": bool(method_model["gate"]["passed"]),
            "baseline_model_passed": bool(baseline_model["gate"]["passed"]),
            "shared_full_refit_diagnostics": method_model["diagnostics_sha256"]
            == baseline_model["diagnostics_sha256"],
            "benchmark_families_match": tuple(baseline["benchmark_families"])
            == BENCHMARK_FAMILIES
            == tuple(baseline_payload.get("benchmark_families", ())),
            "repair_evidence_matches": (
                method.get("repair_evidence")
                == baseline.get("repair_evidence")
                == expected_repair_evidence
            ),
        }
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"full-budget city gate failed: {city}")

        source_cache_ids.update(
            (
                str(method["source_cache_aggregate_sha256"]),
                str(baseline["source_cache_aggregate_sha256"]),
            )
        )
        adaptation_cache_ids.update(
            (
                str(method["external_cache_aggregate_sha256"]),
                str(baseline["external_cache_aggregate_sha256"]),
            )
        )
        development_audit_ids.update(
            (
                str(method["development_audit_sha256"]),
                str(baseline["development_audit_sha256"]),
            )
        )
        protocol_ids.update(
            (
                str(method["protocol_spec_sha256"]),
                str(baseline["protocol_spec_sha256"]),
            )
        )
        source_tree_ids.update(
            (
                str(method["runtime"].get("source_tree_sha256")),
                str(baseline["runtime"].get("source_tree_sha256")),
            )
        )
        city_rows[city] = {
            "gate": row_gate,
            "target_group_count": len(method_groups),
            "method_result": {
                "path": str(method_result_path.resolve()),
                "sha256": _sha256(method_result_path),
                "selected_candidate": method["selected_candidate"],
            },
            "baseline_result": {
                "path": str(baseline_result_path.resolve()),
                "sha256": _sha256(baseline_result_path),
            },
            "method_model": method_model,
            "baseline_model": baseline_model,
            "selected_group_ids_sha256": hashlib.sha256(
                "\n".join(method_groups).encode("utf-8")
            ).hexdigest(),
            "fold_assignment_sha256": _canonical_sha256(assignments),
        }

    evaluation_root = Path(evaluation_cache_root)
    evaluation_files = (
        sorted(path for path in evaluation_root.rglob("*") if path.is_file())
        if evaluation_root.exists()
        else []
    )
    expected_source_tree = str(launch.get("source_tree_sha256", ""))
    expected_launch_protocol = (
        "v43r39-external-full-budget-freeze-submission-v1"
        if generation == "v8"
        else "v54r50-external-v9-full-budget-freeze-submission-v1"
    )
    global_gate = {
        "all_cities_present": set(city_rows) == set(CITIES),
        "all_city_gates_passed": bool(city_rows)
        and all(row["gate"]["passed"] for row in city_rows.values()),
        "one_source_cache_identity": len(source_cache_ids) == 1,
        "one_adaptation_cache_identity": len(adaptation_cache_ids) == 1,
        "one_development_audit_identity": len(development_audit_ids) == 1,
        "one_protocol_identity": protocol_ids == {_sha256(protocol_spec_path)},
        "source_tree_matches_launch": source_tree_ids == {expected_source_tree},
        "evaluation_cache_absent_before_joint_freeze": not evaluation_files,
        "freeze_launch_submitted": bool(launch.get("submitted")),
        "freeze_launch_protocol_matches": launch.get("protocol")
        == expected_launch_protocol,
        "two_launch_specs": len(launch.get("specs", ())) == len(CITIES),
        "no_errors": not errors,
    }
    global_gate["passed"] = all(global_gate.values())
    return {
        "protocol": (
            "tsc-v43r39-external-full-budget-joint-freeze-audit-v1"
            if generation == "v8"
            else "tsc-v54r50-external-v9-full-budget-joint-freeze-audit-v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if global_gate["passed"] else "FAIL",
        "protocol_spec_path": str(Path(protocol_spec_path).resolve()),
        "protocol_spec_sha256": _sha256(protocol_spec_path),
        "launch_manifest_sha256": _sha256(launch_manifest_path),
        "evaluation_cache_root": str(evaluation_root.resolve()),
        "evaluation_cache_file_count_at_freeze": len(evaluation_files),
        "evaluation_seed": expected_evaluation_seed,
        "generation": generation,
        "source_cache_sha256": next(iter(source_cache_ids), None),
        "adaptation_cache_sha256": next(iter(adaptation_cache_ids), None),
        "development_audit_sha256": next(iter(development_audit_ids), None),
        "source_tree_sha256": next(iter(source_tree_ids), None),
        "cities": city_rows,
        "errors": errors,
        "gate": global_gate,
        "decision": (
            DECISION if generation == "v8" else V9_DECISION
        )
        if global_gate["passed"]
        else (
            "prohibit_external_seed8171_collection"
            if generation == "v8"
            else "prohibit_external_v9_seed9277_collection"
        ),
    }


def _write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    generation = str(payload.get("generation", "v8"))
    evaluation_seed = int(payload.get("evaluation_seed", EVALUATION_SEED))
    lines = [
        f"# TSC external full-budget joint-freeze audit ({generation})",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        "| City | Adaptation groups | Method candidate | Method model SHA-256 | Baseline model SHA-256 | Gate |",
        "|---|---:|---|---|---|---|",
    ]
    for city in CITIES:
        row = payload.get("cities", {}).get(city, {})
        lines.append(
            "| {city} | {groups} | {candidate} | `{method}` | `{baseline}` | {gate} |".format(
                city=city,
                groups=row.get("target_group_count", "missing"),
                candidate=row.get("method_result", {}).get(
                    "selected_candidate", "missing"
                ),
                method=row.get("method_model", {}).get("sha256", "missing"),
                baseline=row.get("baseline_model", {}).get("sha256", "missing"),
                gate="PASS" if row.get("gate", {}).get("passed") else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            f"The fresh seed `{evaluation_seed}` cache contained "
            f"{payload['evaluation_cache_file_count_at_freeze']} files when this audit was frozen.",
            "",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-cache-root", type=Path, required=True)
    parser.add_argument("--generation", choices=("v8", "v9"), default="v8")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit_external_full_budget_freeze(
        freeze_root=args.freeze_root,
        protocol_spec_path=args.protocol_spec,
        launch_manifest_path=args.launch_manifest,
        evaluation_cache_root=args.evaluation_cache_root,
        generation=args.generation,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(payload, args.markdown_out)
    print(json.dumps({"status": payload["status"], "out": str(args.out)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
