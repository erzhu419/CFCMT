#!/usr/bin/env python3
"""Audit method and baseline artifacts before external held-out collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_baseline_freeze import (
    BENCHMARK_FAMILIES,
    MODEL_ARTIFACT_PROTOCOL as BASELINE_MODEL_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EVALUATION_SEEDS,
    FOLD_COUNT,
    FROZEN_SELECTOR_KEY,
    MODEL_ARTIFACT_PROTOCOL as METHOD_MODEL_PROTOCOL,
    TARGET_GROUP_BUDGET,
)


CITIES = ("los_angeles", "jinan")
METHOD_RESULT_PROTOCOL = "tsc-v42r38-external-city-b100-oof-freeze-v1"
BASELINE_RESULT_PROTOCOL = "tsc-v42r38-external-city-baseline-b100-freeze-v1"


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


def _model_audit(
    result: Mapping[str, Any],
    model_path: Path,
    *,
    expected_protocol: str,
    city: str,
    expected_families: Sequence[str],
) -> dict[str, Any]:
    embedded = dict(result.get("model_artifact", {}))
    digest = _sha256(model_path)
    payload = pickle.loads(Path(model_path).read_bytes())
    models = tuple(payload.get("model_ensemble", ()))
    family_sets = [sorted(model.family_models) for model in models]
    gate = {
        "file_nonempty": Path(model_path).stat().st_size > 0,
        "sha256_matches_result": digest == str(embedded.get("sha256", "")),
        "protocol_matches": payload.get("protocol") == expected_protocol,
        "city_matches": payload.get("city") == city,
        "ensemble_size_is_five": len(models) == FOLD_COUNT,
        "family_set_matches": all(
            set(values) == set(expected_families) for values in family_sets
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "path": str(Path(model_path).resolve()),
        "sha256": digest,
        "size_bytes": Path(model_path).stat().st_size,
        "payload_protocol": payload.get("protocol"),
        "family_sets": family_sets,
        "gate": gate,
    }


def audit_external_city_freeze(
    *,
    method_root: Path,
    baseline_root: Path,
    protocol_spec_path: Path,
    method_launch_manifest_path: Path,
    baseline_launch_manifest_path: Path,
    evaluation_cache_root: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_spec_path)
    method_launch = _load_json(method_launch_manifest_path)
    baseline_launch = _load_json(baseline_launch_manifest_path)
    expected_scenarios = {
        str(city): tuple(str(value) for value in scenarios)
        for city, scenarios in protocol["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    if set(expected_scenarios) != set(CITIES):
        raise ValueError("external freeze city protocol changed")

    city_rows = {}
    errors = []
    common_source_cache = set()
    common_external_cache = set()
    common_development_audit = set()
    for city in CITIES:
        method_result_path = Path(method_root) / city / "freeze.json"
        method_model_path = Path(method_root) / city / "model_ensemble.pkl"
        baseline_result_path = Path(baseline_root) / city / "freeze.json"
        baseline_model_path = (
            Path(baseline_root) / city / "baseline_model_ensemble.pkl"
        )
        for path in (
            method_result_path,
            method_model_path,
            baseline_result_path,
            baseline_model_path,
        ):
            if not path.is_file() or path.stat().st_size <= 0:
                errors.append(f"missing or empty freeze artifact: {path}")
        if errors:
            continue
        method = _load_json(method_result_path)
        baseline = _load_json(baseline_result_path)
        method_model = _model_audit(
            method,
            method_model_path,
            expected_protocol=METHOD_MODEL_PROTOCOL,
            city=city,
            expected_families=tuple(method.get("model_artifact", {}).get("base_families", ()))
            or (
                "causal_group_normalized_rigid_advantage",
                "causal_antisymmetric_pairwise_advantage",
            ),
        )
        # Method artifacts predate an explicit base-family field in model metadata.
        if not method_model["gate"]["family_set_matches"]:
            expected_method_families = {
                "causal_group_normalized_rigid_advantage",
                "causal_antisymmetric_pairwise_advantage",
            }
            method_model["gate"]["family_set_matches"] = all(
                set(values) == expected_method_families
                for values in method_model["family_sets"]
            )
            method_model["gate"]["passed"] = all(
                value
                for key, value in method_model["gate"].items()
                if key != "passed"
            )
        baseline_model = _model_audit(
            baseline,
            baseline_model_path,
            expected_protocol=BASELINE_MODEL_PROTOCOL,
            city=city,
            expected_families=BENCHMARK_FAMILIES,
        )

        method_groups = tuple(str(value) for value in method["selected_group_ids"])
        baseline_groups = tuple(str(value) for value in baseline["selected_group_ids"])
        method_folds = method["fold_assignment"]
        baseline_folds = baseline["fold_assignment"]
        observed_adaptation_seeds = {
            int(part[4:])
            for group in method_groups
            for part in group.split(":")
            if part.startswith("seed")
        }
        row_gate = {
            "method_protocol": method.get("protocol") == METHOD_RESULT_PROTOCOL,
            "baseline_protocol": baseline.get("protocol") == BASELINE_RESULT_PROTOCOL,
            "method_freeze_gate": bool(method.get("freeze_gate", {}).get("passed")),
            "baseline_freeze_gate": bool(
                baseline.get("freeze_gate", {}).get("passed")
            ),
            "city_matches": method.get("city") == baseline.get("city") == city,
            "scenarios_match_protocol": tuple(method.get("scenarios", ()))
            == tuple(baseline.get("scenarios", ()))
            == expected_scenarios[city],
            "exactly_b100_unique_groups": len(method_groups)
            == len(set(method_groups))
            == TARGET_GROUP_BUDGET,
            "method_baseline_groups_identical": method_groups == baseline_groups,
            "method_baseline_folds_identical": method_folds["assignments"]
            == baseline_folds["assignments"],
            "five_equal_folds": method_folds["fold_group_counts"]
            == baseline_folds["fold_group_counts"]
            == [TARGET_GROUP_BUDGET // FOLD_COUNT] * FOLD_COUNT,
            "adaptation_seeds_only": observed_adaptation_seeds
            == set(ADAPTATION_SEEDS),
            "evaluation_seed_absent": not (
                observed_adaptation_seeds & set(EVALUATION_SEEDS)
            ),
            "frozen_selector_matches": method.get("frozen_selector_key")
            == FROZEN_SELECTOR_KEY,
            "method_model_passed": bool(method_model["gate"]["passed"]),
            "baseline_model_passed": bool(baseline_model["gate"]["passed"]),
            "benchmark_families_match": tuple(baseline["benchmark_families"])
            == BENCHMARK_FAMILIES,
        }
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"external freeze city gate failed: {city}")
        common_source_cache.update(
            (
                str(method["source_cache_aggregate_sha256"]),
                str(baseline["source_cache_aggregate_sha256"]),
            )
        )
        common_external_cache.update(
            (
                str(method["external_cache_aggregate_sha256"]),
                str(baseline["external_cache_aggregate_sha256"]),
            )
        )
        common_development_audit.update(
            (
                str(method["development_audit_sha256"]),
                str(baseline["development_audit_sha256"]),
            )
        )
        city_rows[city] = {
            "gate": row_gate,
            "method_result": {
                "path": str(method_result_path.resolve()),
                "sha256": _sha256(method_result_path),
                "selected_candidate": method["selected_candidate"],
            },
            "method_model": method_model,
            "baseline_result": {
                "path": str(baseline_result_path.resolve()),
                "sha256": _sha256(baseline_result_path),
            },
            "baseline_model": baseline_model,
            "selected_group_ids_sha256": hashlib.sha256(
                "\n".join(method_groups).encode("utf-8")
            ).hexdigest(),
            "fold_assignment_sha256": hashlib.sha256(
                json.dumps(
                    method_folds["assignments"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }

    evaluation_root = Path(evaluation_cache_root)
    evaluation_files = (
        sorted(path for path in evaluation_root.rglob("*") if path.is_file())
        if evaluation_root.exists()
        else []
    )
    global_gate = {
        "all_cities_present": set(city_rows) == set(CITIES),
        "all_city_gates_passed": bool(city_rows)
        and all(row["gate"]["passed"] for row in city_rows.values()),
        "one_source_cache_identity": len(common_source_cache) == 1,
        "one_adaptation_cache_identity": len(common_external_cache) == 1,
        "one_v41_development_audit_identity": len(common_development_audit) == 1,
        "evaluation_cache_absent_before_joint_freeze": not evaluation_files,
        "method_launch_submitted": bool(method_launch.get("submitted")),
        "baseline_launch_submitted": bool(baseline_launch.get("submitted")),
        "no_errors": not errors,
    }
    global_gate["passed"] = all(global_gate.values())
    return {
        "protocol": "tsc-v42r38-external-method-baseline-joint-freeze-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if global_gate["passed"] else "FAIL",
        "protocol_spec_path": str(Path(protocol_spec_path).resolve()),
        "protocol_spec_sha256": _sha256(protocol_spec_path),
        "method_launch_manifest_sha256": _sha256(method_launch_manifest_path),
        "baseline_launch_manifest_sha256": _sha256(baseline_launch_manifest_path),
        "evaluation_cache_root": str(evaluation_root.resolve()),
        "evaluation_cache_file_count_at_freeze": len(evaluation_files),
        "source_cache_sha256": next(iter(common_source_cache), None),
        "adaptation_cache_sha256": next(iter(common_external_cache), None),
        "development_audit_sha256": next(iter(common_development_audit), None),
        "cities": city_rows,
        "errors": errors,
        "gate": global_gate,
        "decision": (
            "authorize_external_seed7079_collection"
            if global_gate["passed"]
            else "prohibit_external_seed7079_collection"
        ),
    }


def _write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# TSC v42/r38 external joint-freeze audit",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        "| City | Method candidate | Method model SHA-256 | Baseline model SHA-256 | Gate |",
        "|---|---|---|---|---|",
    ]
    for city in CITIES:
        row = payload.get("cities", {}).get(city, {})
        lines.append(
            "| {city} | {candidate} | `{method}` | `{baseline}` | {gate} |".format(
                city=city,
                candidate=row.get("method_result", {}).get("selected_candidate", "missing"),
                method=row.get("method_model", {}).get("sha256", "missing"),
                baseline=row.get("baseline_model", {}).get("sha256", "missing"),
                gate="PASS" if row.get("gate", {}).get("passed") else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "The held-out seed `7079` cache contained "
            f"{payload['evaluation_cache_file_count_at_freeze']} files when this audit was frozen.",
            "",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--method-launch-manifest", type=Path, required=True)
    parser.add_argument("--baseline-launch-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit_external_city_freeze(
        method_root=args.method_root,
        baseline_root=args.baseline_root,
        protocol_spec_path=args.protocol_spec,
        method_launch_manifest_path=args.method_launch_manifest,
        baseline_launch_manifest_path=args.baseline_launch_manifest,
        evaluation_cache_root=args.evaluation_cache_root,
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
