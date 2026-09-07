#!/usr/bin/env python3
"""Audit v40 OOF shards and apply the frozen nested seven-city selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    EXPECTED_CACHE_FILE_COUNT,
    TARGET_GROUP_BUDGET,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    ANCHOR_KEY,
    CANDIDATE_KEYS,
    DEVELOPMENT_CITY_GROUPS,
)
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_development import (
    GROUPS_PER_FOLD,
    _load_protocol,
)
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_selection import (
    FOLD_COUNT,
    summarize_cross_fitted_anchored_development,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def audit_development(
    *,
    results_root: Path,
    v39b_results_root: Path,
    v39b_audit_path: Path,
    manifest_path: Path,
    protocol_spec_path: Path,
    preregistration_path: Path,
    expected_source_tree_sha256: str,
    expected_cache_sha256: str,
    expected_baseline_sha256: str,
    expected_v39b_audit_sha256: str,
) -> dict[str, Any]:
    protocol_spec = _load_protocol(protocol_spec_path)
    development = protocol_spec["development"]
    manifest = load_traffic_signal_manifest(manifest_path)
    targets = tuple(str(value) for value in development["evaluation_targets"])
    target_set = set(targets)
    source_only_exclusions = {
        str(name): int(count)
        for name, count in development["source_only_capacity_exclusions"].items()
    }
    errors: list[str] = []
    oof_by_target: dict[str, dict[str, dict[str, Any]]] = {}
    evaluation_by_target: dict[str, dict[str, dict[str, float]]] = {}
    target_city_groups: dict[str, str] = {}
    result_provenance = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        target_set | set(source_only_exclusions) == set(manifest.sumocfgs),
        "v40 capacity partition does not cover the manifest",
    )
    require(
        {manifest.city_groups[target] for target in targets}
        == set(DEVELOPMENT_CITY_GROUPS),
        "v40 evaluated city set changed",
    )
    require(
        _sha256(v39b_audit_path) == expected_v39b_audit_sha256,
        "v39b audit hash mismatch",
    )
    v39b_audit = _read_json(v39b_audit_path)
    require(v39b_audit.get("integrity_passed") is True, "v39b integrity did not pass")
    require(v39b_audit.get("status") == "FAIL", "v39b efficacy status changed")
    v39b_provenance = {
        str(row["target"]): row for row in v39b_audit.get("result_provenance", ())
    }
    require(set(v39b_provenance) == target_set, "v39b result provenance set changed")
    observed_dirs = {
        path.parent.name for path in Path(results_root).glob("*/result.json")
    }
    require(observed_dirs == target_set, "v40 OOF target partition changed")
    protocol_sha256 = _sha256(protocol_spec_path)
    manifest_sha256 = _sha256(manifest_path)

    for target in targets:
        target_root = Path(results_root) / target
        result_path = target_root / "result.json"
        shard_path = target_root / "shard_result.json"
        launch_path = target_root / "launch_manifest.json"
        for path in (result_path, shard_path, launch_path):
            require(path.is_file(), f"{target}: missing v40 {path.name}")
        v39b_result_path = Path(v39b_results_root) / target / "result.json"
        require(v39b_result_path.is_file(), f"{target}: missing v39b result")
        if not all(path.is_file() for path in (result_path, shard_path, launch_path)):
            continue
        if not v39b_result_path.is_file():
            continue
        result = _read_json(result_path)
        shard = _read_json(shard_path)
        launch = _read_json(launch_path)
        v39b_result = _read_json(v39b_result_path)
        result_sha256 = _sha256(result_path)
        v39b_sha256 = _sha256(v39b_result_path)
        require(shard.get("passed") is True, f"{target}: v40 shard did not pass")
        require(int(shard.get("returncode", -1)) == 0, f"{target}: v40 return code nonzero")
        require(
            shard.get("result_sha256") == result_sha256,
            f"{target}: v40 result hash differs from shard",
        )
        require(
            launch.get("protocol")
            == "v40r36-single-target-cross-fitted-anchored-launch-v1",
            f"{target}: v40 launch protocol changed",
        )
        require(launch.get("target") == target, f"{target}: launch target changed")
        require(
            launch.get("protocol_spec_sha256") == protocol_sha256,
            f"{target}: launch protocol hash mismatch",
        )
        require(
            launch.get("manifest_sha256") == manifest_sha256,
            f"{target}: launch manifest hash mismatch",
        )
        require(result.get("target") == target, f"{target}: result target changed")
        require(
            result.get("protocol") == "tsc-v40r36-cross-fitted-anchored-target-oof-v1",
            f"{target}: v40 result protocol changed",
        )
        require(
            result.get("protocol_spec_sha256") == protocol_sha256,
            f"{target}: v40 protocol hash mismatch",
        )
        require(
            result.get("runtime", {}).get("source_tree_sha256")
            == expected_source_tree_sha256,
            f"{target}: v40 source-tree hash mismatch",
        )
        require(
            result.get("cache_aggregate_sha256") == expected_cache_sha256
            and int(result.get("cache_file_count", -1)) == EXPECTED_CACHE_FILE_COUNT,
            f"{target}: v40 cache identity changed",
        )
        require(
            result.get("baseline_sha256") == expected_baseline_sha256,
            f"{target}: v40 baseline hash mismatch",
        )
        require(
            result.get("development_evaluation_targets") == list(targets)
            and result.get("source_only_capacity_exclusions")
            == source_only_exclusions,
            f"{target}: v40 capacity partition changed",
        )
        require(
            int(result.get("target_safe_complete_group_count", -1))
            >= TARGET_GROUP_BUDGET,
            f"{target}: v40 target capacity below B60",
        )
        selected = [str(value) for value in result.get("selected_group_ids", ())]
        require(
            len(selected) == TARGET_GROUP_BUDGET
            and len(set(selected)) == TARGET_GROUP_BUDGET,
            f"{target}: v40 selected group count changed",
        )
        fold_assignment = result.get("fold_assignment", {})
        assignments = {
            str(group): int(fold)
            for group, fold in fold_assignment.get("assignments", {}).items()
        }
        require(
            set(assignments) == set(selected)
            and set(assignments.values()) == set(range(FOLD_COUNT)),
            f"{target}: v40 fold assignment changed",
        )
        folds = result.get("fold_results", ())
        require(len(folds) == FOLD_COUNT, f"{target}: v40 fold count changed")
        validation_union: set[str] = set()
        fold_candidate_rows: list[dict[str, Any]] = []
        for fold_index, fold in enumerate(folds):
            training = [str(value) for value in fold.get("training_group_ids", ())]
            validation = [str(value) for value in fold.get("validation_group_ids", ())]
            candidates = fold.get("candidate_metrics", {})
            require(
                int(fold.get("fold_index", -1)) == fold_index,
                f"{target}: fold order changed",
            )
            require(
                len(training) == 48
                and len(set(training)) == 48
                and len(validation) == GROUPS_PER_FOLD
                and len(set(validation)) == GROUPS_PER_FOLD
                and not set(training) & set(validation)
                and set(training) | set(validation) == set(selected),
                f"{target}: fold {fold_index} train/validation leakage",
            )
            require(
                not validation_union & set(validation),
                f"{target}: OOF validation groups repeated",
            )
            validation_union.update(validation)
            require(
                set(candidates) == set(CANDIDATE_KEYS),
                f"{target}: fold {fold_index} candidate set changed",
            )
            diagnostics = fold.get("model_diagnostics", {})
            require(
                diagnostics.get("target_adaptation_protocol")
                == "explicit-complete-action-group-list-v1"
                and int(diagnostics.get("target_adaptation_groups", -1)) == 48
                and diagnostics.get("target_adaptation_group_ids") == sorted(training),
                f"{target}: fold {fold_index} explicit fit contract failed",
            )
            for candidate in CANDIDATE_KEYS:
                row = candidates.get(candidate, {})
                regret = float(row.get("mean_normalized_action_regret", float("nan")))
                alpha = float(row.get("mean_alpha", float("nan")))
                require(
                    math.isfinite(regret) and regret >= 0.0,
                    f"{target}/{fold_index}/{candidate}: invalid OOF regret",
                )
                require(
                    math.isfinite(alpha) and 0.0 <= alpha <= 1.0,
                    f"{target}/{fold_index}/{candidate}: invalid OOF alpha",
                )
                require(
                    int(row.get("group_count", -1)) == GROUPS_PER_FOLD,
                    f"{target}/{fold_index}/{candidate}: OOF group count changed",
                )
            fold_candidate_rows.append(candidates)
        require(
            validation_union == set(selected),
            f"{target}: OOF folds do not cover B60 exactly once",
        )
        oof = result.get("oof_candidates", {})
        require(
            int(result.get("oof_candidate_count", -1)) == len(CANDIDATE_KEYS)
            and set(oof) == set(CANDIDATE_KEYS),
            f"{target}: OOF summary candidate set changed",
        )
        compact_oof = {}
        for candidate in CANDIDATE_KEYS:
            expected_folds = [
                float(rows[candidate]["mean_normalized_action_regret"])
                for rows in fold_candidate_rows
            ]
            observed_folds = [
                float(value) for value in oof.get(candidate, {}).get("fold_regrets", ())
            ]
            require(
                len(observed_folds) == FOLD_COUNT
                and np.allclose(
                    observed_folds, expected_folds, rtol=0.0, atol=1e-12
                ),
                f"{target}/{candidate}: OOF summary differs from folds",
            )
            mean_alpha = float(
                oof.get(candidate, {}).get("mean_alpha", float("nan"))
            )
            require(
                math.isfinite(mean_alpha) and 0.0 <= mean_alpha <= 1.0,
                f"{target}/{candidate}: invalid OOF summary alpha",
            )
            compact_oof[candidate] = {
                "fold_regrets": observed_folds,
                "mean_alpha": mean_alpha,
            }
        oof_by_target[target] = compact_oof

        require(
            v39b_provenance.get(target, {}).get("result_sha256") == v39b_sha256,
            f"{target}: v39b result hash differs from audited provenance",
        )
        require(
            v39b_result.get("cache_aggregate_sha256") == expected_cache_sha256
            and v39b_result.get("baseline_sha256") == expected_baseline_sha256,
            f"{target}: v39b evaluation evidence identity changed",
        )
        v39b_candidates = v39b_result.get("offline_evaluation", {}).get(
            "candidates", {}
        )
        require(
            set(v39b_candidates) == set(CANDIDATE_KEYS),
            f"{target}: v39b candidate evaluation set changed",
        )
        evaluation_by_target[target] = {
            candidate: {
                "regret": float(
                    v39b_candidates[candidate]["mean_normalized_action_regret"]
                ),
                "mean_alpha": float(v39b_candidates[candidate]["mean_alpha"]),
            }
            for candidate in CANDIDATE_KEYS
        }
        target_city_groups[target] = manifest.city_groups[target]
        result_provenance.append(
            {
                "target": target,
                "oof_result_path": str(result_path.resolve()),
                "oof_result_sha256": result_sha256,
                "oof_shard_sha256": _sha256(shard_path),
                "oof_launch_sha256": _sha256(launch_path),
                "v39b_evaluation_result_path": str(v39b_result_path.resolve()),
                "v39b_evaluation_result_sha256": v39b_sha256,
            }
        )

    integrity_passed = bool(
        not errors
        and set(oof_by_target) == target_set
        and set(evaluation_by_target) == target_set
    )
    stage = (
        summarize_cross_fitted_anchored_development(
            oof_by_target=oof_by_target,
            evaluation_by_target=evaluation_by_target,
            target_city_groups=target_city_groups,
        )
        if integrity_passed
        else None
    )
    return {
        "audit": "tsc_v40r36_cross_fitted_anchored_seven_city_development",
        "integrity_passed": integrity_passed,
        "errors": errors,
        "status": "PASS" if integrity_passed and stage and stage["passed"] else "FAIL",
        "protocol_spec": {
            "path": str(protocol_spec_path.resolve()),
            "sha256": protocol_sha256,
        },
        "preregistration": {
            "path": str(preregistration_path.resolve()),
            "sha256": _sha256(preregistration_path),
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
            "scenario_count": len(manifest.sumocfgs),
            "evaluated_target_count": len(targets),
        },
        "source_tree_sha256": expected_source_tree_sha256,
        "cache_aggregate_sha256": expected_cache_sha256,
        "baseline_sha256": expected_baseline_sha256,
        "v39b_audit": {
            "path": str(v39b_audit_path.resolve()),
            "sha256": expected_v39b_audit_sha256,
        },
        "result_provenance": result_provenance,
        "development_gate": stage,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TSC v40/r36 Cross-Fitted Anchored Selector Outcome",
        "",
        "This document reports the frozen nested seven-city selector using only B60 OOF evidence for each target decision.",
        "",
        f"- integrity: `{'PASS' if payload['integrity_passed'] else 'FAIL'}`",
        f"- development gate: `{payload['status']}`",
        f"- cache SHA-256: `{payload['cache_aggregate_sha256']}`",
        f"- source tree SHA-256: `{payload['source_tree_sha256']}`",
        f"- v39b evaluation audit SHA-256: `{payload['v39b_audit']['sha256']}`",
    ]
    gate = payload.get("development_gate")
    if gate is not None:
        global_selection = gate["global_selection"]
        global_summary = global_selection["selected_summary"]
        lines.extend(
            [
                "",
                "## Frozen gate",
                "",
                f"- LOCO anchor macro regret: `{gate['loco']['anchor_macro_regret']:.9f}`",
                f"- LOCO selected macro regret: `{gate['loco']['selected_macro_regret']:.9f}`",
                f"- LOCO relative improvement: `{gate['loco']['macro_relative_improvement_pct']:.4f}%`",
                f"- LOCO improved cities: `{gate['loco']['improved_city_count']}/7`",
                f"- LOCO maximum city regression: `{gate['loco']['maximum_absolute_city_regression']:.9f}`",
                f"- global selector: `{global_selection['selected_selector']}`",
                f"- global relative improvement: `{100.0 * global_summary['macro_relative_improvement']:.4f}%`",
                f"- global improved cities: `{global_summary['improved_city_count']}/7`",
                f"- decision: `{gate['decision']}`",
                "",
                "| Held-out city | Selector | Anchor regret | Selected regret | Improvement | Non-anchor targets |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in gate["loco"]["rows"]:
            lines.append(
                f"| {row['heldout_city']} | {row['selected_selector']} | "
                f"{row['anchor_regret']:.6f} | {row['selected_regret']:.6f} | "
                f"{row['improvement']:.6f} | "
                f"{row['selected_non_anchor_target_count']} |"
            )
        lines.extend(
            [
                "",
                "## Global target decisions",
                "",
                "| Target | City | Candidate | Anchor regret | Selected regret |",
                "|---|---|---|---:|---:|",
            ]
        )
        decisions = global_selection["selected_application"]["target_decisions"]
        for target, row in decisions.items():
            lines.append(
                f"| {target} | {row['city_group']} | {row['selected_candidate']} | "
                f"{row['anchor_evaluation_regret']:.6f} | {row['evaluation_regret']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This is target-simulator adaptation with 60 complete matched action groups and nested development-city selection. It is not zero-shot, passive real-world adaptation, or external-city confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--v39b-results-root", type=Path, required=True)
    parser.add_argument("--v39b-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-source-tree-sha256", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-v39b-audit-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_development(
        results_root=args.results_root,
        v39b_results_root=args.v39b_results_root,
        v39b_audit_path=args.v39b_audit,
        manifest_path=args.manifest,
        protocol_spec_path=args.protocol_spec,
        preregistration_path=args.preregistration,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        expected_cache_sha256=args.expected_cache_sha256,
        expected_baseline_sha256=args.expected_baseline_sha256,
        expected_v39b_audit_sha256=args.expected_v39b_audit_sha256,
    )
    _atomic_json(args.out, payload)
    _atomic_text(args.md_out, render_markdown(payload))
    if not payload["integrity_passed"]:
        raise RuntimeError("v40 cross-fitted anchored integrity audit failed")
    print(
        f"v40 cross-fitted anchored {payload['status']}: "
        f"LOCO={payload['development_gate']['loco']['macro_relative_improvement_pct']:.4f}%"
    )


if __name__ == "__main__":
    main()
