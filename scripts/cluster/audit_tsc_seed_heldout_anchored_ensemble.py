#!/usr/bin/env python3
"""Audit v41 seed-heldout ensemble shards and apply the frozen city gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    EXPECTED_CACHE_FILE_COUNT,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    CANDIDATE_KEYS,
    DEVELOPMENT_CITY_GROUPS,
)
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_selection import (
    summarize_cross_fitted_anchored_development,
)
from cf_h2o.eval.traffic_signal_seed_heldout_anchored_development import (
    ADAPTATION_SEEDS,
    EVALUATION_SEEDS,
    FOLD_COUNT,
    GROUPS_PER_FOLD,
    TARGET_GROUP_BUDGET,
    TRAINING_GROUPS_PER_FOLD,
    _load_protocol,
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


def _group_seed(group: str) -> int | None:
    match = re.search(r":seed(-?\d+):", str(group))
    return int(match.group(1)) if match else None


def audit_development(
    *,
    results_root: Path,
    v40_audit_path: Path,
    manifest_path: Path,
    protocol_spec_path: Path,
    preregistration_path: Path,
    expected_source_tree_sha256: str,
    expected_cache_sha256: str,
    expected_baseline_sha256: str,
    expected_v40_audit_sha256: str,
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
        "v41 capacity partition does not cover the manifest",
    )
    require(
        {manifest.city_groups[target] for target in targets}
        == set(DEVELOPMENT_CITY_GROUPS),
        "v41 evaluated city set changed",
    )
    require(
        _sha256(v40_audit_path) == expected_v40_audit_sha256,
        "v40 audit hash mismatch",
    )
    v40_audit = _read_json(v40_audit_path)
    require(v40_audit.get("integrity_passed") is True, "v40 integrity did not pass")
    require(v40_audit.get("status") == "FAIL", "v40 efficacy status changed")
    observed_dirs = {
        path.parent.name for path in Path(results_root).glob("*/result.json")
    }
    require(observed_dirs == target_set, "v41 result target partition changed")
    protocol_sha256 = _sha256(protocol_spec_path)
    manifest_sha256 = _sha256(manifest_path)

    for target in targets:
        target_root = Path(results_root) / target
        result_path = target_root / "result.json"
        shard_path = target_root / "shard_result.json"
        launch_path = target_root / "launch_manifest.json"
        for path in (result_path, shard_path, launch_path):
            require(path.is_file(), f"{target}: missing v41 {path.name}")
        if not all(path.is_file() for path in (result_path, shard_path, launch_path)):
            continue
        result = _read_json(result_path)
        shard = _read_json(shard_path)
        launch = _read_json(launch_path)
        result_sha256 = _sha256(result_path)
        require(shard.get("passed") is True, f"{target}: v41 shard did not pass")
        require(int(shard.get("returncode", -1)) == 0, f"{target}: v41 return code nonzero")
        require(
            shard.get("result_sha256") == result_sha256,
            f"{target}: v41 result hash differs from shard",
        )
        require(
            launch.get("protocol")
            == "v41r37-single-target-seed-heldout-ensemble-launch-v1",
            f"{target}: v41 launch protocol changed",
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
            result.get("protocol")
            == "tsc-v41r37-seed-heldout-b100-cross-fitted-ensemble-target-v1",
            f"{target}: result protocol changed",
        )
        require(
            result.get("protocol_spec_sha256") == protocol_sha256,
            f"{target}: result protocol hash mismatch",
        )
        require(
            result.get("runtime", {}).get("source_tree_sha256")
            == expected_source_tree_sha256,
            f"{target}: source-tree hash mismatch",
        )
        require(
            result.get("cache_aggregate_sha256") == expected_cache_sha256
            and int(result.get("cache_file_count", -1)) == EXPECTED_CACHE_FILE_COUNT,
            f"{target}: cache identity changed",
        )
        require(
            result.get("baseline_sha256") == expected_baseline_sha256,
            f"{target}: baseline hash mismatch",
        )
        require(
            result.get("development_evaluation_targets") == list(targets)
            and result.get("source_only_capacity_exclusions")
            == source_only_exclusions,
            f"{target}: capacity partition changed",
        )
        require(
            tuple(result.get("adaptation_seeds", ())) == ADAPTATION_SEEDS
            and tuple(result.get("evaluation_seeds", ())) == EVALUATION_SEEDS,
            f"{target}: seed partition changed",
        )
        selected = [str(value) for value in result.get("selected_group_ids", ())]
        evaluation_groups = [
            str(value) for value in result.get("evaluation_group_ids", ())
        ]
        require(
            len(selected) == TARGET_GROUP_BUDGET
            and len(set(selected)) == TARGET_GROUP_BUDGET,
            f"{target}: selected group count changed",
        )
        require(
            {_group_seed(group) for group in selected} <= set(ADAPTATION_SEEDS)
            and None not in {_group_seed(group) for group in selected},
            f"{target}: selected groups leave adaptation seeds",
        )
        require(
            bool(evaluation_groups)
            and len(evaluation_groups) == len(set(evaluation_groups))
            and {_group_seed(group) for group in evaluation_groups}
            == set(EVALUATION_SEEDS),
            f"{target}: evaluation groups do not exactly use held-out seeds",
        )
        require(
            not set(selected) & set(evaluation_groups),
            f"{target}: adaptation/evaluation group leakage",
        )
        fold_assignment = result.get("fold_assignment", {})
        assignments = {
            str(group): int(fold)
            for group, fold in fold_assignment.get("assignments", {}).items()
        }
        require(
            set(assignments) == set(selected)
            and set(assignments.values()) == set(range(FOLD_COUNT))
            and fold_assignment.get("fold_group_counts")
            == [GROUPS_PER_FOLD] * FOLD_COUNT,
            f"{target}: fold assignment changed",
        )
        folds = result.get("fold_results", ())
        require(len(folds) == FOLD_COUNT, f"{target}: fold count changed")
        validation_union: set[str] = set()
        fold_candidate_rows: list[dict[str, Any]] = []
        for fold_index, fold in enumerate(folds):
            training = [str(value) for value in fold.get("training_group_ids", ())]
            validation = [
                str(value) for value in fold.get("validation_group_ids", ())
            ]
            oof_candidates = fold.get("oof_candidate_metrics", {})
            heldout_candidates = fold.get("heldout_seed_candidate_metrics", {})
            require(
                int(fold.get("fold_index", -1)) == fold_index,
                f"{target}: fold order changed",
            )
            require(
                len(training) == TRAINING_GROUPS_PER_FOLD
                and len(set(training)) == TRAINING_GROUPS_PER_FOLD
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
                set(oof_candidates) == set(CANDIDATE_KEYS)
                and set(heldout_candidates) == set(CANDIDATE_KEYS),
                f"{target}: fold {fold_index} candidate set changed",
            )
            diagnostics = fold.get("model_diagnostics", {})
            require(
                diagnostics.get("target_adaptation_protocol")
                == "explicit-complete-action-group-list-v1"
                and int(diagnostics.get("target_adaptation_groups", -1))
                == TRAINING_GROUPS_PER_FOLD
                and diagnostics.get("target_adaptation_group_ids")
                == sorted(training),
                f"{target}: fold {fold_index} explicit fit contract failed",
            )
            for candidate in CANDIDATE_KEYS:
                for role, candidates, expected_count in (
                    ("oof", oof_candidates, GROUPS_PER_FOLD),
                    ("heldout", heldout_candidates, len(evaluation_groups)),
                ):
                    row = candidates.get(candidate, {})
                    regret = float(
                        row.get("mean_normalized_action_regret", float("nan"))
                    )
                    alpha = float(row.get("mean_alpha", float("nan")))
                    require(
                        math.isfinite(regret)
                        and regret >= 0.0
                        and math.isfinite(alpha)
                        and 0.0 <= alpha <= 1.0
                        and int(row.get("group_count", -1)) == expected_count,
                        f"{target}/{fold_index}/{candidate}: invalid {role} metric",
                    )
            fold_candidate_rows.append(oof_candidates)
        require(
            validation_union == set(selected),
            f"{target}: OOF folds do not cover B100 exactly once",
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
                float(value)
                for value in oof.get(candidate, {}).get("fold_regrets", ())
            ]
            mean_alpha = float(oof.get(candidate, {}).get("mean_alpha", float("nan")))
            require(
                len(observed_folds) == FOLD_COUNT
                and np.allclose(observed_folds, expected_folds, rtol=0.0, atol=1e-12),
                f"{target}/{candidate}: OOF summary differs from folds",
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
        ensemble = result.get("ensemble_evaluation", {})
        ensemble_candidates = ensemble.get("candidates", {})
        require(
            ensemble.get("protocol")
            == "anchored-pairwise-cross-fitted-score-ensemble-offline-regret-v1"
            and int(ensemble.get("model_ensemble_size", -1)) == FOLD_COUNT
            and int(ensemble.get("evaluation_group_count", -1))
            == len(evaluation_groups)
            and set(ensemble_candidates) == set(CANDIDATE_KEYS),
            f"{target}: ensemble evaluation contract changed",
        )
        evaluation_by_target[target] = {
            candidate: {
                "regret": float(
                    ensemble_candidates[candidate]["mean_normalized_action_regret"]
                ),
                "mean_alpha": float(ensemble_candidates[candidate]["mean_alpha"]),
            }
            for candidate in CANDIDATE_KEYS
        }
        target_city_groups[target] = manifest.city_groups[target]
        result_provenance.append(
            {
                "target": target,
                "result_path": str(result_path.resolve()),
                "result_sha256": result_sha256,
                "shard_sha256": _sha256(shard_path),
                "launch_sha256": _sha256(launch_path),
                "evaluation_group_count": len(evaluation_groups),
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
        "audit": "tsc_v41r37_seed_heldout_b100_cross_fitted_ensemble_development",
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
        "v40_audit": {
            "path": str(v40_audit_path.resolve()),
            "sha256": expected_v40_audit_sha256,
        },
        "result_provenance": result_provenance,
        "development_gate": stage,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TSC v41/r37 Seed-Heldout B100 Ensemble Outcome",
        "",
        "This document reports the frozen nested seven-city selector with B100 OOF selection and independent seed-4047 ensemble evaluation.",
        "",
        f"- integrity: `{'PASS' if payload['integrity_passed'] else 'FAIL'}`",
        f"- development gate: `{payload['status']}`",
        f"- cache SHA-256: `{payload['cache_aggregate_sha256']}`",
        f"- source tree SHA-256: `{payload['source_tree_sha256']}`",
        f"- v40 audit SHA-256: `{payload['v40_audit']['sha256']}`",
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
            "This is development evidence for simulator-supported target offline adaptation using 100 complete matched action groups and an independently held-out simulator seed. It is not zero-shot, passive real-world adaptation, or external-city confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--v40-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-source-tree-sha256", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-v40-audit-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_development(
        results_root=args.results_root,
        v40_audit_path=args.v40_audit,
        manifest_path=args.manifest,
        protocol_spec_path=args.protocol_spec,
        preregistration_path=args.preregistration,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        expected_cache_sha256=args.expected_cache_sha256,
        expected_baseline_sha256=args.expected_baseline_sha256,
        expected_v40_audit_sha256=args.expected_v40_audit_sha256,
    )
    _atomic_json(args.out, payload)
    _atomic_text(args.md_out, render_markdown(payload))
    if not payload["integrity_passed"]:
        raise RuntimeError("v41 seed-heldout ensemble integrity audit failed")
    print(
        f"v41 seed-heldout ensemble {payload['status']}: "
        f"LOCO={payload['development_gate']['loco']['macro_relative_improvement_pct']:.4f}%"
    )


if __name__ == "__main__":
    main()
