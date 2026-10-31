#!/usr/bin/env python3
"""Audit all v39 shards and apply the frozen seven-city LOCO gate."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    BASE_FAMILIES,
    EXPECTED_CACHE_FILE_COUNT,
    TARGET_GROUP_BUDGET,
    _load_protocol,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    ANCHOR_KEY,
    CANDIDATE_KEYS,
    DEVELOPMENT_CITY_GROUPS,
    summarize_anchored_development,
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
    manifest_path: Path,
    protocol_spec_path: Path,
    preregistration_path: Path,
    expected_source_tree_sha256: str,
    expected_cache_sha256: str,
    expected_baseline_sha256: str,
) -> dict[str, Any]:
    protocol_spec = _load_protocol(protocol_spec_path)
    manifest = load_traffic_signal_manifest(manifest_path)
    full_scenario_set = set(manifest.sumocfgs)
    development = protocol_spec.get("development", {})
    configured_targets = [
        str(value) for value in development.get("evaluation_targets", ())
    ]
    configured_target_set = set(configured_targets)
    targets = tuple(
        target
        for target in manifest.sumocfgs
        if not configured_targets or target in configured_target_set
    )
    target_set = set(targets)
    source_only_exclusions = {
        str(name): int(count)
        for name, count in development.get(
            "source_only_capacity_exclusions", {}
        ).items()
    }
    errors: list[str] = []
    target_rows: list[dict[str, Any]] = []
    result_provenance: list[dict[str, Any]] = []
    network_candidates: dict[str, dict[str, dict[str, float]]] = {}

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        not configured_targets or configured_target_set == target_set,
        "configured development target set changed",
    )
    require(
        not configured_targets
        or target_set | set(source_only_exclusions) == full_scenario_set,
        "capacity-admission partition does not cover the manifest",
    )
    require(
        {manifest.city_groups[target] for target in targets}
        == set(DEVELOPMENT_CITY_GROUPS),
        "evaluated development city-group set changed",
    )
    observed_result_dirs = {
        path.parent.name for path in Path(results_root).glob("*/result.json")
    }
    require(
        observed_result_dirs == target_set,
        "development result target partition changed",
    )
    protocol_sha256 = _sha256(protocol_spec_path)
    manifest_sha256 = _sha256(manifest_path)
    for target in targets:
        target_root = Path(results_root) / target
        result_path = target_root / "result.json"
        shard_path = target_root / "shard_result.json"
        launch_path = target_root / "launch_manifest.json"
        for path in (result_path, shard_path, launch_path):
            require(path.is_file(), f"{target}: missing {path.name}")
        if not all(path.is_file() for path in (result_path, shard_path, launch_path)):
            continue
        result = _read_json(result_path)
        shard = _read_json(shard_path)
        launch = _read_json(launch_path)
        result_sha256 = _sha256(result_path)
        require(shard.get("passed") is True, f"{target}: shard did not pass")
        require(int(shard.get("returncode", -1)) == 0, f"{target}: nonzero return code")
        require(
            shard.get("result_sha256") == result_sha256,
            f"{target}: result hash differs from shard record",
        )
        require(
            launch.get("protocol")
            == "v39br35-single-target-anchored-pairwise-launch-v1",
            f"{target}: launch protocol changed",
        )
        require(launch.get("target") == target, f"{target}: launch target changed")
        require(
            launch.get("protocol_spec_sha256") == protocol_sha256,
            f"{target}: launch protocol-spec hash mismatch",
        )
        require(
            launch.get("manifest_sha256") == manifest_sha256,
            f"{target}: launch manifest hash mismatch",
        )
        require(result.get("target") == target, f"{target}: target identity changed")
        require(
            result.get("protocol")
            == "tsc-v39br35-capacity-admitted-anchored-pairwise-target-development-v1",
            f"{target}: result protocol changed",
        )
        require(
            result.get("protocol_spec_sha256") == protocol_sha256,
            f"{target}: result protocol-spec hash mismatch",
        )
        require(
            result.get("runtime", {}).get("source_tree_sha256")
            == expected_source_tree_sha256,
            f"{target}: source-tree hash mismatch",
        )
        require(
            result.get("cache_aggregate_sha256") == expected_cache_sha256,
            f"{target}: cache hash mismatch",
        )
        require(
            int(result.get("cache_file_count", -1)) == EXPECTED_CACHE_FILE_COUNT,
            f"{target}: cache file count changed",
        )
        require(
            result.get("baseline_sha256") == expected_baseline_sha256,
            f"{target}: baseline hash mismatch",
        )
        require(
            int(result.get("target_safe_complete_group_count", -1))
            >= TARGET_GROUP_BUDGET,
            f"{target}: target capacity is below B60",
        )
        require(
            result.get("development_evaluation_targets") == configured_targets,
            f"{target}: capacity-admitted target set changed",
        )
        require(
            result.get("source_only_capacity_exclusions")
            == source_only_exclusions,
            f"{target}: source-only capacity exclusions changed",
        )
        target_city = manifest.city_groups[target]
        heldout = sorted(
            scenario
            for scenario, city in manifest.city_groups.items()
            if city == target_city
        )
        sources = sorted(full_scenario_set - set(heldout))
        require(
            result.get("target_city_group") == target_city,
            f"{target}: city identity changed",
        )
        require(
            result.get("heldout_city_scenarios") == heldout,
            f"{target}: same-city holdout changed",
        )
        require(
            result.get("source_scenarios") == sources,
            f"{target}: source scenario set changed",
        )
        require(
            set(result.get("source_city_groups", ()))
            == set(DEVELOPMENT_CITY_GROUPS) - {target_city},
            f"{target}: source city-group set changed",
        )
        selected = [str(value) for value in result.get("selected_group_ids", ())]
        require(
            len(selected) == TARGET_GROUP_BUDGET
            and len(set(selected)) == TARGET_GROUP_BUDGET,
            f"{target}: target group budget changed",
        )
        require(
            result.get("model_families") == list(BASE_FAMILIES),
            f"{target}: base family set/order changed",
        )
        diagnostics = result.get("model_diagnostics", {})
        require(
            diagnostics.get("target_adaptation_protocol")
            == "explicit-complete-action-group-list-v1"
            and int(diagnostics.get("target_adaptation_groups", -1))
            == TARGET_GROUP_BUDGET
            and diagnostics.get("target_adaptation_group_ids") == selected,
            f"{target}: explicit B60 fit contract failed",
        )
        evaluation = result.get("offline_evaluation", {})
        evaluation_groups = int(evaluation.get("evaluation_group_count", -1))
        candidates = evaluation.get("candidates", {})
        require(evaluation_groups > 0, f"{target}: no evaluation groups")
        require(
            int(evaluation.get("excluded_group_count", -1))
            == TARGET_GROUP_BUDGET,
            f"{target}: adaptation groups entered evaluation",
        )
        require(
            int(evaluation.get("candidate_count", -1)) == len(CANDIDATE_KEYS)
            and set(candidates) == set(CANDIDATE_KEYS),
            f"{target}: candidate grid changed",
        )
        target_candidates: dict[str, dict[str, float]] = {}
        for candidate in CANDIDATE_KEYS:
            row = candidates.get(candidate, {})
            regret = float(row.get("mean_normalized_action_regret", float("nan")))
            mean_alpha = float(row.get("mean_alpha", float("nan")))
            require(
                math.isfinite(regret) and regret >= 0.0,
                f"{target}/{candidate}: invalid regret",
            )
            require(
                math.isfinite(mean_alpha) and 0.0 <= mean_alpha <= 1.0,
                f"{target}/{candidate}: invalid mean alpha",
            )
            require(
                int(row.get("group_count", -1)) == evaluation_groups,
                f"{target}/{candidate}: group count mismatch",
            )
            target_candidates[candidate] = {
                "regret": regret,
                "mean_alpha": mean_alpha,
            }
        network_candidates[target] = target_candidates
        target_rows.append(
            {
                "target": target,
                "city_group": target_city,
                "adaptation_group_count": len(selected),
                "evaluation_group_count": evaluation_groups,
                "safe_complete_group_count": int(
                    result.get("target_safe_complete_group_count", -1)
                ),
                "anchor_regret": target_candidates.get(ANCHOR_KEY, {}).get("regret"),
            }
        )
        result_provenance.append(
            {
                "target": target,
                "result_path": str(result_path.resolve()),
                "result_sha256": result_sha256,
                "shard_path": str(shard_path.resolve()),
                "shard_sha256": _sha256(shard_path),
                "launch_path": str(launch_path.resolve()),
                "launch_sha256": _sha256(launch_path),
            }
        )

    integrity_passed = not errors and set(network_candidates) == target_set
    city_rows: dict[str, dict[str, dict[str, float]]] = {}
    if integrity_passed:
        for city in DEVELOPMENT_CITY_GROUPS:
            city_targets = [
                target for target in targets if manifest.city_groups[target] == city
            ]
            city_rows[city] = {}
            for candidate in CANDIDATE_KEYS:
                city_rows[city][candidate] = {
                    "regret": sum(
                        network_candidates[target][candidate]["regret"]
                        for target in city_targets
                    )
                    / len(city_targets),
                    "mean_alpha": sum(
                        network_candidates[target][candidate]["mean_alpha"]
                        for target in city_targets
                    )
                    / len(city_targets),
                }
    stage = summarize_anchored_development(city_rows) if integrity_passed else None
    return {
        "audit": "tsc_v39br35_capacity_admitted_anchored_pairwise_development",
        "integrity_passed": integrity_passed,
        "errors": errors,
        "status": "PASS" if integrity_passed and stage and stage["passed"] else "FAIL",
        "protocol_spec": {
            "path": str(protocol_spec_path.resolve()),
            "sha256": protocol_sha256,
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
            "scenario_count": len(manifest.sumocfgs),
            "evaluated_target_count": len(targets),
            "source_only_capacity_exclusions": source_only_exclusions,
        },
        "preregistration": {
            "path": str(preregistration_path.resolve()),
            "sha256": _sha256(preregistration_path),
        },
        "source_tree_sha256": expected_source_tree_sha256,
        "cache_aggregate_sha256": expected_cache_sha256,
        "cache_file_count": EXPECTED_CACHE_FILE_COUNT,
        "baseline_sha256": expected_baseline_sha256,
        "target_rows": target_rows,
        "city_candidate_rows": city_rows,
        "result_provenance": result_provenance,
        "development_gate": stage,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TSC v39b/r35 Anchored Pairwise Development Outcome",
        "",
        "This document reports the frozen seven-city development gate. Network metrics are averaged within each city before city-macro selection.",
        "",
        f"- integrity: `{'PASS' if payload['integrity_passed'] else 'FAIL'}`",
        f"- development gate: `{payload['status']}`",
        f"- combined cache SHA-256: `{payload['cache_aggregate_sha256']}`",
        f"- expanded baseline SHA-256: `{payload['baseline_sha256']}`",
        f"- source tree SHA-256: `{payload['source_tree_sha256']}`",
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
                f"- global candidate: `{global_selection['selected_candidate']}`",
                f"- global relative improvement: `{100.0 * global_summary['macro_relative_improvement']:.4f}%`",
                f"- global improved cities: `{global_summary['improved_city_count']}/7`",
                f"- decision: `{gate['decision']}`",
                "",
                "| Held-out city | LOCO candidate | Anchor regret | Selected regret | Improvement |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in gate["loco"]["rows"]:
            lines.append(
                f"| {row['heldout_city']} | {row['selected_candidate']} | "
                f"{row['anchor_regret']:.6f} | {row['selected_regret']:.6f} | "
                f"{row['improvement']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            f"This is seven-city development on {payload['manifest']['evaluated_target_count']} capacity-admitted networks using 60 complete matched simulator action groups per target. Los Angeles and Nanchang remain unopened external holdouts unless this gate passes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-source-tree-sha256", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_development(
        results_root=args.results_root,
        manifest_path=args.manifest,
        protocol_spec_path=args.protocol_spec,
        preregistration_path=args.preregistration,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        expected_cache_sha256=args.expected_cache_sha256,
        expected_baseline_sha256=args.expected_baseline_sha256,
    )
    _atomic_json(args.out, payload)
    _atomic_text(args.md_out, render_markdown(payload))
    if not payload["integrity_passed"]:
        raise RuntimeError("v39 anchored-pairwise development integrity audit failed")
    print(
        f"v39 anchored-pairwise development {payload['status']}: "
        f"LOCO={payload['development_gate']['loco']['macro_relative_improvement_pct']:.4f}%"
    )


if __name__ == "__main__":
    main()
