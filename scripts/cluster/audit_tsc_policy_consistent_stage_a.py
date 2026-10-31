#!/usr/bin/env python3
"""Fail-closed audit and frozen selection for TSC v25/r21 Stage A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_stage_a_selection import (
    EVALUATION_FAMILIES,
    RIGID_FAMILY,
    TARGETS,
    select_stage_a_family,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


SOURCE_RULE_BASELINE_PROTOCOL = "tsc-audited-source-rule-baseline-v1"


CACHE_IDENTITY_FIELDS = (
    "protocol",
    "collection_shards",
    "file_count",
    "used_file_count",
    "unique_identities",
    "scenario_count",
    "seeds",
    "groups_by_scenario",
    "rows_by_scenario",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _cache_identity(cache_audit: Mapping[str, Any]) -> dict[str, Any]:
    return {field: cache_audit.get(field) for field in CACHE_IDENTITY_FIELDS}


def _validate_source_rule_baseline(
    path: Path,
    *,
    expected_sha256: str,
    expected_cache_aggregate_sha256: str,
    expected_source_tree_sha256: str,
) -> dict[str, Any]:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{path}: source-rule baseline SHA-256 mismatch")
    payload = _read_json(path)
    if payload.get("protocol") != SOURCE_RULE_BASELINE_PROTOCOL:
        raise ValueError(f"{path}: unexpected source-rule baseline protocol")
    cache = payload.get("source_rule_cache")
    if not isinstance(cache, dict):
        raise ValueError(f"{path}: missing source-rule cache provenance")
    expected = {
        "aggregate_sha256": expected_cache_aggregate_sha256,
        "source_tree_sha256": expected_source_tree_sha256,
        "cache_file_count": 992,
        "scenario_count": 16,
        "policy_count": 31,
        "row_count": 992,
    }
    for field, value in expected.items():
        if cache.get(field) != value:
            raise ValueError(
                f"{path}: source-rule baseline {field} mismatch; "
                f"expected={value!r}, actual={cache.get(field)!r}"
            )
    costs = payload.get("source_rule_policy_costs")
    if not isinstance(costs, dict) or len(costs) != 16:
        raise ValueError(f"{path}: source-rule cost scenario count mismatch")
    if any(not isinstance(values, dict) or len(values) != 31 for values in costs.values()):
        raise ValueError(f"{path}: source-rule policy cost count mismatch")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        **expected,
    }


def _family_metric(
    row: Mapping[str, Any], *, family: str, source: Path
) -> tuple[float, int, float]:
    evaluation = row.get("offline_evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"{source}: missing offline evaluation")
    families = evaluation.get("families")
    family_row = families.get(family) if isinstance(families, dict) else None
    if not isinstance(family_row, dict):
        raise ValueError(f"{source}: missing family {family}")
    group_count = int(evaluation.get("evaluation_group_count", -1))
    if group_count <= 0 or int(family_row.get("group_count", -1)) != group_count:
        raise ValueError(f"{source}: action-group accounting mismatch for {family}")
    regret = float(family_row["mean_normalized_action_regret"])
    optimal_rate = float(family_row["optimal_action_rate"])
    if not math.isfinite(regret) or regret < 0.0:
        raise ValueError(f"{source}: invalid regret for {family}")
    if not math.isfinite(optimal_rate) or not 0.0 <= optimal_rate <= 1.0:
        raise ValueError(f"{source}: invalid optimal-action rate for {family}")
    return regret, group_count, optimal_rate


def _validate_holdout(
    row: Mapping[str, Any],
    *,
    target: str,
    expected_city_group: str,
    expected_heldout_scenarios: set[str],
    expected_source_domains: set[str],
    source: Path,
) -> dict[str, Any]:
    if row.get("target") != target or row.get("city_group") != expected_city_group:
        raise ValueError(f"{source}: target/city identity mismatch")
    if int(row.get("target_group_budget", -1)) != 0 or row.get("excluded_group_ids"):
        raise ValueError(f"{source}: target labels entered the zero-shot fit")
    diagnostics = row.get("model_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"{source}: missing model diagnostics")
    if diagnostics.get("target_city_group") != expected_city_group:
        raise ValueError(f"{source}: target city-group diagnostic mismatch")
    if int(diagnostics.get("target_adaptation_groups", -1)) != 0:
        raise ValueError(f"{source}: target adaptation groups are nonzero")
    source_domains = {str(value) for value in diagnostics.get("source_domains", ())}
    if source_domains != expected_source_domains:
        raise ValueError(
            f"{source}: source-domain holdout mismatch; got={sorted(source_domains)}"
        )
    if expected_city_group in source_domains:
        raise ValueError(f"{source}: target city entered source domains")
    heldout = {str(value) for value in diagnostics.get("heldout_city_scenarios", ())}
    if heldout != expected_heldout_scenarios:
        raise ValueError(
            f"{source}: incomplete same-city holdout; got={sorted(heldout)}"
        )
    source_scenarios = {str(value) for value in diagnostics.get("source_scenarios", ())}
    if source_scenarios & expected_heldout_scenarios:
        raise ValueError(f"{source}: held-out scenario entered source fit")
    return {
        "source_domains": sorted(source_domains),
        "source_scenario_count": len(source_scenarios),
        "heldout_city_scenarios": sorted(heldout),
        "target_adaptation_groups": 0,
    }


def audit(
    *,
    component_root: Path,
    manifest_path: Path,
    snapshot_manifest: Path,
    expected_cache_root: str,
    cache_aggregate_sha256: str,
    source_rule_baseline_sha256: str,
    source_rule_cache_aggregate_sha256: str,
    source_rule_baseline_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_traffic_signal_manifest(manifest_path)
    snapshot = _read_json(snapshot_manifest)
    if not cache_aggregate_sha256 or len(cache_aggregate_sha256) != 64:
        raise ValueError("cache aggregate SHA-256 must contain 64 hex characters")
    int(cache_aggregate_sha256, 16)

    scenario_groups = manifest.city_groups
    all_city_groups = set(scenario_groups.values())
    regrets = {family: {} for family in EVALUATION_FAMILIES}
    group_counts: dict[str, int] = {}
    integrity_rows: list[dict[str, Any]] = []
    frozen_cache_identity: dict[str, Any] | None = None
    frozen_baseline_result: str | None = None

    for target in TARGETS:
        if target not in scenario_groups:
            raise ValueError(f"manifest is missing Stage-A target {target}")
        city_group = scenario_groups[target]
        heldout_scenarios = {
            scenario
            for scenario, group in scenario_groups.items()
            if group == city_group
        }
        source_domains = all_city_groups - {city_group}
        shard_path = component_root / target / "shard_result.json"
        shard = _read_json(shard_path)
        if shard.get("target") != target or shard.get("passed") is not True:
            raise ValueError(f"{shard_path}: shard did not pass")
        if shard.get("failures") or tuple(shard.get("model_families", ())) != EVALUATION_FAMILIES:
            raise ValueError(f"{shard_path}: family set/failures differ from frozen protocol")
        completed = shard.get("completed")
        if not isinstance(completed, list) or len(completed) != len(EVALUATION_FAMILIES):
            raise ValueError(f"{shard_path}: incomplete family set")
        completed_by_family = {str(item.get("family")): item for item in completed}
        launch_path = component_root / target / "launch_manifest.json"
        launch = _read_json(launch_path)
        if launch.get("target") != target:
            raise ValueError(f"{launch_path}: launch target mismatch")
        if tuple(launch.get("model_families", ())) != EVALUATION_FAMILIES:
            raise ValueError(f"{launch_path}: launch family set differs from protocol")
        if launch.get("baseline_sha256") != source_rule_baseline_sha256:
            raise ValueError(f"{launch_path}: source-rule baseline SHA-256 mismatch")

        for family in EVALUATION_FAMILIES:
            recorded = completed_by_family.get(family)
            if not isinstance(recorded, dict):
                raise ValueError(f"{shard_path}: missing completed record for {family}")
            result_path = component_root / target / family / "result.json"
            result_sha256 = _sha256(result_path)
            if result_sha256 != recorded.get("result_sha256"):
                raise ValueError(f"{result_path}: SHA-256 differs from shard record")
            result = _read_json(result_path)
            if int(result.get("target_group_budget", -1)) != 0:
                raise ValueError(f"{result_path}: target budget is nonzero")
            if result.get("fit_protocol") != "screening":
                raise ValueError(f"{result_path}: non-screening fit protocol")
            if result.get("model_families") != [family]:
                raise ValueError(f"{result_path}: unexpected family payload")
            if len(result.get("targets", ())) != 1:
                raise ValueError(f"{result_path}: expected one target row")
            cache_audit = result.get("cache_audit", {})
            if str(cache_audit.get("cache_root")) != str(expected_cache_root):
                raise ValueError(f"{result_path}: unexpected cache root")
            cache_identity = _cache_identity(cache_audit)
            if frozen_cache_identity is None:
                frozen_cache_identity = cache_identity
            elif cache_identity != frozen_cache_identity:
                raise ValueError(f"{result_path}: frozen-cache identity mismatch")
            baseline_result = str(result.get("baseline_result", ""))
            if not baseline_result:
                raise ValueError(f"{result_path}: missing source-rule baseline provenance")
            if frozen_baseline_result is None:
                frozen_baseline_result = baseline_result
            elif baseline_result != frozen_baseline_result:
                raise ValueError(f"{result_path}: source-rule baseline changed")
            if str(Path(launch.get("baseline_result", "")).resolve()) != baseline_result:
                raise ValueError(f"{result_path}: launch/result source-rule baseline changed")

            row = result["targets"][0]
            holdout = _validate_holdout(
                row,
                target=target,
                expected_city_group=city_group,
                expected_heldout_scenarios=heldout_scenarios,
                expected_source_domains=source_domains,
                source=result_path,
            )
            regret, group_count, optimal_rate = _family_metric(
                row, family=family, source=result_path
            )
            if target in group_counts and group_counts[target] != group_count:
                raise ValueError(f"{result_path}: target group count changed")
            group_counts[target] = group_count
            if not math.isclose(
                regret,
                float(recorded["mean_normalized_action_regret"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"{result_path}: shard/result metric mismatch")
            regrets[family][target] = regret
            integrity_rows.append(
                {
                    "target": target,
                    "city_group": city_group,
                    "family": family,
                    "result_path": str(result_path),
                    "result_sha256": result_sha256,
                    "evaluation_group_count": group_count,
                    "mean_normalized_action_regret": regret,
                    "optimal_action_rate": optimal_rate,
                    "holdout": holdout,
                }
            )

    if frozen_baseline_result is None:
        raise ValueError("Stage A did not record a source-rule baseline")
    verified_baseline_path = Path(
        source_rule_baseline_path or frozen_baseline_result
    )
    source_rule_baseline = _validate_source_rule_baseline(
        verified_baseline_path,
        expected_sha256=source_rule_baseline_sha256,
        expected_cache_aggregate_sha256=source_rule_cache_aggregate_sha256,
        expected_source_tree_sha256=str(snapshot["source_tree_sha256"]),
    )
    source_rule_baseline["launch_path"] = frozen_baseline_result
    source_rule_baseline["verified_path"] = str(verified_baseline_path)
    selection = select_stage_a_family(regrets)
    return {
        "protocol": "tsc-v25r21-policy-consistent-stage-a-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_passed": True,
        "stage_a_passed": selection["passed"],
        "decision": selection["decision"],
        "selected_family": selection["selected_family"],
        "diagnostic_best_family": selection["diagnostic_best_family"],
        "component_root": str(component_root),
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "snapshot": {
            "manifest_path": str(snapshot_manifest),
            "manifest_sha256": _sha256(snapshot_manifest),
            **snapshot,
        },
        "frozen_cache": {
            **(frozen_cache_identity or {}),
            "cache_root": expected_cache_root,
            "aggregate_sha256": cache_aggregate_sha256,
        },
        "source_rule_baseline": source_rule_baseline,
        "targets": list(TARGETS),
        "families": list(EVALUATION_FAMILIES),
        "group_counts": group_counts,
        "shards_checked": len(TARGETS),
        "results_checked": len(integrity_rows),
        "integrity_rows": integrity_rows,
        "selection": selection,
    }


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fields = (
        "family",
        "label",
        "mechanism_count",
        "city_macro_mean_normalized_action_regret",
        "relative_improvement_vs_rigid_pct",
        "cities_improved",
        "max_absolute_regression",
        "stage_a_passed",
        *TARGETS,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        rigid = payload["selection"]["rigid"]
        writer.writerow(
            {
                "family": RIGID_FAMILY,
                "label": "Rigid",
                "mechanism_count": 0,
                "city_macro_mean_normalized_action_regret": rigid[
                    "city_macro_mean_normalized_action_regret"
                ],
                "relative_improvement_vs_rigid_pct": 0.0,
                "cities_improved": 0,
                "max_absolute_regression": 0.0,
                "stage_a_passed": False,
                **rigid["targets"],
            }
        )
        for row in payload["selection"]["family_summaries"]:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "stage_a_passed": row["stage_a_gate"]["passed"],
                    **row["targets"],
                }
            )
    temporary.replace(path)


def _markdown(payload: Mapping[str, Any]) -> str:
    selection = payload["selection"]
    rigid = selection["rigid"]
    lines = [
        "# TSC v25/r21 Policy-Consistent Stage-A Outcome",
        "",
        f"Generated: {payload['created_at_utc']}",
        "",
        "## Integrity",
        "",
        f"**PASS**: {payload['shards_checked']} shards and {payload['results_checked']} "
        "independent results passed SHA-256, complete city-group holdout, zero-target-label, "
        "finite-metric, and frozen-cache checks.",
        "",
        f"- snapshot SHA-256: `{payload['snapshot']['snapshot_sha256']}`",
        f"- source tree SHA-256: `{payload['snapshot']['source_tree_sha256']}`",
        f"- cache aggregate SHA-256: `{payload['frozen_cache']['aggregate_sha256']}`",
        f"- evaluated action groups: {sum(payload['group_counts'].values()):,}",
        "",
        "## Frozen Decision",
        "",
        f"Stage A: **{'PASS' if payload['stage_a_passed'] else 'FAIL'}**",
        f"Selected family: `{payload['selected_family'] or 'none'}`",
        f"Diagnostic best family: `{payload['diagnostic_best_family']}`",
        "",
        "| Family | Macro regret | Improvement vs rigid | Cities improved | Max regression | Gate |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Rigid | {rigid['city_macro_mean_normalized_action_regret']:.6f} | "
        "0.00% | 0/6 | 0.000000 | reference |",
    ]
    for row in selection["family_summaries"]:
        lines.append(
            f"| {row['label']} | {row['city_macro_mean_normalized_action_regret']:.6f} | "
            f"{row['relative_improvement_vs_rigid_pct']:+.2f}% | "
            f"{row['cities_improved']}/6 | {row['max_absolute_regression']:.6f} | "
            f"{'PASS' if row['stage_a_gate']['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "The selector is mechanical. Salt Lake data were not read and cannot alter this decision.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--expected-cache-root", required=True)
    parser.add_argument("--cache-aggregate-sha256", required=True)
    parser.add_argument("--source-rule-baseline-sha256", required=True)
    parser.add_argument("--source-rule-cache-aggregate-sha256", required=True)
    parser.add_argument("--source-rule-baseline", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit(
        component_root=args.component_root,
        manifest_path=args.manifest,
        snapshot_manifest=args.snapshot_manifest,
        expected_cache_root=args.expected_cache_root,
        cache_aggregate_sha256=args.cache_aggregate_sha256,
        source_rule_baseline_sha256=args.source_rule_baseline_sha256,
        source_rule_cache_aggregate_sha256=args.source_rule_cache_aggregate_sha256,
        source_rule_baseline_path=args.source_rule_baseline,
    )
    _atomic_json(args.out_json, payload)
    _write_csv(args.out_csv, payload)
    _atomic_text(args.out_md, _markdown(payload))
    print(
        "DONE "
        f"integrity=PASS stage_a={'PASS' if payload['stage_a_passed'] else 'FAIL'} "
        f"selected={payload['selected_family'] or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
