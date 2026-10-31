#!/usr/bin/env python3
"""Fail-closed audit for the frozen v26/r22 rigid-residual screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_rigid_residual_selection import (
    EVALUATION_FAMILIES,
    RIGID_FAMILY,
    TARGETS,
    select_rigid_residual_family,
)
from cf_h2o.traffic_signal.causal_mechanism_advantage import (
    LOCAL_PHYSICAL_RESIDUAL_PROTOCOL,
    RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from scripts.cluster.audit_tsc_policy_consistent_stage_a import (
    _atomic_json,
    _atomic_text,
    _cache_identity,
    _family_metric,
    _read_json,
    _sha256,
    _validate_holdout,
    _validate_source_rule_baseline,
)


RESIDUAL_FAMILIES = tuple(
    family for family in EVALUATION_FAMILIES if family != RIGID_FAMILY
)


def _validate_residual_diagnostics(
    row: Mapping[str, Any],
    *,
    family: str,
    source: Path,
) -> dict[str, Any]:
    diagnostics = row.get("model_diagnostics")
    fit = diagnostics.get("fit") if isinstance(diagnostics, dict) else None
    family_fit = fit.get(family) if isinstance(fit, dict) else None
    if not isinstance(family_fit, dict):
        raise ValueError(f"{source}: missing fit diagnostics for {family}")
    config = family_fit.get("config")
    source_fit = family_fit.get("source")
    target_fit = family_fit.get("target")
    if not all(isinstance(value, dict) for value in (config, source_fit, target_fit)):
        raise ValueError(f"{source}: incomplete residual diagnostics for {family}")
    expected_protocol = RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
    if config.get("source_stack_protocol") != expected_protocol:
        raise ValueError(f"{source}: residual protocol missing from config")
    if source_fit.get("source_stack_protocol") != expected_protocol:
        raise ValueError(f"{source}: residual protocol missing from source fit")
    if source_fit.get("mechanism_dataset_protocol") != LOCAL_PHYSICAL_RESIDUAL_PROTOCOL:
        raise ValueError(f"{source}: physical mechanism protocol mismatch")
    if int(source_fit.get("source_domain_count", -1)) != 5:
        raise ValueError(f"{source}: expected five source city domains")
    if int(source_fit.get("pair_excluded_core_fit_count", -1)) != 10:
        raise ValueError(f"{source}: pair-excluded rigid fit count mismatch")
    if int(source_fit.get("pair_excluded_mechanism_fit_count", -1)) != 10:
        raise ValueError(f"{source}: pair-excluded mechanism fit count mismatch")
    if target_fit.get("enabled") is not False:
        raise ValueError(f"{source}: target adaptation unexpectedly enabled")
    if target_fit.get("reason") != "no_target_counterfactual_groups":
        raise ValueError(f"{source}: target zero-label reason mismatch")
    candidate = source_fit.get("selected_candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"{source}: missing nested source candidate")
    return {
        "source_stack_protocol": expected_protocol,
        "mechanism_dataset_protocol": LOCAL_PHYSICAL_RESIDUAL_PROTOCOL,
        "source_domain_count": 5,
        "pair_excluded_core_fit_count": 10,
        "pair_excluded_mechanism_fit_count": 10,
        "stack_enabled": bool(source_fit.get("stack_enabled")),
        "stack_weight": float(source_fit.get("stack_weight", 0.0)),
        "selected_alpha": float(candidate["alpha"]),
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
    source_rule_generator_source_sha256: str,
    source_rule_baseline_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_traffic_signal_manifest(manifest_path)
    snapshot = _read_json(snapshot_manifest)
    scenario_groups = manifest.city_groups
    all_city_groups = set(scenario_groups.values())
    regrets = {family: {} for family in EVALUATION_FAMILIES}
    group_counts: dict[str, int] = {}
    integrity_rows: list[dict[str, Any]] = []
    frozen_cache_identity: dict[str, Any] | None = None
    frozen_baseline_result: str | None = None
    expected_snapshot_root = str(snapshot["snapshot_root"])

    for target in TARGETS:
        if target not in scenario_groups:
            raise ValueError(f"manifest is missing rigid-residual target {target}")
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
        if shard.get("failures") or tuple(shard.get("model_families", ())) != (
            EVALUATION_FAMILIES
        ):
            raise ValueError(f"{shard_path}: family set or failures changed")
        completed = shard.get("completed")
        if not isinstance(completed, list) or len(completed) != len(
            EVALUATION_FAMILIES
        ):
            raise ValueError(f"{shard_path}: incomplete family set")
        completed_by_family = {str(item.get("family")): item for item in completed}

        launch_path = component_root / target / "launch_manifest.json"
        launch = _read_json(launch_path)
        if launch.get("target") != target:
            raise ValueError(f"{launch_path}: target mismatch")
        if tuple(launch.get("model_families", ())) != EVALUATION_FAMILIES:
            raise ValueError(f"{launch_path}: family order changed")
        if launch.get("baseline_sha256") != source_rule_baseline_sha256:
            raise ValueError(f"{launch_path}: baseline SHA-256 mismatch")
        children = launch.get("children")
        if not isinstance(children, list) or len(children) != len(
            EVALUATION_FAMILIES
        ):
            raise ValueError(f"{launch_path}: incomplete child launch set")
        for child in children:
            command = child.get("command") if isinstance(child, dict) else None
            if not isinstance(command, list) or not any(
                str(value).startswith(expected_snapshot_root) for value in command
            ):
                raise ValueError(f"{launch_path}: child did not use frozen snapshot")

        for family in EVALUATION_FAMILIES:
            recorded = completed_by_family.get(family)
            if not isinstance(recorded, dict):
                raise ValueError(f"{shard_path}: missing {family}")
            result_path = component_root / target / family / "result.json"
            result_sha256 = _sha256(result_path)
            if result_sha256 != recorded.get("result_sha256"):
                raise ValueError(f"{result_path}: SHA-256 differs from shard")
            result = _read_json(result_path)
            if result.get("model_families") != [family]:
                raise ValueError(f"{result_path}: family payload mismatch")
            if int(result.get("target_group_budget", -1)) != 0:
                raise ValueError(f"{result_path}: target budget is nonzero")
            if result.get("fit_protocol") != "screening":
                raise ValueError(f"{result_path}: deployment fit entered screen")
            if len(result.get("targets", ())) != 1:
                raise ValueError(f"{result_path}: expected one target row")
            cache_audit = result.get("cache_audit", {})
            if str(cache_audit.get("cache_root")) != str(expected_cache_root):
                raise ValueError(f"{result_path}: cache root mismatch")
            cache_identity = _cache_identity(cache_audit)
            if frozen_cache_identity is None:
                frozen_cache_identity = cache_identity
            elif cache_identity != frozen_cache_identity:
                raise ValueError(f"{result_path}: cache identity changed")
            baseline_result = str(result.get("baseline_result", ""))
            if frozen_baseline_result is None:
                frozen_baseline_result = baseline_result
            elif baseline_result != frozen_baseline_result:
                raise ValueError(f"{result_path}: baseline path changed")

            row = result["targets"][0]
            holdout = _validate_holdout(
                row,
                target=target,
                expected_city_group=city_group,
                expected_heldout_scenarios=heldout_scenarios,
                expected_source_domains=source_domains,
                source=result_path,
            )
            residual = (
                _validate_residual_diagnostics(
                    row,
                    family=family,
                    source=result_path,
                )
                if family in RESIDUAL_FAMILIES
                else None
            )
            regret, group_count, optimal_rate = _family_metric(
                row,
                family=family,
                source=result_path,
            )
            if target in group_counts and group_counts[target] != group_count:
                raise ValueError(f"{result_path}: action-group count changed")
            group_counts[target] = group_count
            if not math.isclose(
                regret,
                float(recorded["mean_normalized_action_regret"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"{result_path}: shard metric mismatch")
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
                    "residual_fit": residual,
                }
            )

    if frozen_baseline_result is None:
        raise ValueError("screen did not record a source-rule baseline")
    verified_baseline_path = Path(source_rule_baseline_path or frozen_baseline_result)
    source_rule_baseline = _validate_source_rule_baseline(
        verified_baseline_path,
        expected_sha256=source_rule_baseline_sha256,
        expected_cache_aggregate_sha256=source_rule_cache_aggregate_sha256,
        expected_source_tree_sha256=source_rule_generator_source_sha256,
    )
    source_rule_baseline["launch_path"] = frozen_baseline_result
    source_rule_baseline["verified_path"] = str(verified_baseline_path)
    selection = select_rigid_residual_family(regrets)
    return {
        "protocol": "tsc-v26r22-rigid-anchored-residual-stage-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_passed": True,
        "stage_passed": selection["passed"],
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
        "stage_passed",
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
                "stage_passed": False,
                **rigid["targets"],
            }
        )
        for row in payload["selection"]["family_summaries"]:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "stage_passed": row["stage_gate"]["passed"],
                    **row["targets"],
                }
            )
    temporary.replace(path)


def _markdown(payload: Mapping[str, Any]) -> str:
    selection = payload["selection"]
    rigid = selection["rigid"]
    lines = [
        "# TSC v26/r22 Rigid-Anchored Residual Outcome",
        "",
        f"Generated: {payload['created_at_utc']}",
        "",
        "## Integrity",
        "",
        f"**PASS**: {payload['shards_checked']} shards and "
        f"{payload['results_checked']} independent results passed SHA-256, "
        "complete city holdout, zero-target-label, residual-estimand, "
        "pair-excluded-fit, and frozen-cache checks.",
        "",
        f"- snapshot SHA-256: `{payload['snapshot']['snapshot_sha256']}`",
        f"- model source-tree SHA-256: `{payload['snapshot']['source_tree_sha256']}`",
        f"- cache aggregate SHA-256: `{payload['frozen_cache']['aggregate_sha256']}`",
        f"- evaluated action groups: {sum(payload['group_counts'].values()):,}",
        "",
        "## Frozen Decision",
        "",
        f"Stage: **{'PASS' if payload['stage_passed'] else 'FAIL'}**",
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
            f"| {row['label']} | "
            f"{row['city_macro_mean_normalized_action_regret']:.6f} | "
            f"{row['relative_improvement_vs_rigid_pct']:+.2f}% | "
            f"{row['cities_improved']}/6 | "
            f"{row['max_absolute_regression']:.6f} | "
            f"{'PASS' if row['stage_gate']['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Salt Lake data were not read and cannot alter this decision.",
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
    parser.add_argument("--source-rule-generator-source-sha256", required=True)
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
        source_rule_generator_source_sha256=(
            args.source_rule_generator_source_sha256
        ),
        source_rule_baseline_path=args.source_rule_baseline,
    )
    _atomic_json(args.out_json, payload)
    _write_csv(args.out_csv, payload)
    _atomic_text(args.out_md, _markdown(payload))
    print(
        "DONE integrity=PASS "
        f"stage={'PASS' if payload['stage_passed'] else 'FAIL'} "
        f"selected={payload['selected_family'] or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
