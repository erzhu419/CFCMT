#!/usr/bin/env python3
"""Fail-closed audit for the frozen v34/r30 target-simulator curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_target_curve import (
    CURVE_FAMILIES,
    TARGET_ADAPTATION_GROUPS,
    TARGET_GROUP_BUDGETS,
    summarize_pairwise_target_curve,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _target_adaptation_split_v3,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import (
    TARGETS,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_ranker import (
    GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL,
    PAIRWISE_CAUSAL_ACTION_PARENTS,
    PAIRWISE_CAUSAL_STATE_PARENTS,
)
from cf_h2o.traffic_signal.action_scaling import (
    GROUP_ACTION_REGRET_SCALE_PROTOCOL,
)
from cf_h2o.traffic_signal.benchmark_manifest import (
    load_traffic_signal_manifest,
)
from scripts.cluster.audit_tsc_policy_consistent_stage_a import (
    _atomic_json,
    _atomic_text,
    _cache_identity,
    _family_metric,
    _read_json,
    _sha256,
    _validate_source_rule_baseline,
)
from scripts.cluster.audit_tsc_selective_residual_stage import (
    _validate_reference,
)


def _flag_value(command: Sequence[str], flag: str) -> str:
    values = [str(value) for value in command]
    if values.count(flag) != 1:
        raise ValueError(f"child command does not contain exactly one {flag}")
    index = values.index(flag)
    if index + 1 >= len(values):
        raise ValueError(f"child command has no value for {flag}")
    return values[index + 1]


def _validate_pairwise_fit(
    row: Mapping[str, Any],
    *,
    expected_domain_count: int,
    source: Path,
) -> dict[str, Any]:
    diagnostics = row["model_diagnostics"]
    family_fit = diagnostics["fit"].get(PAIRWISE_PREFERENCE_FAMILY)
    if not isinstance(family_fit, dict):
        raise ValueError(f"{source}: missing pairwise fit diagnostics")
    expected = {
        "target_name": "interval_cost",
        "estimator": "HistGradientBoostingRegressor",
        "causal": True,
        "preference_protocol": "all_action_antisymmetric_pairwise_v1",
        "target_normalization_protocol": (
            GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
        ),
        "target_scale_protocol": GROUP_ACTION_REGRET_SCALE_PROTOCOL,
        "uses_context_features": False,
        "antisymmetric_augmentation": True,
    }
    for key, value in expected.items():
        if family_fit.get(key) != value:
            raise ValueError(f"{source}: pairwise diagnostic {key} changed")
    if tuple(family_fit.get("state_feature_names", ())) != tuple(
        PAIRWISE_CAUSAL_STATE_PARENTS
    ):
        raise ValueError(f"{source}: pairwise state parent set changed")
    if tuple(family_fit.get("action_feature_names", ())) != tuple(
        PAIRWISE_CAUSAL_ACTION_PARENTS
    ):
        raise ValueError(f"{source}: pairwise action parent set changed")
    if int(family_fit.get("source_domain_count", -1)) != expected_domain_count:
        raise ValueError(f"{source}: pairwise fit domain count changed")
    unordered = int(family_fit.get("unordered_pair_count", -1))
    oriented = int(family_fit.get("oriented_pair_count", -1))
    if unordered <= 0 or oriented != 2 * unordered:
        raise ValueError(f"{source}: antisymmetric pair accounting failed")
    if float(family_fit.get("pair_target_minimum", -2.0)) < -1.0 - 1e-12:
        raise ValueError(f"{source}: pair target below normalized support")
    if float(family_fit.get("pair_target_maximum", 2.0)) > 1.0 + 1e-12:
        raise ValueError(f"{source}: pair target above normalized support")
    return {
        "source_domain_count": expected_domain_count,
        "action_group_count": int(family_fit.get("action_group_count", -1)),
        "unordered_pair_count": unordered,
        "oriented_pair_count": oriented,
    }


def _expected_target_splits(
    *,
    local_cache_root: Path,
    manifest_path: Path,
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, int]]:
    manifest = load_traffic_signal_manifest(manifest_path)
    bank, _ = load_frozen_counterfactual_bank(
        local_cache_root,
        manifest,
        seeds=(2027, 3037, 4047),
        collection_shards=16,
        workers=32,
    )
    splits: dict[tuple[int, str], dict[str, Any]] = {}
    totals: dict[str, int] = {}
    for target in TARGETS:
        dataset = bank[target]
        totals[target] = len(
            set(str(value) for value in dataset.metadata["action_group_ids"])
        )
        for budget in TARGET_GROUP_BUDGETS:
            split = _target_adaptation_split_v3(
                dataset,
                group_budget=budget,
                selection_seed=20260803,
                calibration_seeds=(4047,),
                calibration_fraction=0.4,
            )
            selected = tuple(
                sorted(
                    str(value)
                    for value in split["selected"].metadata[
                        "target_adaptation_selected_group_ids"
                    ]
                )
            )
            adaptation_count = int(
                split["adaptation"].metadata["target_role_groups"]
            )
            calibration_count = int(
                split["calibration"].metadata["target_role_groups"]
            )
            adaptation_group_ids = tuple(
                sorted(
                    set(
                        str(value)
                        for value in split["adaptation"].metadata[
                            "action_group_ids"
                        ]
                    )
                )
            )
            calibration_group_ids = tuple(
                sorted(
                    set(
                        str(value)
                        for value in split["calibration"].metadata[
                            "action_group_ids"
                        ]
                    )
                )
            )
            if len(selected) != budget:
                raise ValueError(
                    f"{target}: budget {budget} cannot select {budget} groups"
                )
            if adaptation_count != TARGET_ADAPTATION_GROUPS[budget]:
                raise ValueError(
                    f"{target}: budget {budget} adaptation split changed"
                )
            if adaptation_count + calibration_count != budget:
                raise ValueError(f"{target}: target split accounting failed")
            splits[(budget, target)] = {
                "selected_group_ids": selected,
                "adaptation_group_count": adaptation_count,
                "calibration_group_count": calibration_count,
                "adaptation_group_ids": adaptation_group_ids,
                "calibration_group_ids": calibration_group_ids,
            }
    return splits, totals


def audit(
    *,
    component_root: Path,
    manifest_path: Path,
    snapshot_manifest: Path,
    reference_audit: Path,
    reference_audit_sha256: str,
    local_cache_root: Path,
    expected_cache_root: str,
    cache_aggregate_sha256: str,
    source_rule_baseline_sha256: str,
    source_rule_cache_aggregate_sha256: str,
    source_rule_generator_source_sha256: str,
    source_rule_baseline_path: Path,
) -> dict[str, Any]:
    _, reference_group_counts, reference = _validate_reference(
        reference_audit,
        expected_sha256=reference_audit_sha256,
        cache_aggregate_sha256=cache_aggregate_sha256,
        source_rule_baseline_sha256=source_rule_baseline_sha256,
    )
    manifest = load_traffic_signal_manifest(manifest_path)
    snapshot = _read_json(snapshot_manifest)
    expected_splits, cache_group_counts = _expected_target_splits(
        local_cache_root=local_cache_root,
        manifest_path=manifest_path,
    )
    if cache_group_counts != reference_group_counts:
        raise ValueError("local frozen-cache group counts differ from v26 reference")

    scenario_groups = manifest.city_groups
    all_city_groups = set(scenario_groups.values())
    expected_snapshot_root = str(snapshot["snapshot_root"])
    regrets = {
        budget: {
            family: {} for family in CURVE_FAMILIES
        }
        for budget in TARGET_GROUP_BUDGETS
    }
    integrity_rows: list[dict[str, Any]] = []
    frozen_cache_identity: dict[str, Any] | None = None
    frozen_baseline_result: str | None = None

    for budget in TARGET_GROUP_BUDGETS:
        budget_root = component_root / f"budget_{budget:03d}" / "stage_a"
        for target in TARGETS:
            target_root = budget_root / target
            expected_split = expected_splits[(budget, target)]
            city_group = scenario_groups[target]
            heldout_scenarios = {
                scenario
                for scenario, group in scenario_groups.items()
                if group == city_group
            }
            source_scenarios = set(scenario_groups) - heldout_scenarios
            fit_domains = all_city_groups - {city_group}
            if expected_split["adaptation_group_count"]:
                fit_domains = set(all_city_groups)

            shard_path = target_root / "shard_result.json"
            shard = _read_json(shard_path)
            if (
                shard.get("passed") is not True
                or shard.get("target") != target
                or int(shard.get("target_group_budget", -1)) != budget
                or shard.get("failures")
                or tuple(shard.get("model_families", ())) != CURVE_FAMILIES
            ):
                raise ValueError(f"{shard_path}: shard identity or status changed")
            completed = shard.get("completed")
            if not isinstance(completed, list) or len(completed) != len(CURVE_FAMILIES):
                raise ValueError(f"{shard_path}: incomplete family result set")
            completed_by_family = {str(row["family"]): row for row in completed}
            if set(completed_by_family) != set(CURVE_FAMILIES):
                raise ValueError(f"{shard_path}: completed family set changed")

            launch_path = target_root / "launch_manifest.json"
            launch = _read_json(launch_path)
            if (
                launch.get("target") != target
                or int(launch.get("target_group_budget", -1)) != budget
                or tuple(launch.get("model_families", ())) != CURVE_FAMILIES
                or launch.get("baseline_sha256") != source_rule_baseline_sha256
            ):
                raise ValueError(f"{launch_path}: launch identity changed")
            children = launch.get("children")
            if not isinstance(children, list) or len(children) != len(CURVE_FAMILIES):
                raise ValueError(f"{launch_path}: child launch set incomplete")
            for child in children:
                command = child.get("command")
                if not isinstance(command, list):
                    raise ValueError(f"{launch_path}: child command is invalid")
                if not any(
                    str(value).startswith(expected_snapshot_root) for value in command
                ):
                    raise ValueError(f"{launch_path}: child did not use snapshot")
                if int(_flag_value(command, "--target-group-budget")) != budget:
                    raise ValueError(f"{launch_path}: child budget changed")
                if _flag_value(command, "--targets") != target:
                    raise ValueError(f"{launch_path}: child target changed")

            family_excluded_ids: dict[str, tuple[str, ...]] = {}
            for family in CURVE_FAMILIES:
                result_path = target_root / family / "result.json"
                result_sha256 = _sha256(result_path)
                recorded = completed_by_family[family]
                if result_sha256 != recorded.get("result_sha256"):
                    raise ValueError(f"{result_path}: SHA-256 differs from shard")
                result = _read_json(result_path)
                if (
                    result.get("fit_protocol") != "screening"
                    or result.get("model_families") != [family]
                    or int(result.get("target_group_budget", -1)) != budget
                ):
                    raise ValueError(f"{result_path}: fit protocol changed")
                cache_audit = result.get("cache_audit", {})
                if str(cache_audit.get("cache_root")) != expected_cache_root:
                    raise ValueError(f"{result_path}: cache root changed")
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

                rows = result.get("targets")
                if not isinstance(rows, list) or len(rows) != 1:
                    raise ValueError(f"{result_path}: expected one target row")
                row = rows[0]
                if (
                    row.get("target") != target
                    or row.get("city_group") != city_group
                    or int(row.get("target_group_budget", -1)) != budget
                ):
                    raise ValueError(f"{result_path}: target identity changed")
                excluded = tuple(
                    sorted(str(value) for value in row.get("excluded_group_ids", ()))
                )
                if excluded != expected_split["selected_group_ids"]:
                    raise ValueError(f"{result_path}: selected target groups changed")
                family_excluded_ids[family] = excluded

                diagnostics = row.get("model_diagnostics")
                if not isinstance(diagnostics, dict):
                    raise ValueError(f"{result_path}: missing model diagnostics")
                if diagnostics.get("target_city_group") != city_group:
                    raise ValueError(f"{result_path}: target city group changed")
                if int(diagnostics.get("target_adaptation_groups", -1)) != int(
                    expected_split["adaptation_group_count"]
                ):
                    raise ValueError(f"{result_path}: adaptation count changed")
                if set(str(value) for value in diagnostics.get("source_domains", ())) != fit_domains:
                    raise ValueError(f"{result_path}: fit-domain set changed")
                if set(str(value) for value in diagnostics.get("heldout_city_scenarios", ())) != heldout_scenarios:
                    raise ValueError(f"{result_path}: same-city holdout changed")
                if set(str(value) for value in diagnostics.get("source_scenarios", ())) != source_scenarios:
                    raise ValueError(f"{result_path}: source scenario set changed")
                fit = diagnostics.get("fit")
                if not isinstance(fit, dict) or set(fit) != {family}:
                    raise ValueError(f"{result_path}: fitted family set changed")
                pairwise_fit = None
                if family == PAIRWISE_PREFERENCE_FAMILY:
                    pairwise_fit = _validate_pairwise_fit(
                        row,
                        expected_domain_count=len(fit_domains),
                        source=result_path,
                    )

                regret, group_count, optimal_rate = _family_metric(
                    row,
                    family=family,
                    source=result_path,
                )
                expected_group_count = reference_group_counts[target] - budget
                if group_count != expected_group_count:
                    raise ValueError(f"{result_path}: evaluation group count changed")
                if int(row["offline_evaluation"].get("excluded_group_count", -1)) != budget:
                    raise ValueError(f"{result_path}: excluded group accounting failed")
                if not math.isclose(
                    regret,
                    float(recorded["mean_normalized_action_regret"]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise ValueError(f"{result_path}: shard metric changed")
                if int(recorded.get("excluded_group_count", -1)) != budget:
                    raise ValueError(f"{shard_path}: recorded excluded count changed")
                if int(recorded.get("target_adaptation_groups", -1)) != int(
                    expected_split["adaptation_group_count"]
                ):
                    raise ValueError(f"{shard_path}: recorded adaptation count changed")
                regrets[budget][family][target] = regret
                integrity_rows.append(
                    {
                        "target_group_budget": budget,
                        "adaptation_group_count": expected_split[
                            "adaptation_group_count"
                        ],
                        "calibration_group_count": expected_split[
                            "calibration_group_count"
                        ],
                        "target": target,
                        "city_group": city_group,
                        "family": family,
                        "result_path": str(result_path),
                        "result_sha256": result_sha256,
                        "evaluation_group_count": group_count,
                        "mean_normalized_action_regret": regret,
                        "optimal_action_rate": optimal_rate,
                        "pairwise_fit": pairwise_fit,
                    }
                )
            if len(set(family_excluded_ids.values())) != 1:
                raise ValueError(
                    f"budget={budget}/{target}: families used different target groups"
                )

    source_rule_baseline = _validate_source_rule_baseline(
        source_rule_baseline_path,
        expected_sha256=source_rule_baseline_sha256,
        expected_cache_aggregate_sha256=source_rule_cache_aggregate_sha256,
        expected_source_tree_sha256=source_rule_generator_source_sha256,
    )
    source_rule_baseline["launch_path"] = frozen_baseline_result
    source_rule_baseline["verified_path"] = str(source_rule_baseline_path)
    selection = summarize_pairwise_target_curve(regrets)
    return {
        "protocol": "tsc-v34r30-pairwise-target-simulator-curve-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_passed": True,
        "stage_passed": selection["passed"],
        "decision": selection["decision"],
        "component_root": str(component_root),
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "snapshot": {
            "manifest_path": str(snapshot_manifest),
            "manifest_sha256": _sha256(snapshot_manifest),
            **snapshot,
        },
        "v26_reference": reference,
        "frozen_cache": {
            **(frozen_cache_identity or {}),
            "local_cache_root": str(local_cache_root),
            "launch_cache_root": expected_cache_root,
            "aggregate_sha256": cache_aggregate_sha256,
        },
        "source_rule_baseline": source_rule_baseline,
        "targets": list(TARGETS),
        "budgets": list(TARGET_GROUP_BUDGETS),
        "families": list(CURVE_FAMILIES),
        "full_group_counts": reference_group_counts,
        "shards_checked": len(TARGET_GROUP_BUDGETS) * len(TARGETS),
        "results_checked": len(integrity_rows),
        "integrity_rows": integrity_rows,
        "regrets": regrets,
        "selection": selection,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    selection = payload["selection"]
    lines = [
        "# TSC v34/r30 Pairwise Target-Simulator Adaptation Outcome",
        "",
        f"Generated: {payload['created_at_utc']}",
        "",
        "## Integrity",
        "",
        f"**PASS**: {payload['shards_checked']} city-budget shards and "
        f"{payload['results_checked']} independent family results passed "
        "snapshot, SHA-256, frozen-cache, complete-city holdout, deterministic "
        "target-split, same-subset, causal-parent, and action-group audits.",
        "",
        f"- snapshot SHA-256: `{payload['snapshot']['snapshot_sha256']}`",
        f"- cache aggregate SHA-256: `{payload['frozen_cache']['aggregate_sha256']}`",
        f"- v26 reference SHA-256: `{payload['v26_reference']['sha256']}`",
        "",
        "## Curve",
        "",
        "| B | Adapt | Calib | Rigid | Group-rigid | Target-only | Pairwise | Pair vs rigid | Cities | Max regression |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in payload["budgets"]:
        row = selection["budget_rows"][budget]
        macros = row["family_macro_regrets"]
        lines.append(
            f"| {budget} | {TARGET_ADAPTATION_GROUPS[budget]} | "
            f"{budget - TARGET_ADAPTATION_GROUPS[budget]} | "
            f"{macros[CURVE_FAMILIES[0]]:.6f} | "
            f"{macros[GROUP_NORMALIZED_RIGID_FAMILY]:.6f} | "
            f"{macros['causal_target_only']:.6f} | "
            f"{macros[PAIRWISE_PREFERENCE_FAMILY]:.6f} | "
            f"{row['relative_pairwise_improvement_vs_rigid_pct']:+.2f}% | "
            f"{row['pairwise_cities_improved']}/6 | "
            f"{row['pairwise_max_absolute_regression']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Decision",
            "",
            f"Stage: **{'PASS' if payload['stage_passed'] else 'FAIL'}**",
            f"Decision: `{payload['decision']}`",
            f"Pairwise budget-regret Spearman: "
            f"{selection['pairwise_budget_regret_spearman']:.6f}",
            "",
        ]
    )
    for key, passed in selection["stage_gate"].items():
        if key != "passed":
            lines.append(f"- `{key}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "Salt Lake City data were not read and cannot alter this decision.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fields = (
        "target_group_budget",
        "adaptation_group_count",
        "calibration_group_count",
        "family",
        "city_macro_mean_normalized_action_regret",
        *TARGETS,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for budget in payload["budgets"]:
            for family in payload["families"]:
                targets = payload["regrets"][budget][family]
                writer.writerow(
                    {
                        "target_group_budget": budget,
                        "adaptation_group_count": TARGET_ADAPTATION_GROUPS[budget],
                        "calibration_group_count": (
                            budget - TARGET_ADAPTATION_GROUPS[budget]
                        ),
                        "family": family,
                        "city_macro_mean_normalized_action_regret": sum(
                            targets.values()
                        )
                        / len(TARGETS),
                        **targets,
                    }
                )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--reference-audit-sha256", required=True)
    parser.add_argument("--local-cache-root", type=Path, required=True)
    parser.add_argument("--expected-cache-root", required=True)
    parser.add_argument("--cache-aggregate-sha256", required=True)
    parser.add_argument("--source-rule-baseline-sha256", required=True)
    parser.add_argument("--source-rule-cache-aggregate-sha256", required=True)
    parser.add_argument("--source-rule-generator-source-sha256", required=True)
    parser.add_argument("--source-rule-baseline", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit(
        component_root=args.component_root,
        manifest_path=args.manifest,
        snapshot_manifest=args.snapshot_manifest,
        reference_audit=args.reference_audit,
        reference_audit_sha256=args.reference_audit_sha256,
        local_cache_root=args.local_cache_root,
        expected_cache_root=args.expected_cache_root,
        cache_aggregate_sha256=args.cache_aggregate_sha256,
        source_rule_baseline_sha256=args.source_rule_baseline_sha256,
        source_rule_cache_aggregate_sha256=(
            args.source_rule_cache_aggregate_sha256
        ),
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
        f"decision={payload['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
