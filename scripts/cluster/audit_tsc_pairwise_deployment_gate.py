#!/usr/bin/env python3
"""Fail-closed audit for the frozen v35/r31 deployment gate."""

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
from cf_h2o.eval.traffic_signal_pairwise_deployment_gate import (
    TARGET_GROUP_BUDGET,
    select_target_pairwise,
    summarize_deployment_gate,
)
from cf_h2o.eval.traffic_signal_pairwise_deployment_screen import (
    DEPLOYMENT_FAMILIES,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import TARGETS
from cf_h2o.traffic_signal.benchmark_manifest import (
    load_traffic_signal_manifest,
)
from scripts.cluster.audit_tsc_pairwise_target_curve import (
    _expected_target_splits,
    _validate_pairwise_fit,
)
from scripts.cluster.audit_tsc_policy_consistent_stage_a import (
    _atomic_json,
    _atomic_text,
    _cache_identity,
    _read_json,
    _sha256,
    _validate_source_rule_baseline,
)


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
    )


def audit(
    *,
    component_root: Path,
    manifest_path: Path,
    snapshot_manifest: Path,
    v34_audit_path: Path,
    v34_audit_sha256: str,
    local_cache_root: Path,
    expected_cache_root: str,
    cache_aggregate_sha256: str,
    source_rule_baseline_sha256: str,
    source_rule_cache_aggregate_sha256: str,
    source_rule_generator_source_sha256: str,
    source_rule_baseline_path: Path,
) -> dict[str, Any]:
    if _sha256(v34_audit_path) != v34_audit_sha256:
        raise ValueError("v34 reference audit SHA-256 changed")
    v34 = _read_json(v34_audit_path)
    if (
        v34.get("integrity_passed") is not True
        or v34.get("stage_passed") is not True
        or v34.get("decision") != "promote_pairwise_target_adaptation"
        or v34.get("frozen_cache", {}).get("aggregate_sha256")
        != cache_aggregate_sha256
        or v34.get("source_rule_baseline", {}).get("sha256")
        != source_rule_baseline_sha256
    ):
        raise ValueError("v34 reference did not pass the frozen promotion gate")

    manifest = load_traffic_signal_manifest(manifest_path)
    snapshot = _read_json(snapshot_manifest)
    expected_snapshot_root = str(snapshot["snapshot_root"])
    splits, group_counts = _expected_target_splits(
        local_cache_root=local_cache_root,
        manifest_path=manifest_path,
    )
    v34_group_counts = {
        str(key): int(value) for key, value in v34["full_group_counts"].items()
    }
    if group_counts != v34_group_counts:
        raise ValueError("local cache group counts changed from v34")

    scenario_groups = manifest.city_groups
    all_city_groups = set(scenario_groups.values())
    decisions: dict[str, Mapping[str, Any]] = {}
    evaluation_regrets = {
        family: {} for family in DEPLOYMENT_FAMILIES
    }
    integrity_rows = []
    frozen_cache_identity: dict[str, Any] | None = None
    frozen_baseline_result: str | None = None

    for target in TARGETS:
        target_root = component_root / target
        shard_path = target_root / "shard_result.json"
        shard = _read_json(shard_path)
        if (
            shard.get("passed") is not True
            or shard.get("target") != target
            or int(shard.get("returncode", -1)) != 0
        ):
            raise ValueError(f"{shard_path}: deployment shard did not pass")

        launch_path = target_root / "launch_manifest.json"
        launch = _read_json(launch_path)
        if (
            launch.get("target") != target
            or launch.get("baseline_sha256") != source_rule_baseline_sha256
            or launch.get("manifest_sha256") != _sha256(manifest_path)
        ):
            raise ValueError(f"{launch_path}: launch identity changed")
        command = launch.get("command")
        if not isinstance(command, list) or not any(
            str(value).startswith(expected_snapshot_root) for value in command
        ):
            raise ValueError(f"{launch_path}: launch did not use frozen snapshot")

        result_path = target_root / "result.json"
        result_sha256 = _sha256(result_path)
        if result_sha256 != shard.get("result_sha256"):
            raise ValueError(f"{result_path}: SHA-256 differs from shard")
        result = _read_json(result_path)
        expected_split = splits[(TARGET_GROUP_BUDGET, target)]
        if (
            result.get("protocol")
            != "tsc-v35r31-calibration-only-pairwise-deployment-screen-v1"
            or result.get("target") != target
            or result.get("city_group") != scenario_groups[target]
            or int(result.get("target_group_budget", -1)) != TARGET_GROUP_BUDGET
            or int(result.get("adaptation_group_count", -1)) != 36
            or int(result.get("calibration_group_count", -1)) != 24
            or tuple(result.get("model_families", ())) != DEPLOYMENT_FAMILIES
            or tuple(sorted(result.get("selected_group_ids", ())))
            != expected_split["selected_group_ids"]
        ):
            raise ValueError(f"{result_path}: deployment result contract changed")

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

        diagnostics = result.get("model_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"{result_path}: missing model diagnostics")
        heldout = {
            scenario
            for scenario, group in scenario_groups.items()
            if group == scenario_groups[target]
        }
        if (
            set(diagnostics.get("source_domains", ())) != all_city_groups
            or int(diagnostics.get("target_adaptation_groups", -1)) != 36
            or set(diagnostics.get("heldout_city_scenarios", ())) != heldout
            or set(diagnostics.get("source_scenarios", ())) != set(scenario_groups) - heldout
            or set(diagnostics.get("fit", {})) != set(DEPLOYMENT_FAMILIES)
        ):
            raise ValueError(f"{result_path}: fit-domain holdout changed")
        pairwise_fit = _validate_pairwise_fit(
            {"model_diagnostics": diagnostics},
            expected_domain_count=len(all_city_groups),
            source=result_path,
        )

        calibration_rows = result.get("calibration_rows")
        if not isinstance(calibration_rows, list) or len(calibration_rows) != 24:
            raise ValueError(f"{result_path}: calibration row count changed")
        calibration_ids = tuple(
            sorted(str(row.get("group_id", "")) for row in calibration_rows)
        )
        if calibration_ids != expected_split["calibration_group_ids"]:
            raise ValueError(f"{result_path}: calibration group identities changed")
        for row in calibration_rows:
            family_regrets = row.get("family_regrets")
            if not isinstance(family_regrets, dict) or set(family_regrets) != set(
                DEPLOYMENT_FAMILIES
            ):
                raise ValueError(f"{result_path}: calibration family set changed")
            fallback = float(family_regrets[GROUP_NORMALIZED_RIGID_FAMILY])
            pairwise = float(family_regrets[PAIRWISE_PREFERENCE_FAMILY])
            gain = float(row["paired_gain"])
            if not all(math.isfinite(value) for value in (fallback, pairwise, gain)):
                raise ValueError(f"{result_path}: non-finite calibration metric")
            if not math.isclose(gain, fallback - pairwise, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"{result_path}: paired gain arithmetic changed")

        recomputed_decision = select_target_pairwise(calibration_rows)
        recorded_decision = result.get("deployment_decision")
        if not _same_json(recomputed_decision, recorded_decision):
            raise ValueError(f"{result_path}: calibration decision is not reproducible")
        if result.get("selected_family") != recomputed_decision["selected_family"]:
            raise ValueError(f"{result_path}: selected family changed")

        evaluation = result.get("offline_evaluation")
        if (
            not isinstance(evaluation, dict)
            or int(evaluation.get("evaluation_group_count", -1))
            != group_counts[target] - TARGET_GROUP_BUDGET
            or int(evaluation.get("excluded_group_count", -1))
            != TARGET_GROUP_BUDGET
            or set(evaluation.get("families", {})) != set(DEPLOYMENT_FAMILIES)
        ):
            raise ValueError(f"{result_path}: evaluation accounting changed")
        for family in DEPLOYMENT_FAMILIES:
            metric = evaluation["families"][family]
            regret = float(metric["mean_normalized_action_regret"])
            if (
                int(metric.get("group_count", -1))
                != group_counts[target] - TARGET_GROUP_BUDGET
                or not math.isfinite(regret)
                or regret < 0.0
            ):
                raise ValueError(f"{result_path}: invalid evaluation metric")
            v34_regret = float(v34["regrets"][str(TARGET_GROUP_BUDGET)][family][target])
            if not math.isclose(regret, v34_regret, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"{result_path}: model result differs from v34")
            evaluation_regrets[family][target] = regret
        selected_metric = evaluation["families"][result["selected_family"]]
        if not _same_json(selected_metric, result.get("selected_evaluation")):
            raise ValueError(f"{result_path}: selected evaluation changed")

        decisions[target] = recorded_decision
        integrity_rows.append(
            {
                "target": target,
                "city_group": scenario_groups[target],
                "result_path": str(result_path),
                "result_sha256": result_sha256,
                "evaluation_group_count": int(evaluation["evaluation_group_count"]),
                "selected_family": result["selected_family"],
                "calibration_gate_passed": bool(
                    recorded_decision["gate"]["passed"]
                ),
                "mean_paired_gain": float(recorded_decision["mean_paired_gain"]),
                "one_sided_95_lower_bound": float(
                    recorded_decision["one_sided_95_lower_bound"]
                ),
                "nonworse_fraction": float(
                    recorded_decision["nonworse_fraction"]
                ),
                "worst_temporal_block_gain": float(
                    recorded_decision["worst_temporal_block_gain"]
                ),
                "fallback_regret": evaluation_regrets[
                    GROUP_NORMALIZED_RIGID_FAMILY
                ][target],
                "pairwise_regret": evaluation_regrets[
                    PAIRWISE_PREFERENCE_FAMILY
                ][target],
                "pairwise_fit": pairwise_fit,
            }
        )

    source_rule_baseline = _validate_source_rule_baseline(
        source_rule_baseline_path,
        expected_sha256=source_rule_baseline_sha256,
        expected_cache_aggregate_sha256=source_rule_cache_aggregate_sha256,
        expected_source_tree_sha256=source_rule_generator_source_sha256,
    )
    source_rule_baseline["launch_path"] = frozen_baseline_result
    source_rule_baseline["verified_path"] = str(source_rule_baseline_path)
    selection = summarize_deployment_gate(
        decisions=decisions,
        evaluation_regrets=evaluation_regrets,
    )
    return {
        "protocol": "tsc-v35r31-pairwise-deployment-gate-audit-v1",
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
        "v34_reference": {
            "path": str(v34_audit_path),
            "sha256": v34_audit_sha256,
            "stage_passed": True,
            "snapshot_sha256": v34["snapshot"]["snapshot_sha256"],
        },
        "frozen_cache": {
            **(frozen_cache_identity or {}),
            "local_cache_root": str(local_cache_root),
            "launch_cache_root": expected_cache_root,
            "aggregate_sha256": cache_aggregate_sha256,
        },
        "source_rule_baseline": source_rule_baseline,
        "targets": list(TARGETS),
        "families": list(DEPLOYMENT_FAMILIES),
        "shards_checked": len(TARGETS),
        "results_checked": len(integrity_rows),
        "integrity_rows": integrity_rows,
        "decisions": decisions,
        "evaluation_regrets": evaluation_regrets,
        "selection": selection,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# TSC v35/r31 Calibration-Only Deployment Gate Outcome",
        "",
        f"Generated: {payload['created_at_utc']}",
        "",
        "## Integrity",
        "",
        f"**PASS**: {payload['shards_checked']} shards and "
        f"{payload['results_checked']} results passed snapshot, SHA-256, "
        "deterministic split, disjoint calibration, decision reproduction, "
        "same-city holdout, and exact v34 metric checks.",
        "",
        f"- snapshot SHA-256: `{payload['snapshot']['snapshot_sha256']}`",
        f"- v34 audit SHA-256: `{payload['v34_reference']['sha256']}`",
        f"- cache aggregate SHA-256: `{payload['frozen_cache']['aggregate_sha256']}`",
        "",
        "## Decisions",
        "",
        "| Target | Mean gain | 95% LCB | Non-worse | Worst block | Selected | Fallback regret | Pairwise regret |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["integrity_rows"]:
        lines.append(
            f"| {row['target']} | {row['mean_paired_gain']:.6f} | "
            f"{row['one_sided_95_lower_bound']:.6f} | "
            f"{row['nonworse_fraction']:.3f} | "
            f"{row['worst_temporal_block_gain']:.6f} | "
            f"{row['selected_family']} | {row['fallback_regret']:.6f} | "
            f"{row['pairwise_regret']:.6f} |"
        )
    selection = payload["selection"]
    lines.extend(
        [
            "",
            "## Frozen Decision",
            "",
            f"Stage: **{'PASS' if payload['stage_passed'] else 'FAIL'}**",
            f"Decision: `{payload['decision']}`",
            f"Pairwise selected: {selection['selected_pairwise_target_count']}/6",
            f"Selected-policy macro regret: {selection['selected_policy_macro_regret']:.6f}",
            f"Fallback macro regret: {selection['fallback_macro_regret']:.6f}",
            f"Relative improvement: {selection['relative_improvement_vs_fallback_pct']:+.2f}%",
            f"Maximum absolute regression: {selection['max_absolute_regression']:.6f}",
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
        "target",
        "city_group",
        "selected_family",
        "calibration_gate_passed",
        "mean_paired_gain",
        "one_sided_95_lower_bound",
        "nonworse_fraction",
        "worst_temporal_block_gain",
        "fallback_regret",
        "pairwise_regret",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["integrity_rows"]:
            writer.writerow({key: row[key] for key in fields})
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--v34-audit", type=Path, required=True)
    parser.add_argument("--v34-audit-sha256", required=True)
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
        v34_audit_path=args.v34_audit,
        v34_audit_sha256=args.v34_audit_sha256,
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
