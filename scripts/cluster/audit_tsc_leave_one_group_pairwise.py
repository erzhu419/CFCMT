#!/usr/bin/env python3
"""Fail-closed audit for the frozen v37/r33 LOGO selector."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_cross_fitted_pairwise import target_group_records
from cf_h2o.eval.traffic_signal_leave_one_group_pairwise import (
    select_leave_one_group_pairwise,
    summarize_logo_deployment,
)
from cf_h2o.eval.traffic_signal_pairwise_deployment_gate import (
    TARGET_GROUP_BUDGET,
)
from cf_h2o.eval.traffic_signal_pairwise_deployment_screen import (
    DEPLOYMENT_FAMILIES,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _target_adaptation_split_v3,
)
from cf_h2o.eval.traffic_signal_rigid_residual_selection import TARGETS
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import (
    load_traffic_signal_manifest,
)
from scripts.cluster.audit_tsc_cross_fitted_pairwise import (
    _validate_explicit_fit,
    _validate_reference_audit,
)
from scripts.cluster.audit_tsc_pairwise_deployment_gate import _same_json
from scripts.cluster.audit_tsc_policy_consistent_stage_a import (
    _atomic_json,
    _atomic_text,
    _cache_identity,
    _read_json,
    _sha256,
    _validate_source_rule_baseline,
)


def audit(
    *,
    component_root: Path,
    manifest_path: Path,
    snapshot_manifest: Path,
    v34_audit_path: Path,
    v34_audit_sha256: str,
    v35_audit_path: Path,
    v35_audit_sha256: str,
    v36_audit_path: Path,
    v36_audit_sha256: str,
    local_cache_root: Path,
    expected_cache_root: str,
    cache_aggregate_sha256: str,
    source_rule_baseline_sha256: str,
    source_rule_cache_aggregate_sha256: str,
    source_rule_generator_source_sha256: str,
    source_rule_baseline_path: Path,
) -> dict[str, Any]:
    v34 = _validate_reference_audit(
        v34_audit_path,
        expected_sha256=v34_audit_sha256,
        expected_stage_passed=True,
        expected_decision="promote_pairwise_target_adaptation",
    )
    v35 = _validate_reference_audit(
        v35_audit_path,
        expected_sha256=v35_audit_sha256,
        expected_stage_passed=False,
        expected_decision="reject_calibration_only_pairwise_deployment",
    )
    v36 = _validate_reference_audit(
        v36_audit_path,
        expected_sha256=v36_audit_sha256,
        expected_stage_passed=False,
        expected_decision="reject_cross_fitted_pairwise_deployment",
    )
    for name, reference in (("v34", v34), ("v35", v35), ("v36", v36)):
        if reference.get("frozen_cache", {}).get("aggregate_sha256") != (
            cache_aggregate_sha256
        ):
            raise ValueError(f"{name} reference does not use the frozen cache")

    manifest = load_traffic_signal_manifest(manifest_path)
    snapshot = _read_json(snapshot_manifest)
    expected_snapshot_root = str(snapshot["snapshot_root"])
    bank, local_cache_audit = load_frozen_counterfactual_bank(
        local_cache_root,
        manifest,
        seeds=(2027, 3037, 4047),
        collection_shards=16,
        workers=32,
    )
    scenario_groups = manifest.city_groups
    all_city_groups = set(scenario_groups.values())
    decisions: dict[str, Mapping[str, Any]] = {}
    evaluation_regrets = {family: {} for family in DEPLOYMENT_FAMILIES}
    integrity_rows = []
    frozen_cache_identity: dict[str, Any] | None = None
    frozen_baseline_result: str | None = None
    logo_fits_checked = 0

    for target in TARGETS:
        split = _target_adaptation_split_v3(
            bank[target],
            group_budget=TARGET_GROUP_BUDGET,
            selection_seed=20260803,
            calibration_seeds=(4047,),
            calibration_fraction=0.4,
        )
        expected_selected = tuple(
            sorted(
                str(value)
                for value in split["selected"].metadata[
                    "target_adaptation_selected_group_ids"
                ]
            )
        )
        records = target_group_records(
            bank[target],
            selected_group_ids=expected_selected,
        )
        record_by_id = {str(row["group_id"]): row for row in records}
        expected_full_group_count = len(
            set(str(value) for value in bank[target].metadata["action_group_ids"])
        )
        heldout_scenarios = {
            scenario
            for scenario, group in scenario_groups.items()
            if group == scenario_groups[target]
        }
        expected_source_scenarios = set(scenario_groups) - heldout_scenarios

        target_root = component_root / target
        shard_path = target_root / "shard_result.json"
        shard = _read_json(shard_path)
        if (
            shard.get("passed") is not True
            or shard.get("target") != target
            or int(shard.get("returncode", -1)) != 0
        ):
            raise ValueError(f"{shard_path}: LOGO shard did not pass")
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
        parallelism = result.get("parallelism", {})
        if (
            result.get("protocol")
            != "tsc-v37r33-logo-pairwise-deployment-screen-v1"
            or result.get("target") != target
            or result.get("city_group") != scenario_groups[target]
            or int(result.get("target_group_budget", -1)) != 60
            or int(result.get("final_fit_group_count", -1)) != 60
            or tuple(result.get("model_families", ())) != DEPLOYMENT_FAMILIES
            or tuple(sorted(result.get("selected_group_ids", ())))
            != expected_selected
            or int(parallelism.get("logo_workers_actual", -1)) != 12
            or parallelism.get("executor") != "forked_process_pool"
        ):
            raise ValueError(f"{result_path}: LOGO result contract changed")

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

        logo_fits = result.get("logo_fits")
        logo_rows = result.get("logo_rows")
        if (
            not isinstance(logo_fits, list)
            or len(logo_fits) != 60
            or not isinstance(logo_rows, list)
            or len(logo_rows) != 60
        ):
            raise ValueError(f"{result_path}: LOGO row set incomplete")
        fit_by_group = {
            str(row.get("heldout_group_id", "")): row for row in logo_fits
        }
        row_by_group = {str(row.get("group_id", "")): row for row in logo_rows}
        if set(fit_by_group) != set(expected_selected) or set(row_by_group) != set(
            expected_selected
        ):
            raise ValueError(f"{result_path}: LOGO group identities changed")
        if len(fit_by_group) != 60 or len(row_by_group) != 60:
            raise ValueError(f"{result_path}: duplicate LOGO group identity")

        for group_id in expected_selected:
            fit_row = fit_by_group[group_id]
            logo_row = row_by_group[group_id]
            training_ids = tuple(
                value for value in expected_selected if value != group_id
            )
            if (
                tuple(fit_row.get("training_group_ids", ())) != training_ids
                or not _same_json(fit_row.get("logo_row"), logo_row)
                or int(logo_row.get("training_group_count", -1)) != 59
            ):
                raise ValueError(f"{result_path}: LOGO split changed for {group_id}")
            diagnostics = fit_row.get("model_diagnostics")
            if not isinstance(diagnostics, dict):
                raise ValueError(f"{result_path}: LOGO diagnostics missing")
            if (
                set(diagnostics.get("heldout_city_scenarios", ()))
                != heldout_scenarios
                or set(diagnostics.get("source_scenarios", ()))
                != expected_source_scenarios
            ):
                raise ValueError(f"{result_path}: LOGO city holdout changed")
            _validate_explicit_fit(
                diagnostics,
                target_group_ids=training_ids,
                expected_domains=all_city_groups,
                source=result_path,
            )
            record = record_by_id[group_id]
            family_regrets = logo_row.get("family_regrets")
            if not isinstance(family_regrets, dict) or set(family_regrets) != set(
                DEPLOYMENT_FAMILIES
            ):
                raise ValueError(f"{result_path}: LOGO family set changed")
            fallback = float(family_regrets[DEPLOYMENT_FAMILIES[0]])
            pairwise = float(family_regrets[PAIRWISE_PREFERENCE_FAMILY])
            gain = float(logo_row["paired_gain"])
            if (
                int(logo_row.get("simulator_seed", -1))
                != int(record["simulator_seed"])
                or str(logo_row.get("tls_id", "")) != str(record["tls_id"])
                or not math.isclose(
                    float(logo_row.get("snapshot_time_sec", float("nan"))),
                    float(record["snapshot_time_sec"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    gain,
                    fallback - pairwise,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(f"{result_path}: LOGO metric accounting changed")
            logo_fits_checked += 1

        recomputed_decision = select_leave_one_group_pairwise(logo_rows)
        recorded_decision = result.get("deployment_decision")
        if not _same_json(recomputed_decision, recorded_decision):
            raise ValueError(f"{result_path}: LOGO decision is not reproducible")
        if result.get("selected_family") != recomputed_decision["selected_family"]:
            raise ValueError(f"{result_path}: selected family changed")

        final_diagnostics = result.get("final_model_diagnostics")
        if not isinstance(final_diagnostics, dict):
            raise ValueError(f"{result_path}: final diagnostics missing")
        if (
            set(final_diagnostics.get("heldout_city_scenarios", ()))
            != heldout_scenarios
            or set(final_diagnostics.get("source_scenarios", ()))
            != expected_source_scenarios
        ):
            raise ValueError(f"{result_path}: final city holdout changed")
        final_fit_audit = _validate_explicit_fit(
            final_diagnostics,
            target_group_ids=expected_selected,
            expected_domains=all_city_groups,
            source=result_path,
        )

        evaluation = result.get("offline_evaluation")
        if (
            not isinstance(evaluation, dict)
            or int(evaluation.get("evaluation_group_count", -1))
            != expected_full_group_count - 60
            or int(evaluation.get("excluded_group_count", -1)) != 60
            or set(evaluation.get("families", {})) != set(DEPLOYMENT_FAMILIES)
        ):
            raise ValueError(f"{result_path}: evaluation accounting changed")
        for family in DEPLOYMENT_FAMILIES:
            metric = evaluation["families"][family]
            regret = float(metric["mean_normalized_action_regret"])
            if (
                int(metric.get("group_count", -1))
                != expected_full_group_count - 60
                or not math.isfinite(regret)
                or regret < 0.0
            ):
                raise ValueError(f"{result_path}: invalid evaluation metric")
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
                "logo_gate_passed": bool(recorded_decision["gate"]["passed"]),
                "mean_logo_paired_gain": float(
                    recorded_decision["mean_logo_paired_gain"]
                ),
                "logo_nonworse_fraction": float(
                    recorded_decision["logo_nonworse_fraction"]
                ),
                "worst_seed_gain": float(recorded_decision["worst_seed_gain"]),
                "fallback_regret": evaluation_regrets[DEPLOYMENT_FAMILIES[0]][
                    target
                ],
                "pairwise_regret": evaluation_regrets[
                    PAIRWISE_PREFERENCE_FAMILY
                ][target],
                "final_fit_audit": final_fit_audit,
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
    selection = summarize_logo_deployment(
        decisions=decisions,
        evaluation_regrets=evaluation_regrets,
    )
    return {
        "protocol": "tsc-v37r33-logo-pairwise-audit-v1",
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
        },
        "v35_reference": {
            "path": str(v35_audit_path),
            "sha256": v35_audit_sha256,
            "stage_passed": False,
        },
        "v36_reference": {
            "path": str(v36_audit_path),
            "sha256": v36_audit_sha256,
            "stage_passed": False,
        },
        "frozen_cache": {
            **(frozen_cache_identity or {}),
            "local_cache_root": str(local_cache_root),
            "launch_cache_root": expected_cache_root,
            "aggregate_sha256": cache_aggregate_sha256,
            "local_load_audit": local_cache_audit,
        },
        "source_rule_baseline": source_rule_baseline,
        "targets": list(TARGETS),
        "families": list(DEPLOYMENT_FAMILIES),
        "shards_checked": len(TARGETS),
        "results_checked": len(integrity_rows),
        "logo_fits_checked": logo_fits_checked,
        "integrity_rows": integrity_rows,
        "decisions": decisions,
        "evaluation_regrets": evaluation_regrets,
        "selection": selection,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# TSC v37/r33 Leave-One-Group-Out Pairwise Gate Outcome",
        "",
        f"Generated: {payload['created_at_utc']}",
        "",
        "## Integrity",
        "",
        f"**PASS**: {payload['shards_checked']} shards, "
        f"{payload['logo_fits_checked']} independent 59-group fits, and "
        f"{payload['results_checked']} final results passed snapshot, SHA-256, "
        "LOGO identity, explicit-fit, causal-parent, complete-city holdout, "
        "decision reproduction, and external evaluation audits.",
        "",
        f"- snapshot SHA-256: `{payload['snapshot']['snapshot_sha256']}`",
        f"- v34 audit SHA-256: `{payload['v34_reference']['sha256']}`",
        f"- v35 negative audit SHA-256: `{payload['v35_reference']['sha256']}`",
        f"- v36 negative audit SHA-256: `{payload['v36_reference']['sha256']}`",
        f"- cache aggregate SHA-256: `{payload['frozen_cache']['aggregate_sha256']}`",
        "",
        "## Decisions",
        "",
        "| Target | LOGO gain | Non-worse | Worst seed | Selected | Fallback regret | Pairwise regret |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["integrity_rows"]:
        lines.append(
            f"| {row['target']} | {row['mean_logo_paired_gain']:.6f} | "
            f"{row['logo_nonworse_fraction']:.3f} | "
            f"{row['worst_seed_gain']:.6f} | {row['selected_family']} | "
            f"{row['fallback_regret']:.6f} | {row['pairwise_regret']:.6f} |"
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
        "logo_gate_passed",
        "mean_logo_paired_gain",
        "logo_nonworse_fraction",
        "worst_seed_gain",
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
    parser.add_argument("--v35-audit", type=Path, required=True)
    parser.add_argument("--v35-audit-sha256", required=True)
    parser.add_argument("--v36-audit", type=Path, required=True)
    parser.add_argument("--v36-audit-sha256", required=True)
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
        v35_audit_path=args.v35_audit,
        v35_audit_sha256=args.v35_audit_sha256,
        v36_audit_path=args.v36_audit,
        v36_audit_sha256=args.v36_audit_sha256,
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
