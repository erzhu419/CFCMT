#!/usr/bin/env python3
"""Fail-closed audit for the frozen v31/r27 group-normalized core screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
    select_group_normalized_rigid_family,
)
from cf_h2o.traffic_signal.action_ranker import (
    CAUSAL_RIGID_RANKING_PARENTS,
    GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL,
)
from cf_h2o.traffic_signal.action_scaling import GROUP_ACTION_REGRET_SCALE_PROTOCOL
from scripts.cluster.audit_tsc_direct_rollout_value_stage import (
    _audit_from_args_for_family,
)
from scripts.cluster.audit_tsc_policy_consistent_stage_a import (
    _atomic_json,
    _atomic_text,
)
from scripts.cluster.audit_tsc_rollout_value_stage import _write_csv


def _validate_group_normalized_diagnostics(
    row: Mapping[str, Any],
    *,
    source: Path,
) -> dict[str, Any]:
    diagnostics = row.get("model_diagnostics")
    fit = diagnostics.get("fit") if isinstance(diagnostics, dict) else None
    family_fit = (
        fit.get(GROUP_NORMALIZED_RIGID_FAMILY)
        if isinstance(fit, dict)
        else None
    )
    if not isinstance(family_fit, dict):
        raise ValueError(f"{source}: missing group-normalized fit diagnostics")
    source_domains = diagnostics.get("source_domains")
    if not isinstance(source_domains, list) or len(source_domains) != 5:
        raise ValueError(f"{source}: expected five source city domains")
    if int(diagnostics.get("target_adaptation_groups", -1)) != 0:
        raise ValueError(f"{source}: target labels entered the fit")
    if family_fit.get("target_name") != "interval_cost":
        raise ValueError(f"{source}: action target changed")
    if family_fit.get("estimator") != "HistGradientBoostingRegressor":
        raise ValueError(f"{source}: estimator changed")
    if family_fit.get("causal") is not True:
        raise ValueError(f"{source}: dense features entered the causal core")
    if tuple(family_fit.get("feature_names", ())) != tuple(
        CAUSAL_RIGID_RANKING_PARENTS
    ):
        raise ValueError(f"{source}: rigid causal parent set changed")
    if family_fit.get("candidate_only") is not True:
        raise ValueError(f"{source}: reference rows entered model fitting")
    sign_balance = family_fit.get("sign_balance")
    if not isinstance(sign_balance, dict) or sign_balance.get("enabled") is not True:
        raise ValueError(f"{source}: candidate sign balancing is disabled")
    if family_fit.get("target_normalization_protocol") != (
        GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
    ):
        raise ValueError(f"{source}: target normalization protocol mismatch")
    if family_fit.get("domain_target_scale_protocol") != (
        GROUP_ACTION_REGRET_SCALE_PROTOCOL
    ):
        raise ValueError(f"{source}: evaluation-regret scale protocol mismatch")
    if family_fit.get("domain_target_scales") != {}:
        raise ValueError(f"{source}: domain-level target scales remained enabled")
    scale_summary = family_fit.get("group_target_scale_summary")
    if not isinstance(scale_summary, dict) or int(scale_summary.get("group_count", 0)) <= 0:
        raise ValueError(f"{source}: missing matched-group scale audit")
    rows = int(family_fit.get("rows", -1))
    training_rows = int(family_fit.get("training_rows", -1))
    if rows <= 0 or not 0 < training_rows < rows:
        raise ValueError(f"{source}: candidate/reference row accounting failed")
    return {
        "source_domain_count": 5,
        "feature_count": int(family_fit.get("feature_count", -1)),
        "rows": rows,
        "training_rows": training_rows,
        "target_normalization_protocol": (
            GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
        ),
        "target_scale_protocol": GROUP_ACTION_REGRET_SCALE_PROTOCOL,
        "group_scale_summary": scale_summary,
    }


def audit_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return _audit_from_args_for_family(
        args,
        family=GROUP_NORMALIZED_RIGID_FAMILY,
        diagnostic_validator=_validate_group_normalized_diagnostics,
        selector=select_group_normalized_rigid_family,
        audit_protocol="tsc-v31r27-group-normalized-stage-audit-v1",
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    selection = payload["selection"]
    rigid = selection["rigid"]
    row = selection["candidate_summary"]
    return "\n".join(
        [
            "# TSC v31/r27 Group-Normalized Core Outcome",
            "",
            f"Generated: {payload['created_at_utc']}",
            "",
            "## Integrity",
            "",
            f"**PASS**: {payload['shards_checked']} shards and "
            f"{payload['results_checked']} independent results passed SHA-256, "
            "complete-city holdout, zero-target-label, rigid-parent, "
            "matched-group-scale, and frozen-cache checks.",
            "",
            f"- snapshot SHA-256: `{payload['snapshot']['snapshot_sha256']}`",
            f"- v26 reference SHA-256: `{payload['v26_reference']['sha256']}`",
            f"- cache aggregate SHA-256: `{payload['frozen_cache']['aggregate_sha256']}`",
            f"- evaluated action groups: {sum(payload['group_counts'].values()):,}",
            "",
            "## Frozen Decision",
            "",
            f"Stage: **{'PASS' if payload['stage_passed'] else 'FAIL'}**",
            f"Selected family: `{payload['selected_family'] or 'none'}`",
            "",
            "| Family | Macro regret | Improvement | Cities | Max regression | Gate |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Domain-normalized rigid | "
            f"{rigid['city_macro_mean_normalized_action_regret']:.6f} | "
            "0.00% | 0/6 | 0.000000 | reference |",
            f"| Group-normalized rigid | "
            f"{row['city_macro_mean_normalized_action_regret']:.6f} | "
            f"{row['relative_improvement_vs_rigid_pct']:+.2f}% | "
            f"{row['cities_improved']}/6 | "
            f"{row['max_absolute_regression']:.6f} | "
            f"{'PASS' if row['stage_gate']['passed'] else 'FAIL'} |",
            "",
            "Salt Lake data were not read and cannot alter this decision.",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--reference-audit-sha256", required=True)
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
    payload = audit_from_args(args)
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
