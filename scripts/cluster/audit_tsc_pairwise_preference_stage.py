#!/usr/bin/env python3
"""Fail-closed audit for the frozen v32/r28 antisymmetric pairwise screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
    select_pairwise_preference_family,
)
from cf_h2o.traffic_signal.action_ranker import (
    GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL,
    PAIRWISE_CAUSAL_ACTION_PARENTS,
    PAIRWISE_CAUSAL_STATE_PARENTS,
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


def _validate_pairwise_preference_diagnostics(
    row: Mapping[str, Any],
    *,
    source: Path,
) -> dict[str, Any]:
    diagnostics = row.get("model_diagnostics")
    fit = diagnostics.get("fit") if isinstance(diagnostics, dict) else None
    family_fit = (
        fit.get(PAIRWISE_PREFERENCE_FAMILY)
        if isinstance(fit, dict)
        else None
    )
    if not isinstance(family_fit, dict):
        raise ValueError(f"{source}: missing pairwise preference diagnostics")
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
        raise ValueError(f"{source}: causal restriction is disabled")
    if family_fit.get("preference_protocol") != (
        "all_action_antisymmetric_pairwise_v1"
    ):
        raise ValueError(f"{source}: preference protocol mismatch")
    if family_fit.get("target_normalization_protocol") != (
        GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
    ):
        raise ValueError(f"{source}: target normalization protocol mismatch")
    if family_fit.get("target_scale_protocol") != GROUP_ACTION_REGRET_SCALE_PROTOCOL:
        raise ValueError(f"{source}: target scale protocol mismatch")
    if tuple(family_fit.get("state_feature_names", ())) != tuple(
        PAIRWISE_CAUSAL_STATE_PARENTS
    ):
        raise ValueError(f"{source}: causal state parent set changed")
    if tuple(family_fit.get("action_feature_names", ())) != tuple(
        PAIRWISE_CAUSAL_ACTION_PARENTS
    ):
        raise ValueError(f"{source}: causal action parent set changed")
    if family_fit.get("uses_context_features") is not False:
        raise ValueError(f"{source}: city context entered pairwise features")
    if int(family_fit.get("source_domain_count", -1)) != 5:
        raise ValueError(f"{source}: fit source-domain count mismatch")
    if family_fit.get("antisymmetric_augmentation") is not True:
        raise ValueError(f"{source}: antisymmetric augmentation is disabled")
    unordered = int(family_fit.get("unordered_pair_count", -1))
    oriented = int(family_fit.get("oriented_pair_count", -1))
    if unordered <= 0 or oriented != 2 * unordered:
        raise ValueError(f"{source}: pair orientation accounting failed")
    if float(family_fit.get("pair_target_minimum", -2.0)) < -1.0 - 1e-12:
        raise ValueError(f"{source}: pair target fell below normalized support")
    if float(family_fit.get("pair_target_maximum", 2.0)) > 1.0 + 1e-12:
        raise ValueError(f"{source}: pair target exceeded normalized support")
    group_count = int(family_fit.get("action_group_count", -1))
    scale_summary = family_fit.get("group_target_scale_summary")
    if (
        group_count <= 0
        or not isinstance(scale_summary, dict)
        or int(scale_summary.get("group_count", -1)) != group_count
    ):
        raise ValueError(f"{source}: matched-group scale accounting failed")
    return {
        "source_domain_count": 5,
        "state_feature_count": int(family_fit.get("state_feature_count", -1)),
        "action_feature_count": int(family_fit.get("action_feature_count", -1)),
        "action_group_count": group_count,
        "unordered_pair_count": unordered,
        "oriented_pair_count": oriented,
        "preference_protocol": "all_action_antisymmetric_pairwise_v1",
        "target_scale_protocol": GROUP_ACTION_REGRET_SCALE_PROTOCOL,
    }


def audit_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return _audit_from_args_for_family(
        args,
        family=PAIRWISE_PREFERENCE_FAMILY,
        diagnostic_validator=_validate_pairwise_preference_diagnostics,
        selector=select_pairwise_preference_family,
        audit_protocol="tsc-v32r28-antisymmetric-pairwise-stage-audit-v1",
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    selection = payload["selection"]
    rigid = selection["rigid"]
    row = selection["candidate_summary"]
    return "\n".join(
        [
            "# TSC v32/r28 Antisymmetric Pairwise Core Outcome",
            "",
            f"Generated: {payload['created_at_utc']}",
            "",
            "## Integrity",
            "",
            f"**PASS**: {payload['shards_checked']} shards and "
            f"{payload['results_checked']} independent results passed SHA-256, "
            "complete-city holdout, zero-target-label, causal-parent, "
            "antisymmetry, pair-count, and frozen-cache checks.",
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
            f"| Rigid reference | "
            f"{rigid['city_macro_mean_normalized_action_regret']:.6f} | "
            "0.00% | 0/6 | 0.000000 | reference |",
            f"| Antisymmetric pairwise | "
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
