#!/usr/bin/env python3
"""Fail-closed audit for the frozen v33/r29 source-gated pairwise screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_source_gated_pairwise_selection import (
    SOURCE_GATED_PAIRWISE_FAMILY,
    select_source_gated_pairwise_family,
)
from cf_h2o.traffic_signal.action_ranker import (
    CAUSAL_RIGID_RANKING_PARENTS,
    PAIRWISE_CAUSAL_ACTION_PARENTS,
    PAIRWISE_CAUSAL_STATE_PARENTS,
)
from cf_h2o.traffic_signal.source_gated_pairwise import (
    SOURCE_GATED_PAIRWISE_PROTOCOL,
)
from scripts.cluster.audit_tsc_direct_rollout_value_stage import (
    _audit_from_args_for_family,
)
from scripts.cluster.audit_tsc_policy_consistent_stage_a import (
    _atomic_json,
    _atomic_text,
)
from scripts.cluster.audit_tsc_rollout_value_stage import _write_csv


def _validate_source_gated_pairwise_diagnostics(
    row: Mapping[str, Any],
    *,
    source: Path,
) -> dict[str, Any]:
    diagnostics = row.get("model_diagnostics")
    fit = diagnostics.get("fit") if isinstance(diagnostics, dict) else None
    family_fit = (
        fit.get(SOURCE_GATED_PAIRWISE_FAMILY)
        if isinstance(fit, dict)
        else None
    )
    if not isinstance(family_fit, dict):
        raise ValueError(f"{source}: missing source-gated pairwise diagnostics")
    source_domains = diagnostics.get("source_domains")
    if not isinstance(source_domains, list) or len(source_domains) != 5:
        raise ValueError(f"{source}: expected five source city domains")
    if int(diagnostics.get("target_adaptation_groups", -1)) != 0:
        raise ValueError(f"{source}: target labels entered the fit")
    if family_fit.get("estimator") != (
        "source_gated_antisymmetric_pairwise_advantage"
    ):
        raise ValueError(f"{source}: estimator changed")
    config = family_fit.get("config")
    source_fit = family_fit.get("source")
    if not isinstance(config, dict) or not isinstance(source_fit, dict):
        raise ValueError(f"{source}: incomplete source-gate diagnostics")
    if float(config.get("gate_ridge_alpha", -1.0)) != 10.0:
        raise ValueError(f"{source}: gate ridge penalty changed")
    if float(config.get("gate_error_quantile", -1.0)) != 0.75:
        raise ValueError(f"{source}: gate error quantile changed")
    if int(config.get("minimum_disagreement_groups", -1)) != 24:
        raise ValueError(f"{source}: minimum gate support changed")
    if float(config.get("minimum_robust_gain", -1.0)) != 0.002:
        raise ValueError(f"{source}: source robust-gain gate changed")
    if float(config.get("worst_regret_tolerance", -1.0)) != 0.01:
        raise ValueError(f"{source}: source worst-regret tolerance changed")
    if source_fit.get("protocol") != SOURCE_GATED_PAIRWISE_PROTOCOL:
        raise ValueError(f"{source}: source gate protocol mismatch")
    if int(source_fit.get("source_domain_count", -1)) != 5:
        raise ValueError(f"{source}: fit source-domain count mismatch")
    if int(source_fit.get("source_city_oof_fold_count", -1)) != 5:
        raise ValueError(f"{source}: source-city OOF fold count mismatch")
    if source_fit.get("target_labels_used") is not False:
        raise ValueError(f"{source}: target-label audit failed")
    rigid_fit = source_fit.get("rigid_fit")
    pairwise_fit = source_fit.get("pairwise_fit")
    if not isinstance(rigid_fit, dict) or not isinstance(pairwise_fit, dict):
        raise ValueError(f"{source}: missing expert fit diagnostics")
    if tuple(rigid_fit.get("feature_names", ())) != tuple(
        CAUSAL_RIGID_RANKING_PARENTS
    ):
        raise ValueError(f"{source}: rigid parent set changed")
    if pairwise_fit.get("preference_protocol") != (
        "all_action_antisymmetric_pairwise_v1"
    ):
        raise ValueError(f"{source}: pairwise preference protocol changed")
    if tuple(pairwise_fit.get("state_feature_names", ())) != tuple(
        PAIRWISE_CAUSAL_STATE_PARENTS
    ):
        raise ValueError(f"{source}: pairwise state parent set changed")
    if tuple(pairwise_fit.get("action_feature_names", ())) != tuple(
        PAIRWISE_CAUSAL_ACTION_PARENTS
    ):
        raise ValueError(f"{source}: pairwise action parent set changed")
    disagreement_count = int(source_fit.get("disagreement_group_count", -1))
    enabled = bool(source_fit.get("gate_enabled"))
    if disagreement_count < 24:
        expected_reason = "insufficient_source_oof_disagreements"
        if enabled:
            raise ValueError(f"{source}: under-supported source gate was enabled")
    else:
        expected_reason = {
            True: "selected_by_nested_source_city_oof_regret",
            False: "pairwise_gate_failed_source_city_oof_regret",
        }[enabled]
    if source_fit.get("reason") != expected_reason:
        raise ValueError(f"{source}: source-gate decision reason mismatch")
    if enabled and int(source_fit.get("selected_pairwise_group_count", 0)) <= 0:
        raise ValueError(f"{source}: enabled gate selected no pairwise groups")
    return {
        "source_domain_count": 5,
        "source_city_oof_fold_count": 5,
        "disagreement_group_count": int(
            disagreement_count
        ),
        "gate_enabled": enabled,
        "selected_pairwise_group_count": int(
            source_fit.get("selected_pairwise_group_count", 0)
        ),
        "one_sided_error_margin": float(
            source_fit.get("one_sided_error_margin", 0.0)
        ),
        "robust_gain_vs_rigid": float(
            source_fit.get("robust_gain_vs_rigid", 0.0)
        ),
    }


def audit_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return _audit_from_args_for_family(
        args,
        family=SOURCE_GATED_PAIRWISE_FAMILY,
        diagnostic_validator=_validate_source_gated_pairwise_diagnostics,
        selector=select_source_gated_pairwise_family,
        audit_protocol="tsc-v33r29-source-gated-pairwise-stage-audit-v1",
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    selection = payload["selection"]
    rigid = selection["rigid"]
    row = selection["candidate_summary"]
    enabled = sum(
        bool(item["rollout_value_fit"]["gate_enabled"])
        for item in payload["integrity_rows"]
    )
    return "\n".join(
        [
            "# TSC v33/r29 Source-Gated Pairwise Outcome",
            "",
            f"Generated: {payload['created_at_utc']}",
            "",
            "## Integrity",
            "",
            f"**PASS**: {payload['shards_checked']} shards and "
            f"{payload['results_checked']} independent results passed SHA-256, "
            "complete-city holdout, zero-target-label, nested source-city OOF, "
            "causal-parent, and frozen-cache checks.",
            "",
            f"- snapshot SHA-256: `{payload['snapshot']['snapshot_sha256']}`",
            f"- v26 reference SHA-256: `{payload['v26_reference']['sha256']}`",
            f"- cache aggregate SHA-256: `{payload['frozen_cache']['aggregate_sha256']}`",
            f"- evaluated action groups: {sum(payload['group_counts'].values()):,}",
            f"- target folds with source-approved pairwise gate: {enabled}/6",
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
            f"| Source-gated pairwise | "
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
