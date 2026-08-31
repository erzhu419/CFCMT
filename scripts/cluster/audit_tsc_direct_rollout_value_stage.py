#!/usr/bin/env python3
"""Fail-closed audit for the frozen v29/r25 direct rollout-value screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_direct_rollout_value_selection import (
    DIRECT_ROLLOUT_VALUE_FAMILY,
    select_direct_rollout_value_family,
)
from cf_h2o.traffic_signal.causal_mechanism_advantage import (
    INVARIANT_RIDGE_MECHANISM_ESTIMATOR_PROTOCOL,
    LEGACY_MECHANISM_DATASET_PROTOCOL,
    NO_SOURCE_EXPERT_GATE_PROTOCOL,
    RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL,
)
from scripts.cluster.audit_tsc_policy_consistent_stage_a import (
    _atomic_json,
    _atomic_text,
)
from scripts.cluster.audit_tsc_rollout_value_stage import (
    _audit_single_rollout_value_stage,
    _write_csv,
)


def _validate_direct_rollout_value_diagnostics_for_family(
    row: Mapping[str, Any],
    *,
    source: Path,
    family: str,
    expected_estimator_protocol: str,
) -> dict[str, Any]:
    diagnostics = row.get("model_diagnostics")
    fit = diagnostics.get("fit") if isinstance(diagnostics, dict) else None
    family_fit = (
        fit.get(family) if isinstance(fit, dict) else None
    )
    if not isinstance(family_fit, dict):
        raise ValueError(f"{source}: missing direct rollout-value fit diagnostics")
    config = family_fit.get("config")
    source_fit = family_fit.get("source")
    target_fit = family_fit.get("target")
    if not all(isinstance(value, dict) for value in (config, source_fit, target_fit)):
        raise ValueError(f"{source}: incomplete direct rollout-value diagnostics")
    if config.get("source_stack_protocol") != (
        RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
    ):
        raise ValueError(f"{source}: direct rollout-value stack protocol mismatch")
    if config.get("source_expert_gate_protocol") != NO_SOURCE_EXPERT_GATE_PROTOCOL:
        raise ValueError(f"{source}: an unregistered source expert gate was enabled")
    if config.get("mechanism_dataset_protocol") != (
        LEGACY_MECHANISM_DATASET_PROTOCOL
    ):
        raise ValueError(f"{source}: direct zero-prior protocol mismatch")
    if config.get("mechanism_action_ranking_selection") is not True:
        raise ValueError(f"{source}: action-ranking mechanism selection is disabled")
    if config.get("mechanism_estimator_protocol") != expected_estimator_protocol:
        raise ValueError(f"{source}: mechanism estimator protocol mismatch")
    if source_fit.get("mechanism_dataset_protocol") != (
        LEGACY_MECHANISM_DATASET_PROTOCOL
    ):
        raise ValueError(f"{source}: source direct mechanism protocol mismatch")
    if source_fit.get("source_stack_protocol") != (
        RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
    ):
        raise ValueError(f"{source}: source fit stack protocol mismatch")
    if int(source_fit.get("source_domain_count", -1)) != 5:
        raise ValueError(f"{source}: expected five source city domains")
    if int(source_fit.get("pair_excluded_core_fit_count", -1)) != 10:
        raise ValueError(f"{source}: pair-excluded rigid fit count mismatch")
    if int(source_fit.get("pair_excluded_mechanism_fit_count", -1)) != 10:
        raise ValueError(f"{source}: pair-excluded mechanism fit count mismatch")
    mechanism_fit = source_fit.get("mechanism_fit")
    if not isinstance(mechanism_fit, dict) or set(mechanism_fit) != {
        "terminal_clearance"
    }:
        raise ValueError(f"{source}: mechanism set is not terminal-clearance only")
    terminal = mechanism_fit["terminal_clearance"]
    if not isinstance(terminal, dict) or terminal.get("target_name") != (
        "terminal_system_load"
    ):
        raise ValueError(f"{source}: direct terminal target mismatch")
    if target_fit.get("enabled") is not False or target_fit.get("reason") != (
        "no_target_counterfactual_groups"
    ):
        raise ValueError(f"{source}: target adaptation unexpectedly enabled")
    candidate = source_fit.get("selected_candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"{source}: missing nested source candidate")
    return {
        "source_domain_count": 5,
        "pair_excluded_core_fit_count": 10,
        "pair_excluded_mechanism_fit_count": 10,
        "mechanism_name": "terminal_clearance",
        "target_name": "terminal_system_load",
        "parent_variant": terminal.get("variant_name"),
        "parent_names": terminal.get("parent_names"),
        "mechanism_enabled": bool(terminal.get("enabled")),
        "stack_enabled": bool(source_fit.get("stack_enabled")),
        "stack_weight": float(source_fit.get("stack_weight", 0.0)),
        "selected_alpha": float(candidate["alpha"]),
        "mechanism_estimator_protocol": expected_estimator_protocol,
    }


def _validate_direct_rollout_value_diagnostics(
    row: Mapping[str, Any],
    *,
    source: Path,
) -> dict[str, Any]:
    return _validate_direct_rollout_value_diagnostics_for_family(
        row,
        source=source,
        family=DIRECT_ROLLOUT_VALUE_FAMILY,
        expected_estimator_protocol=(
            INVARIANT_RIDGE_MECHANISM_ESTIMATOR_PROTOCOL
        ),
    )


def _audit_from_args_for_family(
    args: argparse.Namespace,
    *,
    family: str,
    diagnostic_validator,
    selector,
    audit_protocol: str,
) -> dict[str, Any]:
    return _audit_single_rollout_value_stage(
        component_root=args.component_root,
        manifest_path=args.manifest,
        snapshot_manifest=args.snapshot_manifest,
        reference_audit=args.reference_audit,
        reference_audit_sha256=args.reference_audit_sha256,
        expected_cache_root=args.expected_cache_root,
        cache_aggregate_sha256=args.cache_aggregate_sha256,
        source_rule_baseline_sha256=args.source_rule_baseline_sha256,
        source_rule_cache_aggregate_sha256=args.source_rule_cache_aggregate_sha256,
        source_rule_generator_source_sha256=args.source_rule_generator_source_sha256,
        source_rule_baseline_path=args.source_rule_baseline,
        family=family,
        diagnostic_validator=diagnostic_validator,
        selector=selector,
        audit_protocol=audit_protocol,
    )


def audit_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return _audit_from_args_for_family(
        args,
        family=DIRECT_ROLLOUT_VALUE_FAMILY,
        diagnostic_validator=_validate_direct_rollout_value_diagnostics,
        selector=select_direct_rollout_value_family,
        audit_protocol="tsc-v29r25-direct-rollout-value-stage-audit-v1",
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    selection = payload["selection"]
    rigid = selection["rigid"]
    row = selection["candidate_summary"]
    return "\n".join(
        [
            "# TSC v29/r25 Direct Invariant Rollout-Value Outcome",
            "",
            f"Generated: {payload['created_at_utc']}",
            "",
            "## Integrity",
            "",
            f"**PASS**: {payload['shards_checked']} shards and "
            f"{payload['results_checked']} independent results passed SHA-256, "
            "complete-city holdout, zero-target-label, zero-prior terminal-only, "
            "pair-excluded-fit, and frozen-cache checks.",
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
            f"| Rigid | {rigid['city_macro_mean_normalized_action_regret']:.6f} | "
            "0.00% | 0/6 | 0.000000 | reference |",
            f"| Direct rollout-value residual | "
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
