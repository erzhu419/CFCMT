#!/usr/bin/env python3
"""Fail-closed audit for the frozen v30/r26 boosted rollout-value screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_boosted_rollout_value_selection import (
    BOOSTED_ROLLOUT_VALUE_FAMILY,
    select_boosted_rollout_value_family,
)
from cf_h2o.traffic_signal.causal_mechanism_advantage import (
    CAUSAL_BOOSTED_MECHANISM_ESTIMATOR_PROTOCOL,
)
from scripts.cluster.audit_tsc_direct_rollout_value_stage import (
    _audit_from_args_for_family,
    _validate_direct_rollout_value_diagnostics_for_family,
)
from scripts.cluster.audit_tsc_policy_consistent_stage_a import (
    _atomic_json,
    _atomic_text,
)
from scripts.cluster.audit_tsc_rollout_value_stage import _write_csv


def _validate_boosted_rollout_value_diagnostics(
    row: Mapping[str, Any],
    *,
    source: Path,
) -> dict[str, Any]:
    result = _validate_direct_rollout_value_diagnostics_for_family(
        row,
        source=source,
        family=BOOSTED_ROLLOUT_VALUE_FAMILY,
        expected_estimator_protocol=(
            CAUSAL_BOOSTED_MECHANISM_ESTIMATOR_PROTOCOL
        ),
    )
    fit = row["model_diagnostics"]["fit"][BOOSTED_ROLLOUT_VALUE_FAMILY]
    terminal = fit["source"]["mechanism_fit"]["terminal_clearance"]
    if terminal.get("estimator") != "HistGradientBoostingRegressor":
        raise ValueError(f"{source}: boosted terminal estimator mismatch")
    if terminal.get("variant_name") == "dense_all_local_plus_context":
        raise ValueError(f"{source}: boosted terminal estimator entered dense mode")
    result["estimator"] = terminal["estimator"]
    return result


def _markdown(payload: Mapping[str, Any]) -> str:
    selection = payload["selection"]
    rigid = selection["rigid"]
    row = selection["candidate_summary"]
    return "\n".join(
        [
            "# TSC v30/r26 Parent-Restricted Nonlinear Rollout-Value Outcome",
            "",
            f"Generated: {payload['created_at_utc']}",
            "",
            "## Integrity",
            "",
            f"**PASS**: {payload['shards_checked']} shards and "
            f"{payload['results_checked']} independent results passed SHA-256, "
            "complete-city holdout, zero-target-label, boosted terminal-only, "
            "parent-restriction, pair-excluded-fit, and frozen-cache checks.",
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
            f"| Boosted rollout-value residual | "
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
    payload = _audit_from_args_for_family(
        args,
        family=BOOSTED_ROLLOUT_VALUE_FAMILY,
        diagnostic_validator=_validate_boosted_rollout_value_diagnostics,
        selector=select_boosted_rollout_value_family,
        audit_protocol="tsc-v30r26-boosted-rollout-value-stage-audit-v1",
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
