"""Run one V146 target with five-fold OOF source compatibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
    RESULT_PROTOCOL as V143_RESULT_PROTOCOL,
    TARGET_BUDGET,
    run_multicity_uniform_source_target,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v146-multicity-oof-compatibility-gate-target-v1"
COMPATIBILITY_FOLD_COUNT = 5


def run_oof_compatibility_gate_target(**kwargs: Any) -> dict[str, Any]:
    result = run_multicity_uniform_source_target(
        **kwargs,
        adaptation_compatibility_fold_count=COMPATIBILITY_FOLD_COUNT,
        result_protocol=RESULT_PROTOCOL,
        scientific_status="oof-compatibility-source-gate-development-target",
        claim_boundary=(
            "V146 is post-V145 seven-city method development with B25 target "
            "adaptation. Source trust uses five-fold B20-to-B5 cross-fitted "
            "model predictions without scored-fold labels or evaluation data. "
            "It is few-shot development, not untouched-city confirmation."
        ),
    )
    target = str(result["target_city"])
    expected_sources = set(EXPECTED_CITY_GROUPS) - {target}
    diagnostics = result.get("adaptation_compatibility_diagnostics", {})
    if (
        result.get("protocol") != RESULT_PROTOCOL
        or int(result.get("target_budget", -1)) != TARGET_BUDGET
        or set(result.get("source_arm_summaries", {})) != expected_sources
        or set(result.get("source_placebo_arm_summaries", {})) != expected_sources
        or set(result.get("compatibility_shrunk_source_arm_summaries", {}))
        != expected_sources
        or set(result.get("compatibility_shrunk_placebo_arm_summaries", {}))
        != expected_sources
        or diagnostics.get("evaluation_features_or_labels_used") is not False
        or diagnostics.get("scored_adaptation_labels_used") is not False
        or int(diagnostics.get("fold_count", -1)) != COMPATIBILITY_FOLD_COUNT
        or set(diagnostics.get("source_arms", {})) != expected_sources
    ):
        raise ValueError("V146 target result contract is incomplete")
    result["predecessor_protocol"] = V143_RESULT_PROTOCOL
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-city", required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--fit-result-sha256", required=True)
    parser.add_argument("--fit-protocol", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-audit", type=Path, required=True)
    parser.add_argument("--source-cache-audit-sha256", required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--fit-workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V146 result: {args.out}")
    result = run_oof_compatibility_gate_target(
        target_city=args.target_city,
        fit_result_path=args.fit_result,
        expected_fit_result_sha256=args.fit_result_sha256,
        fit_protocol_path=args.fit_protocol,
        source_cache_root=args.source_cache_root,
        source_manifest_path=args.source_manifest,
        source_cache_audit_path=args.source_cache_audit,
        expected_source_cache_audit_sha256=args.source_cache_audit_sha256,
        conversion_root=args.conversion_root,
        cache_workers=args.cache_workers,
        fit_workers=args.fit_workers,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "DONE",
                "protocol": result["protocol"],
                "target_city": result["target_city"],
                "source_count": len(result["source_arm_summaries"]),
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
