#!/usr/bin/env python3
"""Run target-offline cross-fitted source selection for one TSC city."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_causal_source_offline_selection import (
    run_target_offline_source_selection,
)


SCENARIOS_BY_CITY = {
    "los_angeles": ("la_1x4",),
    "jinan": (
        "jinan_3x4_real",
        "jinan_3x4_real_2000",
        "jinan_3x4_real_2500",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-baseline", type=Path, required=True)
    parser.add_argument("--source-repair-audit", type=Path, required=True)
    parser.add_argument("--external-cache-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--external-repair-audit", type=Path, required=True)
    parser.add_argument("--main-model", type=Path, required=True)
    parser.add_argument("--main-model-sha256", required=True)
    parser.add_argument("--main-result", type=Path, required=True)
    parser.add_argument("--main-result-sha256", required=True)
    parser.add_argument("--city", choices=tuple(SCENARIOS_BY_CITY), required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--fit-workers", type=int, default=20)
    parser.add_argument("--minimum-mean-improvement", type=float, default=0.005)
    parser.add_argument("--maximum-fold-regression", type=float, default=0.01)
    parser.add_argument("--source-familywise-alpha", type=float, default=0.025)
    parser.add_argument("--guard-familywise-alpha", type=float, default=0.025)
    parser.add_argument("--guard-minimum-mean-improvement", type=float, default=0.002)
    parser.add_argument("--guard-maximum-fold-regression", type=float, default=0.01)
    parser.add_argument("--guard-min-context-trust", type=float, default=0.1)
    parser.add_argument("--result-out", type=Path, required=True)
    args = parser.parse_args()
    payload = run_target_offline_source_selection(
        source_cache_root=args.source_cache_root,
        source_manifest_path=args.source_manifest,
        source_baseline_result=args.source_baseline,
        source_repair_audit=args.source_repair_audit,
        external_cache_root=args.external_cache_root,
        external_manifest_path=args.external_manifest,
        external_repair_audit=args.external_repair_audit,
        main_model_path=args.main_model,
        expected_main_model_sha256=args.main_model_sha256,
        main_result_path=args.main_result,
        expected_main_result_sha256=args.main_result_sha256,
        city=args.city,
        scenarios_by_city=SCENARIOS_BY_CITY,
        workers=args.workers,
        fit_workers=args.fit_workers,
        minimum_mean_improvement=args.minimum_mean_improvement,
        maximum_fold_regression=args.maximum_fold_regression,
        source_familywise_alpha=args.source_familywise_alpha,
        guard_familywise_alpha=args.guard_familywise_alpha,
        guard_minimum_mean_improvement=args.guard_minimum_mean_improvement,
        guard_maximum_fold_regression=args.guard_maximum_fold_regression,
        guard_min_context_trust=args.guard_min_context_trust,
        result_out=args.result_out,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "city": args.city,
                "selection": payload["selection"],
                "elapsed_sec": payload["elapsed_sec"],
            },
            sort_keys=True,
        )
    )
    print(f"Results saved to {args.result_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
