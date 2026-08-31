#!/usr/bin/env python3
"""Fit one city's strict target-only CFCMT comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_strict_target_only_fit import (
    run_strict_target_only_fit,
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
    parser.add_argument("--external-cache-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--external-repair-audit", type=Path, required=True)
    parser.add_argument("--main-model", type=Path, required=True)
    parser.add_argument("--main-model-sha256", required=True)
    parser.add_argument("--main-result", type=Path, required=True)
    parser.add_argument("--main-result-sha256", required=True)
    parser.add_argument("--city", choices=tuple(SCENARIOS_BY_CITY), required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--result-out", type=Path, required=True)
    args = parser.parse_args()
    payload = run_strict_target_only_fit(
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
        model_out=args.model_out,
        result_out=args.result_out,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "city": args.city,
                "model_sha256": payload["model_artifact"]["sha256"],
                "source_rows_consumed": payload["training_data_audit"][
                    "source_rows_consumed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
