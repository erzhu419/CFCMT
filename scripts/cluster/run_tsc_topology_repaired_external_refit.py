#!/usr/bin/env python3
"""Fit one topology-repaired external-city model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_topology_repaired_external_refit import (
    run_topology_repaired_refit,
)


SCENARIOS = {
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
    parser.add_argument("--city", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fit-workers", type=int, default=3)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--result-out", type=Path, required=True)
    args = parser.parse_args()
    payload = run_topology_repaired_refit(
        source_cache_root=args.source_cache_root,
        source_manifest_path=args.source_manifest,
        source_baseline_result=args.source_baseline,
        source_repair_audit=args.source_repair_audit,
        external_cache_root=args.external_cache_root,
        external_manifest_path=args.external_manifest,
        external_repair_audit=args.external_repair_audit,
        city=args.city,
        scenarios_by_city=SCENARIOS,
        workers=args.workers,
        fit_workers=args.fit_workers,
        model_out=args.model_out,
        result_out=args.result_out,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "city": args.city,
                "elapsed_sec": payload["elapsed_sec"],
                "model_sha256": payload["model_artifact"]["sha256"],
                "source_scenario_count": payload["source_cache_audit"][
                    "scenario_count"
                ],
                "source_used_file_count": payload["source_cache_audit"][
                    "used_file_count"
                ],
            },
            sort_keys=True,
        )
    )
    print(f"Results saved to {args.result_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
