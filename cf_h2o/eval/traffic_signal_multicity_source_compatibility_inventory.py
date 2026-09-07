"""Build one V144 ordered source-to-target compatibility inventory row."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
    RESULT_PROTOCOL as V143_RESULT_PROTOCOL,
    run_multicity_uniform_source_target,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v144-multicity-source-compatibility-inventory-target-v1"


def run_source_compatibility_inventory(**kwargs: Any) -> dict[str, Any]:
    result = run_multicity_uniform_source_target(**kwargs)
    target = str(result["target_city"])
    expected_sources = set(EXPECTED_CITY_GROUPS) - {target}
    if (
        result.get("protocol") != V143_RESULT_PROTOCOL
        or set(result.get("source_arm_summaries", {})) != expected_sources
        or set(result.get("source_placebo_arm_summaries", {})) != expected_sources
        or set(result.get("city_covariate_signatures", {}))
        != set(EXPECTED_CITY_GROUPS)
    ):
        raise ValueError("V144 ordered-pair inventory is incomplete")
    result.update(
        {
            "protocol": RESULT_PROTOCOL,
            "predecessor_protocol": V143_RESULT_PROTOCOL,
            "scientific_status": "ordered-pair-source-compatibility-development-target",
            "claim_boundary": (
                "V144 records all six source-to-target effects and label-free city "
                "covariates for development. It does not select or validate a "
                "deployable source gate."
            ),
        }
    )
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
        raise FileExistsError(f"refusing to overwrite V144 result: {args.out}")
    result = run_source_compatibility_inventory(
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
        {
            "status": "DONE",
            "protocol": result["protocol"],
            "target_city": result["target_city"],
            "source_count": len(result["source_arm_summaries"]),
            "result": str(args.out),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
