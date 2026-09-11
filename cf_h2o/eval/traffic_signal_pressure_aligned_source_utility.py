"""Nested source utility with PhasePressure-aligned candidate actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    aggregate_results as _aggregate_results,
    run_target as _run_target,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v150m-pressure-aligned-source-utility-target-v1"
AGGREGATE_PROTOCOL = "tsc-v150m-pressure-aligned-source-utility-aggregate-v1"
METHOD_PROTOCOL = "nested-pressure-aligned-source-mechanism-utility-v1"
UTILITY_GATE_PROTOCOL = (
    "ridge-upper-cost-bound-over-pressure-aligned-source-proposals-v1"
)
RUNTIME_MODEL_PROTOCOL = "tsc-v150m-pressure-aligned-source-runtime-model-v1"


def run_target(**kwargs: Any) -> dict[str, Any]:
    return _run_target(
        **kwargs,
        candidate_score_constraint=PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
        result_protocol=RESULT_PROTOCOL,
        method_protocol=METHOD_PROTOCOL,
        utility_gate_protocol=UTILITY_GATE_PROTOCOL,
        runtime_model_protocol=RUNTIME_MODEL_PROTOCOL,
        scientific_status="seven-city-pressure-aligned-source-development-target",
        claim_boundary=(
            "V150M is a seven-city B100 target-offline adaptation experiment. "
            "Both real-source and matched-placebo candidates are restricted to "
            "the target-only rigid action or the deploy-observable PhasePressure "
            "reference before nested utility fitting. It is not closed-loop or "
            "fresh-city confirmation."
        ),
    )


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    if any(
        row.get("method", {}).get("protocol") != METHOD_PROTOCOL
        or row.get("method", {})
        .get("selector", {})
        .get("candidate_score_constraint")
        != PRESSURE_ALIGNED_CANDIDATE_PROTOCOL
        for row in rows
    ):
        raise ValueError("V150M aggregate requires pressure-aligned target results")
    return _aggregate_results(
        rows,
        result_protocol=RESULT_PROTOCOL,
        aggregate_protocol=AGGREGATE_PROTOCOL,
        utility_gate_protocol=UTILITY_GATE_PROTOCOL,
        pass_decision="authorize_v150m_pressure_aligned_closed_loop_development",
        reject_decision="retain_rigid_target_adaptation_and_reject_v150m_source_claim",
        claim_boundary=(
            "V150M tests nested source contribution under a predeclared "
            "PhasePressure-aligned action constraint on seven development cities. "
            "Passing does not establish closed-loop or fresh-city efficacy."
        ),
    )


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    target_runner: Any = run_target,
    result_aggregator: Any = aggregate_results,
    version_label: str = "V150M",
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    target = subparsers.add_parser("target")
    target.add_argument("--target-city", required=True)
    target.add_argument("--authorization-result", type=Path, required=True)
    target.add_argument("--authorization-result-sha256", required=True)
    target.add_argument("--v150j-rejection", type=Path, required=True)
    target.add_argument("--v150j-rejection-sha256", required=True)
    target.add_argument("--fit-result", type=Path, required=True)
    target.add_argument("--fit-result-sha256", required=True)
    target.add_argument("--fit-protocol", type=Path, required=True)
    target.add_argument("--source-cache-root", type=Path, required=True)
    target.add_argument("--source-manifest", type=Path, required=True)
    target.add_argument("--source-cache-audit", type=Path, required=True)
    target.add_argument("--source-cache-audit-sha256", required=True)
    target.add_argument("--conversion-root", type=Path, required=True)
    target.add_argument("--cache-workers", type=int, default=8)
    target.add_argument("--fit-workers", type=int, default=6)
    target.add_argument("--runtime-model-out", type=Path)
    target.add_argument("--out", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", nargs=7, type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(
            f"refusing to overwrite {version_label} result: {args.out}"
        )
    if args.mode == "target":
        result = target_runner(
            target_city=args.target_city,
            authorization_result_path=args.authorization_result,
            expected_authorization_sha256=args.authorization_result_sha256,
            v150j_rejection_path=args.v150j_rejection,
            expected_v150j_rejection_sha256=args.v150j_rejection_sha256,
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
            runtime_model_path=args.runtime_model_out,
        )
        status = "DONE"
    else:
        with_results = [
            json.loads(path.read_text(encoding="utf-8")) for path in args.results
        ]
        result = result_aggregator(with_results)
        status = "PASS" if result["development_gate"]["passed"] else "REJECT"
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": status,
                "protocol": result["protocol"],
                "target_city": result.get("target_city"),
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
