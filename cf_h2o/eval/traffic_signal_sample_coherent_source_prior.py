"""Test a source prior whose strength decreases with target evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    SOURCE_PRIOR_STRENGTH,
    _read_json,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    run_target as _run_rigid_target,
)
from cf_h2o.eval.traffic_signal_source_identifiability_budget_curve import (
    AGGREGATE_PROTOCOL as V150H_AGGREGATE_PROTOCOL,
    COMMON_EVALUATION_RESERVE_BUDGET,
    TARGET_BUDGETS,
    aggregate_results as _aggregate_budget_results,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v150i-sample-coherent-source-prior-target-v1"
AGGREGATE_PROTOCOL = "tsc-v150i-sample-coherent-source-prior-aggregate-v1"
METHOD_PROTOCOL = "rigid-source-mechanism-sample-coherent-prior-v1"
REFERENCE_BUDGET = 25


def prior_strength_for_budget(target_budget: int) -> float:
    budget = int(target_budget)
    if budget not in TARGET_BUDGETS:
        raise ValueError(f"unsupported V150I target budget: {budget}")
    return float(SOURCE_PRIOR_STRENGTH * REFERENCE_BUDGET / budget)


def run_target(
    *,
    target_budget: int,
    v150h_rejection_path: Path,
    expected_v150h_rejection_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    budget = int(target_budget)
    prior_strength = prior_strength_for_budget(budget)
    if _sha256(v150h_rejection_path) != str(expected_v150h_rejection_sha256):
        raise ValueError("V150H rejection identity changed")
    rejection = _read_json(v150h_rejection_path)
    if (
        rejection.get("protocol") != V150H_AGGREGATE_PROTOCOL
        or rejection.get("development_gate", {}).get("passed") is not False
        or rejection.get("development_gate", {}).get("decision")
        != "retain_rigid_cfcmt_and_reject_source_closed_loop_claim"
    ):
        raise ValueError("V150I requires the frozen V150H rejection")
    result = _run_rigid_target(
        **kwargs,
        target_budget=budget,
        evaluation_reserve_budget=COMMON_EVALUATION_RESERVE_BUDGET,
        source_prior_strength=prior_strength,
        result_protocol=RESULT_PROTOCOL,
        scientific_status="seven-city-sample-coherent-source-prior-development-target",
        method_protocol=METHOD_PROTOCOL,
        selector_mode="crossfitted_target_and_matched_placebo_gain",
        matched_placebo_follows_source_admission=True,
        version_label="v150i",
        claim_boundary=(
            f"V150I uses B{budget} target-labelled groups and source-prior "
            f"strength {prior_strength:.6g}, equal to 0.10*25/{budget}. All "
            "budgets use the same groups outside the B100 reserve. This is "
            "target-offline development, not zero-shot or confirmation."
        ),
    )
    result["inputs"]["v150h_rejection_sha256"] = str(
        expected_v150h_rejection_sha256
    )
    result["inputs"]["v150h_rejection_protocol"] = str(rejection["protocol"])
    result["method"]["prior_schedule"] = {
        "protocol": "inverse-target-group-count-v1",
        "reference_budget": REFERENCE_BUDGET,
        "reference_strength": SOURCE_PRIOR_STRENGTH,
        "effective_strength": prior_strength,
    }
    return result


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    for row in rows:
        budget = int(row.get("target_budget", -1))
        expected = prior_strength_for_budget(budget)
        observed = float(row.get("method", {}).get("source_prior_strength", -1.0))
        schedule = row.get("method", {}).get("prior_schedule", {})
        if (
            abs(observed - expected) > 1e-12
            or schedule.get("protocol") != "inverse-target-group-count-v1"
            or abs(float(schedule.get("effective_strength", -1.0)) - expected)
            > 1e-12
        ):
            raise ValueError("V150I source-prior schedule changed")
    result = _aggregate_budget_results(
        rows,
        expected_result_protocol=RESULT_PROTOCOL,
        aggregate_protocol=AGGREGATE_PROTOCOL,
        experiment_label="V150I",
        pass_decision="authorize_sample_coherent_b100_closed_loop_development",
    )
    result["source_prior_strength_by_budget"] = {
        str(budget): prior_strength_for_budget(budget)
        for budget in TARGET_BUDGETS
    }
    result["objective_correction"] = (
        "Target group weights have unit total mass at every budget, so source "
        "prior strength scales inversely with target group count."
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    target = subparsers.add_parser("target")
    target.add_argument("--target-city", required=True)
    target.add_argument("--target-budget", type=int, choices=TARGET_BUDGETS, required=True)
    target.add_argument("--authorization-result", type=Path, required=True)
    target.add_argument("--authorization-result-sha256", required=True)
    target.add_argument("--fit-result", type=Path, required=True)
    target.add_argument("--fit-result-sha256", required=True)
    target.add_argument("--fit-protocol", type=Path, required=True)
    target.add_argument("--source-cache-root", type=Path, required=True)
    target.add_argument("--source-manifest", type=Path, required=True)
    target.add_argument("--source-cache-audit", type=Path, required=True)
    target.add_argument("--source-cache-audit-sha256", required=True)
    target.add_argument("--conversion-root", type=Path, required=True)
    target.add_argument("--cache-workers", type=int, default=8)
    target.add_argument("--v150h-rejection", type=Path, required=True)
    target.add_argument("--v150h-rejection-sha256", required=True)
    target.add_argument("--out", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", nargs=21, type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150I result: {args.out}")
    if args.mode == "target":
        result = run_target(
            target_budget=args.target_budget,
            v150h_rejection_path=args.v150h_rejection,
            expected_v150h_rejection_sha256=args.v150h_rejection_sha256,
            target_city=args.target_city,
            authorization_result_path=args.authorization_result,
            expected_authorization_sha256=args.authorization_result_sha256,
            fit_result_path=args.fit_result,
            expected_fit_result_sha256=args.fit_result_sha256,
            fit_protocol_path=args.fit_protocol,
            source_cache_root=args.source_cache_root,
            source_manifest_path=args.source_manifest,
            source_cache_audit_path=args.source_cache_audit,
            expected_source_cache_audit_sha256=args.source_cache_audit_sha256,
            conversion_root=args.conversion_root,
            cache_workers=args.cache_workers,
        )
        status = "DONE"
    else:
        result = aggregate_results([_read_json(path) for path in args.results])
        status = "PASS" if result["development_gate"]["passed"] else "REJECT"
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": status,
                "protocol": result["protocol"],
                "target_city": result.get("target_city"),
                "target_budget": result.get("target_budget"),
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
