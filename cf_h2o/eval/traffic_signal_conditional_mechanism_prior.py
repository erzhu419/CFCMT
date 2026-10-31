"""Evaluate locally conditioned mechanism priors on the frozen B100 split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    _read_json,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    run_target as _run_rigid_target,
)
from cf_h2o.eval.traffic_signal_sample_coherent_source_prior import (
    AGGREGATE_PROTOCOL as V150I_AGGREGATE_PROTOCOL,
    prior_strength_for_budget,
)
from cf_h2o.eval.traffic_signal_source_identifiability_budget_curve import (
    COMMON_EVALUATION_RESERVE_BUDGET,
    PRIMARY_BUDGET,
    _budget_summary,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.mechanism_parameter_prior import (
    CONDITIONAL_MECHANISM_STATE_FEATURES,
    conditional_mechanism_parameter_design,
)


RESULT_PROTOCOL = "tsc-v150j-conditional-mechanism-prior-target-v1"
AGGREGATE_PROTOCOL = "tsc-v150j-conditional-mechanism-prior-aggregate-v1"
METHOD_PROTOCOL = "rigid-source-locally-conditioned-mechanism-prior-v1"
DESIGN_PROTOCOL = "deterministic-local-right-of-way-execution-interactions-v1"
TARGET_BUDGET = PRIMARY_BUDGET
SOURCE_PRIOR_STRENGTH_B100 = prior_strength_for_budget(TARGET_BUDGET)


def run_target(
    *,
    v150i_rejection_path: Path,
    expected_v150i_rejection_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    if _sha256(v150i_rejection_path) != str(expected_v150i_rejection_sha256):
        raise ValueError("V150I rejection identity changed")
    rejection = _read_json(v150i_rejection_path)
    if (
        rejection.get("protocol") != V150I_AGGREGATE_PROTOCOL
        or rejection.get("development_gate", {}).get("passed") is not False
        or rejection.get("development_gate", {}).get("decision")
        != "retain_rigid_cfcmt_and_reject_source_closed_loop_claim"
    ):
        raise ValueError("V150J requires the frozen V150I rejection")
    result = _run_rigid_target(
        **kwargs,
        target_budget=TARGET_BUDGET,
        evaluation_reserve_budget=COMMON_EVALUATION_RESERVE_BUDGET,
        source_prior_strength=SOURCE_PRIOR_STRENGTH_B100,
        mechanism_design_builder=conditional_mechanism_parameter_design,
        mechanism_design_protocol=DESIGN_PROTOCOL,
        result_protocol=RESULT_PROTOCOL,
        scientific_status="seven-city-conditional-mechanism-prior-development-target",
        method_protocol=METHOD_PROTOCOL,
        selector_mode="crossfitted_target_and_matched_placebo_gain",
        matched_placebo_follows_source_admission=True,
        version_label="v150j",
        claim_boundary=(
            "V150J is a B100 representation test using deterministic local "
            "right-of-way and neighboring execution interactions. Evaluation "
            "groups, selector, source-prior strength and controls are inherited "
            "from V150I. It is not closed-loop or fresh-city confirmation."
        ),
    )
    result["inputs"]["v150i_rejection_sha256"] = str(
        expected_v150i_rejection_sha256
    )
    result["inputs"]["v150i_rejection_protocol"] = str(rejection["protocol"])
    result["method"]["conditional_state_features"] = {
        block: list(names)
        for block, names in CONDITIONAL_MECHANISM_STATE_FEATURES.items()
    }
    return result


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    by_city = {str(row.get("target_city")): row for row in rows}
    if (
        len(rows) != len(EXPECTED_CITY_GROUPS)
        or set(by_city) != set(EXPECTED_CITY_GROUPS)
        or any(row.get("protocol") != RESULT_PROTOCOL for row in rows)
        or any(int(row.get("target_budget", -1)) != TARGET_BUDGET for row in rows)
        or any(
            row.get("method", {}).get("mechanism_design_protocol")
            != DESIGN_PROTOCOL
            for row in rows
        )
        or any(
            abs(
                float(row.get("method", {}).get("source_prior_strength", -1.0))
                - SOURCE_PRIOR_STRENGTH_B100
            )
            > 1e-12
            for row in rows
        )
        or any(
            row.get("source_null_contract", {}).get("summaries_equal") is not True
            for row in rows
        )
    ):
        raise ValueError("V150J aggregate requires seven valid B100 results")
    summary = _budget_summary(by_city)
    passed = bool(summary["passed"])
    return {
        "protocol": AGGREGATE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development_gate_pass" if passed else "development_gate_reject"
        ),
        "city_count": len(rows),
        "target_budget": TARGET_BUDGET,
        "source_prior_strength": SOURCE_PRIOR_STRENGTH_B100,
        "mechanism_design_protocol": DESIGN_PROTOCOL,
        "conditional_state_features": {
            block: list(names)
            for block, names in CONDITIONAL_MECHANISM_STATE_FEATURES.items()
        },
        **{key: value for key, value in summary.items() if key not in {"checks", "passed"}},
        "development_gate": {
            "inference_unit": "city",
            "checks": summary["checks"],
            "passed": passed,
            "decision": (
                "authorize_conditional_source_closed_loop_development"
                if passed
                else "retain_sample_coherent_rigid_cfcmt_and_reject_source_claim"
            ),
        },
        "claim_boundary": (
            "V150J tests one deterministic conditional mechanism representation "
            "on seven development cities. Passing cannot establish fresh-city "
            "or closed-loop efficacy."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    target = subparsers.add_parser("target")
    target.add_argument("--target-city", required=True)
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
    target.add_argument("--v150i-rejection", type=Path, required=True)
    target.add_argument("--v150i-rejection-sha256", required=True)
    target.add_argument("--out", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", nargs=7, type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150J result: {args.out}")
    if args.mode == "target":
        result = run_target(
            v150i_rejection_path=args.v150i_rejection,
            expected_v150i_rejection_sha256=args.v150i_rejection_sha256,
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
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
