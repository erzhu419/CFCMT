"""Cross-fit complete source/mechanism selection before rigid integration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_identity_verified_mechanism_prior import (
    AGGREGATE_PROTOCOL as V150E_AGGREGATE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    MINIMUM_OOF_GAIN,
    _paired_effects,
    _read_json,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    _city_bootstrap,
    run_target as _run_rigid_target,
)
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    _paired_bootstrap,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v150g-crossfitted-source-selector-target-v1"
AGGREGATE_PROTOCOL = "tsc-v150g-crossfitted-source-selector-aggregate-v1"
METHOD_PROTOCOL = "rigid-source-mechanism-prior-crossfitted-selection-v1"


def run_target(
    *,
    v150e_rejection_path: Path,
    expected_v150e_rejection_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    if _sha256(v150e_rejection_path) != str(expected_v150e_rejection_sha256):
        raise ValueError("V150E rejection identity changed")
    rejection = _read_json(v150e_rejection_path)
    if (
        rejection.get("protocol") != V150E_AGGREGATE_PROTOCOL
        or rejection.get("development_gate", {}).get("passed") is not False
        or rejection.get("development_gate", {}).get("decision")
        != "retain_rigid_cfcmt_and_reject_source_closed_loop_claim"
    ):
        raise ValueError("V150G requires the frozen V150E rejection")
    result = _run_rigid_target(
        **kwargs,
        result_protocol=RESULT_PROTOCOL,
        scientific_status="seven-city-crossfitted-source-development-target",
        method_protocol=METHOD_PROTOCOL,
        selector_mode="crossfitted_target_and_matched_placebo_gain",
        minimum_identity_gain=MINIMUM_OOF_GAIN,
        matched_placebo_follows_source_admission=True,
        version_label="v150g",
        claim_boundary=(
            "V150G is a post-V150E seven-city redevelopment test. Every B25 "
            "fold evaluates a complete source/mechanism selection made without "
            "that fold. Evaluation groups remain untouched, but all cities are "
            "development data and cannot provide final confirmation."
        ),
    )
    result["inputs"]["v150e_rejection_sha256"] = str(
        expected_v150e_rejection_sha256
    )
    result["inputs"]["v150e_rejection_protocol"] = str(rejection["protocol"])
    return result


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    by_city = {str(row.get("target_city")): row for row in rows}
    if (
        len(rows) != len(EXPECTED_CITY_GROUPS)
        or set(by_city) != set(EXPECTED_CITY_GROUPS)
        or any(row.get("protocol") != RESULT_PROTOCOL for row in rows)
        or any(
            row.get("source_null_contract", {}).get("summaries_equal") is not True
            for row in rows
        )
    ):
        raise ValueError("V150G aggregate requires seven valid target results")
    target_effects, target_city_means = _paired_effects(
        by_city, "selected_source_corrected_rigid", "rigid_target_only"
    )
    placebo_effects, placebo_city_means = _paired_effects(
        by_city,
        "selected_source_corrected_rigid",
        "same_candidate_placebo_corrected_rigid",
    )
    admitted = {
        city: bool(by_city[city]["method"]["selector"]["admitted"])
        for city in EXPECTED_CITY_GROUPS
    }
    admitted_cities = [city for city, value in admitted.items() if value]
    tolerance = 1e-12
    gates = {
        "mean_selected_effect_improves_rigid": _city_bootstrap(
            target_city_means
        )["mean"]
        < 0.0,
        "no_city_regresses_rigid": max(target_city_means.values()) <= tolerance,
        "at_least_two_cities_admit_crossfitted_source": len(admitted_cities) >= 2,
        "every_admitted_city_improves_rigid": all(
            target_city_means[city] < 0.0 for city in admitted_cities
        ),
        "every_admitted_city_beats_matched_placebo": all(
            placebo_city_means[city] < 0.0 for city in admitted_cities
        ),
        "every_rejected_city_exactly_falls_back": all(
            abs(target_city_means[city]) <= tolerance
            for city in EXPECTED_CITY_GROUPS
            if not admitted[city]
        ),
        "mean_selected_effect_beats_matched_placebo": _city_bootstrap(
            placebo_city_means
        )["mean"]
        < 0.0,
    }
    passed = all(gates.values())
    return {
        "protocol": AGGREGATE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development_gate_pass" if passed else "development_gate_reject"
        ),
        "city_count": len(rows),
        "target_seed_unit_count": len(target_effects),
        "selected_effect_vs_rigid_target_only": {
            "seed_unit": _paired_bootstrap(target_effects),
            "city_unit": {
                **_city_bootstrap(target_city_means),
                "city_means": target_city_means,
            },
        },
        "selected_effect_vs_same_candidate_placebo": {
            "seed_unit": _paired_bootstrap(placebo_effects),
            "city_unit": {
                **_city_bootstrap(placebo_city_means),
                "city_means": placebo_city_means,
            },
        },
        "admitted_cities": admitted,
        "selected_candidates": {
            city: by_city[city]["method"]["selector"]
            for city in EXPECTED_CITY_GROUPS
        },
        "development_gate": {
            "inference_unit": "city",
            "checks": gates,
            "passed": passed,
            "decision": (
                "authorize_crossfitted_source_closed_loop_development"
                if passed
                else "retain_rigid_cfcmt_and_reject_source_closed_loop_claim"
            ),
        },
        "claim_boundary": (
            "V150G cross-fits the complete selector on seven development cities. "
            "A pass is not fresh-seed or fresh-city confirmation."
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
    target.add_argument("--v150e-rejection", type=Path, required=True)
    target.add_argument("--v150e-rejection-sha256", required=True)
    target.add_argument("--out", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", nargs=7, type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150G result: {args.out}")
    if args.mode == "target":
        result = run_target(
            v150e_rejection_path=args.v150e_rejection,
            expected_v150e_rejection_sha256=args.v150e_rejection_sha256,
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
