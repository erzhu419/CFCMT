"""Evaluate complete source selection over a nested target-information curve."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_crossfitted_source_selector import (
    AGGREGATE_PROTOCOL as V150G_AGGREGATE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
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


RESULT_PROTOCOL = "tsc-v150h-source-identifiability-budget-target-v1"
AGGREGATE_PROTOCOL = "tsc-v150h-source-identifiability-budget-aggregate-v1"
METHOD_PROTOCOL = "rigid-source-mechanism-prior-budget-crossfit-v1"
TARGET_BUDGETS = (25, 50, 100)
PRIMARY_BUDGET = 100
COMMON_EVALUATION_RESERVE_BUDGET = 100


def run_target(
    *,
    target_budget: int,
    v150g_rejection_path: Path,
    expected_v150g_rejection_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    budget = int(target_budget)
    if budget not in TARGET_BUDGETS:
        raise ValueError(f"unsupported V150H target budget: {budget}")
    if _sha256(v150g_rejection_path) != str(expected_v150g_rejection_sha256):
        raise ValueError("V150G rejection identity changed")
    rejection = _read_json(v150g_rejection_path)
    if (
        rejection.get("protocol") != V150G_AGGREGATE_PROTOCOL
        or rejection.get("development_gate", {}).get("passed") is not False
        or rejection.get("development_gate", {}).get("decision")
        != "retain_rigid_cfcmt_and_reject_source_closed_loop_claim"
    ):
        raise ValueError("V150H requires the frozen V150G rejection")
    result = _run_rigid_target(
        **kwargs,
        target_budget=budget,
        evaluation_reserve_budget=COMMON_EVALUATION_RESERVE_BUDGET,
        result_protocol=RESULT_PROTOCOL,
        scientific_status="seven-city-source-identifiability-budget-development-target",
        method_protocol=METHOD_PROTOCOL,
        selector_mode="crossfitted_target_and_matched_placebo_gain",
        matched_placebo_follows_source_admission=True,
        version_label="v150h",
        claim_boundary=(
            f"V150H is a post-V150G development diagnostic using B{budget} "
            "target-labelled action groups. All budgets evaluate on the same "
            "groups outside the nested B100 reserve. It is target-offline "
            "adaptation, not zero-shot, closed-loop, or fresh-city confirmation."
        ),
    )
    result["inputs"]["v150g_rejection_sha256"] = str(
        expected_v150g_rejection_sha256
    )
    result["inputs"]["v150g_rejection_protocol"] = str(rejection["protocol"])
    return result


def _budget_summary(by_city: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
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
    checks = {
        "mean_selected_effect_improves_rigid": (
            _city_bootstrap(target_city_means)["mean"] < 0.0
        ),
        "mean_selected_effect_beats_matched_placebo": (
            _city_bootstrap(placebo_city_means)["mean"] < 0.0
        ),
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
    }
    return {
        "source_admission_count": len(admitted_cities),
        "admitted_cities": admitted,
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
        "selected_candidates": {
            city: by_city[city]["method"]["selector"]
            for city in EXPECTED_CITY_GROUPS
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def aggregate_results(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_result_protocol: str = RESULT_PROTOCOL,
    aggregate_protocol: str = AGGREGATE_PROTOCOL,
    experiment_label: str = "V150H",
    pass_decision: str = "authorize_b100_source_closed_loop_development",
) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    expected_keys = {
        (city, budget)
        for city in EXPECTED_CITY_GROUPS
        for budget in TARGET_BUDGETS
    }
    by_key = {
        (str(row.get("target_city")), int(row.get("target_budget", -1))): row
        for row in rows
    }
    if (
        len(rows) != len(expected_keys)
        or set(by_key) != expected_keys
        or any(row.get("protocol") != expected_result_protocol for row in rows)
        or any(
            row.get("source_null_contract", {}).get("summaries_equal") is not True
            for row in rows
        )
    ):
        raise ValueError(
            f"{experiment_label} aggregate requires 21 valid city-budget results"
        )

    for city in EXPECTED_CITY_GROUPS:
        selected = {
            budget: set(
                by_key[(city, budget)]["selection_audit"]["selected_group_ids"]
            )
            for budget in TARGET_BUDGETS
        }
        reserves = {
            budget: tuple(
                by_key[(city, budget)]["evaluation_reserve_audit"][
                    "selected_group_ids"
                ]
            )
            for budget in TARGET_BUDGETS
        }
        if any(len(selected[budget]) != budget for budget in TARGET_BUDGETS):
            raise ValueError(
                f"{city}: {experiment_label} target budget accounting changed"
            )
        if not selected[25].issubset(selected[50]) or not selected[50].issubset(
            selected[100]
        ):
            raise ValueError(
                f"{city}: {experiment_label} target budgets are not nested"
            )
        if len(set(reserves.values())) != 1 or len(reserves[25]) != 100:
            raise ValueError(
                f"{city}: {experiment_label} common evaluation reserve changed"
            )

    summaries = {
        str(budget): _budget_summary(
            {
                city: by_key[(city, budget)]
                for city in EXPECTED_CITY_GROUPS
            }
        )
        for budget in TARGET_BUDGETS
    }
    admission_counts = [
        int(summaries[str(budget)]["source_admission_count"])
        for budget in TARGET_BUDGETS
    ]
    macro_effects = [
        float(
            summaries[str(budget)]["selected_effect_vs_rigid_target_only"][
                "city_unit"
            ]["mean"]
        )
        for budget in TARGET_BUDGETS
    ]
    primary = summaries[str(PRIMARY_BUDGET)]
    passed = bool(primary["passed"])
    return {
        "protocol": str(aggregate_protocol),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development_gate_pass" if passed else "development_gate_reject"
        ),
        "city_count": len(EXPECTED_CITY_GROUPS),
        "target_budgets": list(TARGET_BUDGETS),
        "primary_budget": PRIMARY_BUDGET,
        "common_evaluation_reserve_budget": COMMON_EVALUATION_RESERVE_BUDGET,
        "budget_results": summaries,
        "curve_diagnostics": {
            "source_admission_counts": dict(
                zip((str(value) for value in TARGET_BUDGETS), admission_counts)
            ),
            "selected_macro_effects_vs_rigid": dict(
                zip((str(value) for value in TARGET_BUDGETS), macro_effects)
            ),
            "admission_count_nondecreasing": all(
                left <= right
                for left, right in zip(admission_counts, admission_counts[1:])
            ),
        },
        "development_gate": {
            "primary_budget": PRIMARY_BUDGET,
            "inference_unit": "city",
            "checks": primary["checks"],
            "passed": passed,
            "decision": (
                str(pass_decision)
                if passed
                else "retain_rigid_cfcmt_and_reject_source_closed_loop_claim"
            ),
        },
        "claim_boundary": (
            f"{experiment_label} varies target-labelled information on seven development "
            "cities while fixing one common evaluation set. A pass authorizes "
            "B100 closed-loop development only, not efficacy confirmation."
        ),
    }


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
    target.add_argument("--v150g-rejection", type=Path, required=True)
    target.add_argument("--v150g-rejection-sha256", required=True)
    target.add_argument("--out", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", nargs=21, type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150H result: {args.out}")
    if args.mode == "target":
        result = run_target(
            target_budget=args.target_budget,
            v150g_rejection_path=args.v150g_rejection,
            expected_v150g_rejection_sha256=args.v150g_rejection_sha256,
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
