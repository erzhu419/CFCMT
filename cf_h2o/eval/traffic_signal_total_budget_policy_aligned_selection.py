"""Evaluate a policy-aligned discrete source gate under one B25 target budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from cf_h2o.eval.traffic_signal_total_budget_causal_source_weighting import (
    POLICY_ALIGNED_SOURCE_MASSES,
    run_total_budget_source_weighting,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v142-total-budget-policy-aligned-source-selection-v1"
PRIMARY_BUDGET = 25
EXPECTED_SELECTOR_SEED_COUNT = 22
ADAPTATION_SEED_COUNT = 11
EVALUATION_SEED_COUNT = 11


def split_policy_aligned_seeds(
    seeds: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered = tuple(sorted(int(value) for value in seeds))
    if (
        len(ordered) != EXPECTED_SELECTOR_SEED_COUNT
        or len(set(ordered)) != len(ordered)
    ):
        raise ValueError("V142 requires exactly 22 unique selector seeds")
    adaptation = ordered[::2]
    evaluation = ordered[1::2]
    if (
        len(adaptation) != ADAPTATION_SEED_COUNT
        or len(evaluation) != EVALUATION_SEED_COUNT
    ):
        raise ValueError("V142 alternating seed partition changed")
    return adaptation, evaluation


def run_total_budget_policy_aligned_selection(
    *,
    budget: int,
    selector_seeds: Sequence[int],
    **kwargs: Any,
) -> dict[str, Any]:
    if int(budget) != PRIMARY_BUDGET:
        raise ValueError("V142 is frozen to the B25 development diagnostic")
    adaptation, evaluation = split_policy_aligned_seeds(selector_seeds)
    result = run_total_budget_source_weighting(
        budget=int(budget),
        selector_seeds=selector_seeds,
        seed_partition=(adaptation, evaluation),
        selection_strategy="policy_aligned_discrete",
        result_protocol=RESULT_PROTOCOL,
        run_label="v142",
        adaptation_partition_label="alternating_sorted_selector_seeds_11_of_22",
        **kwargs,
    )
    passed = bool(result["development_gate"]["selector_relative_passed"])
    result["scientific_status"] = (
        "post-hoc-jinan-development-diagnostic-not-confirmation"
    )
    result["candidate_family"] = {
        "source_masses": list(POLICY_ALIGNED_SOURCE_MASSES),
        "members": "source_null + seven single-source arms + uniform-source arm",
        "selection_objective": "nested held-out-seed downstream policy value",
    }
    result["development_gate"]["primary_budget_passed"] = passed
    result["development_gate"]["decision"] = (
        "authorize_preregistered_multicity_development"
        if passed
        else "reject_policy_aligned_source_selection"
    )
    result["claim_boundary"] = (
        "V142 is a post-V140 Jinan development diagnostic. A pass identifies "
        "a policy-aligned source-selection mechanism worth multicity testing; "
        "it cannot authorize Boston or any untouched-city claim directly."
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--fit-result-sha256", required=True)
    parser.add_argument("--fit-protocol", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--selector-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-sha256", required=True)
    parser.add_argument("--selector-scenario", required=True)
    parser.add_argument("--selector-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--selector-collection-shards", type=int, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fit-workers", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V142 result: {args.out}")
    result = run_total_budget_policy_aligned_selection(
        budget=args.budget,
        fit_result_path=args.fit_result,
        expected_fit_result_sha256=args.fit_result_sha256,
        fit_protocol_path=args.fit_protocol,
        source_cache_root=args.source_cache_root,
        source_manifest_path=args.source_manifest,
        source_cache_audit_path=args.source_cache_audit,
        selector_cache_root=args.selector_cache_root,
        selector_manifest_path=args.selector_manifest,
        selector_cache_audit_path=args.selector_cache_audit,
        expected_selector_cache_audit_sha256=args.selector_cache_audit_sha256,
        selector_scenario=args.selector_scenario,
        selector_seeds=args.selector_seeds,
        selector_collection_shards=args.selector_collection_shards,
        conversion_root=args.conversion_root,
        cache_workers=max(1, int(args.cache_workers)),
        fit_workers=max(1, int(args.fit_workers)),
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": (
                    "PASS"
                    if result["development_gate"]["selector_relative_passed"]
                    else "REJECT"
                ),
                "protocol": result["protocol"],
                "budget": result["budget"],
                "selected_candidate": result["source_weight_gate"].get(
                    "selected_full_candidate_raw"
                ),
                "development_gate": result["development_gate"],
                "effective_source_weights": result["effective_source_weights"],
                "paired_effects": result["paired_effects"],
                "runtime_seconds": result["runtime_seconds"],
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
