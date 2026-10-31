#!/usr/bin/env python3
"""Freeze the corrected historical-prefix demand diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_external_historical_demand_model_diagnostic import (  # noqa: E402
    BASE_VARIANT,
    DIAGNOSTIC_PROTOCOL,
    HISTORICAL_VARIANT,
    feature_names,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from cf_h2o.traffic_signal.demand_schedule_context import _resolve_cfg_paths  # noqa: E402


PARENT_PROTOCOL_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v56_external_v9_"
    "local_demand_model_diagnostic.json"
)
PRIVILEGED_RESULT_PATH = Path(
    "cf_h2o/results/cluster/tsc_v73r69_external_v9_local_demand_"
    "model_diagnostic_20260810/diagnostic_v1.json"
)
PRIVILEGED_AUDIT_PATH = Path(
    "cf_h2o/results/cluster/tsc_v73r69_external_v9_local_demand_"
    "model_diagnostic_20260810/audit_v1.json"
)
ABANDONED_V75_PROTOCOL_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v58_external_v9_"
    "local_demand_closed_loop_development.json"
)
ABANDONED_V75_ROOT = Path(
    "cf_h2o/results/cluster/tsc_v75r71_external_v9_local_demand_"
    "closed_loop_development_20260810"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_count(path: Path) -> int:
    return (
        sum(1 for item in Path(path).rglob("*") if item.is_file())
        if Path(path).exists()
        else 0
    )


def _route_element_counts(conversion_root: Path, scenarios: Sequence[str]) -> dict[str, Any]:
    rows = {}
    for scenario in scenarios:
        sumocfg = Path(conversion_root) / scenario / f"{scenario}.sumocfg"
        counts = {"vehicle": 0, "trip": 0, "flow": 0}
        for route_file in _resolve_cfg_paths(sumocfg, "route-files"):
            root = ET.parse(route_file).getroot()
            for key in counts:
                counts[key] += len(root.findall(key))
        rows[str(scenario)] = counts
    return rows


def build_protocol(
    *,
    diagnostic_output: Path,
    successor_output_root: Path,
    future_closed_loop_root: Path,
    future_validation_root: Path,
    future_prospective_root: Path,
) -> dict[str, Any]:
    parent = _read_json(PARENT_PROTOCOL_PATH)
    cache_protocol_path = Path(parent["cache_protocol"]["path"])
    cache_protocol = _read_json(cache_protocol_path)
    privileged_result = _read_json(PRIVILEGED_RESULT_PATH)
    privileged_audit = _read_json(PRIVILEGED_AUDIT_PATH)
    abandoned_v75 = _read_json(ABANDONED_V75_PROTOCOL_PATH)
    old_feature_names = privileged_result["cities"]["jinan"]["variants"][
        "topology_local_squared_raw_scaled"
    ]["feature_names"]
    if (
        parent.get("protocol")
        != "tsc-v73r69-external-v9-local-demand-model-diagnostic-v1"
        or privileged_result.get("status") != "PASS"
        or privileged_audit.get("status") != "PASS"
        or privileged_audit.get("decision")
        != "authorize_oof_qualified_city_successor_freeze_only"
        or not any("local_arrivals_next_" in name for name in old_feature_names)
        or abandoned_v75.get("protocol")
        != "tsc-v75r71-external-v9-local-demand-closed-loop-development-v1"
        or _file_count(ABANDONED_V75_ROOT) != 0
    ):
        raise ValueError("historical correction parent evidence changed")
    environment = cache_protocol["environment"]
    conversion_root = Path(environment["conversion_root"])
    scenarios = [
        scenario
        for values in cache_protocol["city_scenarios"].values()
        for scenario in values
    ]
    route_counts = _route_element_counts(conversion_root, scenarios)
    if not all(row["vehicle"] > 0 for row in route_counts.values()):
        raise ValueError("v9 routes are no longer individual-vehicle schedules")
    future_counts = {
        "diagnostic_output": int(Path(diagnostic_output).is_file()),
        "successor_output_root": _file_count(successor_output_root),
        "future_closed_loop_root": _file_count(future_closed_loop_root),
        "future_validation_root": _file_count(future_validation_root),
        "future_prospective_root": _file_count(future_prospective_root),
    }
    if any(future_counts.values()):
        raise FileExistsError(f"historical diagnostic future evidence exists: {future_counts}")
    sources = (
        "cf_h2o/traffic_signal/historical_local_demand_context.py",
        "cf_h2o/eval/traffic_signal_external_historical_demand_model_diagnostic.py",
        "scripts/cluster/run_tsc_external_historical_demand_model_diagnostic.py",
    )
    diagnostic = parent["diagnostic"]
    return {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_privileged_future_feature_rejection_pre_historical_oof_outcome",
        "parent_protocol": {
            "path": str(PARENT_PROTOCOL_PATH),
            "sha256": _sha256(PARENT_PROTOCOL_PATH),
        },
        "privileged_future_diagnostic": {
            "result_path": str(PRIVILEGED_RESULT_PATH),
            "result_sha256": _sha256(PRIVILEGED_RESULT_PATH),
            "audit_path": str(PRIVILEGED_AUDIT_PATH),
            "audit_sha256": _sha256(PRIVILEGED_AUDIT_PATH),
            "rejected_runtime_features": [
                name for name in old_feature_names if "local_arrivals_next_" in name
            ],
            "classification": "development_upper_bound_not_deployable_controller",
        },
        "abandoned_v75": {
            "protocol_path": str(ABANDONED_V75_PROTOCOL_PATH),
            "protocol_sha256": _sha256(ABANDONED_V75_PROTOCOL_PATH),
            "result_file_count_at_correction_freeze": 0,
            "reason": "exact_future_individual_departure_and_route_information",
        },
        "cache_protocol": {
            "path": str(cache_protocol_path),
            "sha256": _sha256(cache_protocol_path),
        },
        "cache_audit": parent["cache_audit"],
        "proposal_parent_protocol": parent["proposal_parent_protocol"],
        "hierarchical_freeze_audit": parent["hierarchical_freeze_audit"],
        "environment": {
            **environment,
            "route_element_counts": route_counts,
        },
        "runtime_executable_sources": {
            path: _sha256(Path(path)) for path in sources
        },
        "diagnostic": {
            "classification": "target_labeled_few_shot_historical_prefix_development",
            "seeds": diagnostic["seeds"],
            "city_scenarios": diagnostic["city_scenarios"],
            "cross_fitting": "leave_one_simulator_seed_out",
            "target": "450_second_counterfactual_candidate_minus_phase_cost",
            "target_transform": "raw_scenario_robust_scaled",
            "variants": {
                BASE_VARIANT: {
                    "feature_names": list(feature_names(BASE_VARIANT)),
                    "information_budget": "current_candidate_state_and_static_topology",
                },
                HISTORICAL_VARIANT: {
                    "feature_names": list(feature_names(HISTORICAL_VARIANT)),
                    "information_budget": (
                        "current_candidate_state_static_topology_and_route_projected_"
                        "arrivals_with_projected_arrival_time_no_later_than_t"
                    ),
                },
            },
            "acceptance_quantiles": diagnostic["acceptance_quantiles"],
            "model": diagnostic["model"],
            "bootstrap": diagnostic["bootstrap"],
            "selection_rule": diagnostic["selection_rule"],
            "advancement_rule": (
                "historical_variant_must_pass_original_v70_rule_and_have_lower_"
                "robust_score_and_mean_delta_than_no_local_baseline"
            ),
        },
        "future_file_counts_at_freeze": future_counts,
        "validation": parent["validation"],
        "prospective_confirmation": parent["prospective_confirmation"],
        "claim_boundary": (
            "Only six disclosed adaptation seeds may be used. Future individual "
            "departures/routes and all sealed validation/prospective seeds are excluded."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--successor-output-root", type=Path, required=True)
    parser.add_argument("--future-closed-loop-root", type=Path, required=True)
    parser.add_argument("--future-validation-root", type=Path, required=True)
    parser.add_argument("--future-prospective-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite historical protocol: {args.out}")
    payload = build_protocol(
        diagnostic_output=args.diagnostic_output,
        successor_output_root=args.successor_output_root,
        future_closed_loop_root=args.future_closed_loop_root,
        future_validation_root=args.future_validation_root,
        future_prospective_root=args.future_prospective_root,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "route_element_counts": payload["environment"]["route_element_counts"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
