"""Audit exact cKDTree support against the frozen brute-force calculation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (
    _read_json,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_guard_diagnostic import (
    HIERARCHICAL,
    hierarchical_variant_rows,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    AUTHORIZATION_DECISION,
    CACHE_AUDIT_DECISION,
    CACHE_AUDIT_PROTOCOL,
    FROZEN_PROTOCOL,
    RESULT_PROTOCOL as OFFLINE_RESULT_PROTOCOL,
    _load_frozen_city_model,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    oof_action_records,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.target_action_support import TargetActionSupport


RESULT_PROTOCOL = "tsc-v62r58-target-support-exact-tree-equivalence-audit-v1"
AUDIT_DECISION = "authorize_exact_tree_hierarchical_closed_loop_runtime"
MAX_ABSOLUTE_DIFFERENCE = 1e-12


class _BruteforceSupport:
    def __init__(self, support: TargetActionSupport) -> None:
        self.support_model = support

    def support(self, dataset) -> np.ndarray:
        return self.support_model.support_bruteforce(dataset)


def _compare_records(
    exact: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact_by_group = {str(row["group_id"]): row for row in exact}
    reference_by_group = {str(row["group_id"]): row for row in reference}
    if (
        len(exact_by_group) != len(exact)
        or len(reference_by_group) != len(reference)
        or set(exact_by_group) != set(reference_by_group)
    ):
        raise ValueError("support equivalence action-group identity changed")
    maximum_support_difference = 0.0
    maximum_trust_difference = 0.0
    common_prediction_equal = True
    for group in exact_by_group:
        left = exact_by_group[group]
        right = reference_by_group[group]
        maximum_support_difference = max(
            maximum_support_difference,
            abs(float(left["local_action_support"]) - float(right["local_action_support"])),
        )
        maximum_trust_difference = max(
            maximum_trust_difference,
            abs(float(left["context_trust"]) - float(right["context_trust"])),
        )
        for key in (
            "scenario",
            "learned_differs",
            "predicted_delta",
            "uncertainty",
            "relative_rule_gap",
            "actual_delta",
        ):
            if isinstance(left[key], (float, int)) and not isinstance(left[key], bool):
                equal = np.isclose(
                    float(left[key]), float(right[key]), rtol=0.0, atol=1e-12
                )
            else:
                equal = left[key] == right[key]
            common_prediction_equal = common_prediction_equal and bool(equal)
    passed = (
        common_prediction_equal
        and maximum_support_difference <= MAX_ABSOLUTE_DIFFERENCE
        and maximum_trust_difference <= MAX_ABSOLUTE_DIFFERENCE
    )
    return {
        "action_group_count": len(exact_by_group),
        "maximum_absolute_support_difference": maximum_support_difference,
        "maximum_absolute_context_trust_difference": maximum_trust_difference,
        "common_prediction_equal": common_prediction_equal,
        "passed": passed,
    }


def run_target_support_equivalence_audit(
    *,
    evaluation_cache_root: Path,
    cache_audit_path: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    protocol_path: Path,
    freeze_audit_path: Path,
    freeze_root: Path,
    offline_result_path: Path,
    expected_offline_result_sha256: str,
    workers: int,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    freeze_audit = _read_json(freeze_audit_path)
    cache_audit = _read_json(cache_audit_path)
    offline_result = _read_json(offline_result_path)
    if (
        protocol.get("protocol") != FROZEN_PROTOCOL
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("decision") != CACHE_AUDIT_DECISION
        or _sha256(offline_result_path) != str(expected_offline_result_sha256)
        or offline_result.get("protocol") != OFFLINE_RESULT_PROTOCOL
        or offline_result.get("decision") != AUTHORIZATION_DECISION
        or not bool(
            offline_result.get("offline_confirmation_gate", {}).get("passed", False)
        )
        or offline_result.get("cache_audit_sha256") != _sha256(cache_audit_path)
        or offline_result.get("freeze_audit_sha256") != _sha256(freeze_audit_path)
        or offline_result.get("frozen_protocol_sha256") != _sha256(protocol_path)
    ):
        raise ValueError("target-support equivalence evidence chain changed")

    offline = dict(protocol["fresh_offline_confirmation"])
    seeds = tuple(int(value) for value in offline["seeds"])
    city_scenarios = {
        str(city): tuple(str(value) for value in scenarios)
        for city, scenarios in protocol["city_scenarios"].items()
    }
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(
        Path(conversion_root).resolve()
    )
    manifest = load_traffic_signal_manifest(external_manifest_path)
    bank, bank_audit = load_frozen_counterfactual_bank(
        evaluation_cache_root,
        manifest,
        seeds=seeds,
        collection_shards=int(offline["collection_shards_per_scenario_seed"]),
        workers=max(int(workers), 1),
    )
    city_results: dict[str, Any] = {}
    for city, scenarios in sorted(city_scenarios.items()):
        _, model = _load_frozen_city_model(
            city=city,
            freeze_root=freeze_root,
            protocol=protocol,
            freeze_audit=freeze_audit,
        )
        dataset = _merge_city_datasets(bank, scenarios, city=city)
        common = {
            "models": model["models"],
            "selected_candidate": str(model["selected_candidate"]),
            "fold_index": 0,
            "validation_seed": 0,
            "scenarios": scenarios,
        }
        started = time.perf_counter()
        exact_pressure = oof_action_records(
            dataset, target_support=model["pressure_target_support"], **common
        )
        exact_conformal = oof_action_records(
            dataset, target_support=model["conformal_target_support"], **common
        )
        exact_seconds = float(time.perf_counter() - started)
        started = time.perf_counter()
        brute_pressure = oof_action_records(
            dataset,
            target_support=_BruteforceSupport(model["pressure_target_support"]),
            **common,
        )
        brute_conformal = oof_action_records(
            dataset,
            target_support=_BruteforceSupport(model["conformal_target_support"]),
            **common,
        )
        brute_seconds = float(time.perf_counter() - started)
        pressure_comparison = _compare_records(exact_pressure, brute_pressure)
        conformal_comparison = _compare_records(exact_conformal, brute_conformal)
        exact_rows = hierarchical_variant_rows(
            pressure_records=exact_pressure,
            conformal_records=exact_conformal,
            regularizer=model["regularizer"],
            guard=model["guard"],
        )[HIERARCHICAL]
        brute_rows = hierarchical_variant_rows(
            pressure_records=brute_pressure,
            conformal_records=brute_conformal,
            regularizer=model["regularizer"],
            guard=model["guard"],
        )[HIERARCHICAL]
        exact_selected = {
            str(row["group_id"]): bool(row["selected"]) for row in exact_rows
        }
        brute_selected = {
            str(row["group_id"]): bool(row["selected"]) for row in brute_rows
        }
        selection_equal = exact_selected == brute_selected
        city_results[city] = {
            "scenarios": list(scenarios),
            "action_group_count": len(exact_rows),
            "pressure_support": pressure_comparison,
            "conformal_support": conformal_comparison,
            "hierarchical_selection_identical": selection_equal,
            "selected_action_count": sum(exact_selected.values()),
            "exact_tree_seconds": exact_seconds,
            "bruteforce_seconds": brute_seconds,
            "measured_speedup": (
                brute_seconds / exact_seconds if exact_seconds > 0.0 else None
            ),
            "passed": bool(
                pressure_comparison["passed"]
                and conformal_comparison["passed"]
                and selection_equal
            ),
        }
    gate = {
        "all_cities_present": set(city_results) == set(city_scenarios),
        "all_numeric_equivalence_gates_passed": bool(city_results)
        and all(row["passed"] for row in city_results.values()),
        "offline_authorization_unchanged": True,
        "models_and_thresholds_unchanged": True,
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "runtime": _runtime_metadata(),
        "claim_boundary": (
            "Numerical implementation equivalence only. The audit does not add "
            "outcome evidence or alter any fitted model, threshold, or selector."
        ),
        "frozen_protocol_sha256": _sha256(protocol_path),
        "freeze_audit_sha256": _sha256(freeze_audit_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "offline_result_sha256": _sha256(offline_result_path),
        "evaluation_cache_sha256": cache_audit["cache_sha256"],
        "evaluation_cache_audit": bank_audit,
        "maximum_allowed_absolute_difference": MAX_ABSOLUTE_DIFFERENCE,
        "city_results": city_results,
        "equivalence_gate": gate,
        "status": "PASS" if gate["passed"] else "FAIL",
        "decision": AUDIT_DECISION if gate["passed"] else "reject_exact_tree_runtime",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-cache-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--offline-result", type=Path, required=True)
    parser.add_argument("--expected-offline-result-sha256", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite support equivalence: {args.out}")
    payload = run_target_support_equivalence_audit(
        evaluation_cache_root=args.evaluation_cache_root,
        cache_audit_path=args.cache_audit,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        protocol_path=args.protocol,
        freeze_audit_path=args.freeze_audit,
        freeze_root=args.freeze_root,
        offline_result_path=args.offline_result,
        expected_offline_result_sha256=args.expected_offline_result_sha256,
        workers=args.workers,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
