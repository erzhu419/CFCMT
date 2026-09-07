"""Audit semantic equivalence of two SUMO counterfactual cache implementations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.traffic_signal.dataset_cache import (
    atomic_write_json,
    load_mechanism_dataset,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


RESULT_PROTOCOL = "tsc-counterfactual-restore-equivalence-audit-v1"
OPTIMIZED_RELOAD_MODE = "full_network_load"
OPTIMIZED_BRANCH_MODE = "snapshot_full_reload_second_pass"
OPTIMIZED_HALTED_READER_PROTOCOL = "libsumo-lane-subscription-halted-cost-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _elapsed_seconds(summary: Mapping[str, Any]) -> float:
    value = summary.get("elapsed_seconds")
    if value is None and isinstance(summary.get("runtime"), Mapping):
        value = summary["runtime"].get("elapsed_seconds")
    elapsed = float(value)
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("counterfactual summary has no positive elapsed_seconds")
    return elapsed


def _array_comparison(
    legacy: np.ndarray,
    optimized: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    legacy = np.asarray(legacy)
    optimized = np.asarray(optimized)
    same_shape = legacy.shape == optimized.shape
    if not same_shape:
        return {
            "same_shape": False,
            "legacy_shape": list(legacy.shape),
            "optimized_shape": list(optimized.shape),
            "passed": False,
        }
    if legacy.dtype.kind in "OUS" or optimized.dtype.kind in "OUS":
        exact = bool(np.array_equal(legacy.astype(str), optimized.astype(str)))
        return {
            "same_shape": True,
            "exact": exact,
            "passed": exact,
        }
    difference = np.abs(
        legacy.astype(float, copy=False) - optimized.astype(float, copy=False)
    )
    max_abs_difference = float(np.max(difference)) if difference.size else 0.0
    exact = bool(np.array_equal(legacy, optimized))
    passed = bool(np.allclose(legacy, optimized, rtol=0.0, atol=float(tolerance)))
    return {
        "same_shape": True,
        "exact": exact,
        "max_abs_difference": max_abs_difference,
        "absolute_tolerance": float(tolerance),
        "passed": passed,
    }


def _aligned_metadata(dataset: MechanismDataset, key: str) -> np.ndarray:
    values = np.asarray(dataset.metadata.get(key, ()))
    if values.shape != (dataset.size,):
        raise ValueError(f"counterfactual row metadata is not aligned: {key}")
    return values


def _semantic_metadata(dataset: MechanismDataset) -> dict[str, Any]:
    keys = (
        "scenario",
        "seed",
        "duration_sec",
        "control_interval_sec",
        "warmup_sec",
        "max_focal_tls",
        "counterfactual_horizon_intervals",
        "counterfactual_horizon_sec",
        "controllable_tls_ids",
        "controllable_tls_count",
        "covered_tls_ids",
        "covered_tls_count",
        "tls_coverage_fraction",
        "counterfactual_branches",
        "state_restores",
        "behavior_trace_sha256",
        "behavior_interval_count",
        "captured_snapshot_count",
        "protocol",
        "strict_safety_monitoring",
        "counterfactual_estimand",
        "counterfactual_cost_mode",
        "counterfactual_cost_scope",
        "counterfactual_cost_population",
        "counterfactual_cost_normalization",
        "mechanism_output_names",
        "local_mechanism_horizon_sec",
        "rollout_value_horizon_sec",
        "rollout_prefix_horizons_sec",
        "mechanism_estimands",
        "behavior_policy",
        "sumo_execution_protocol",
        "phase_execution_audit",
        "behavior_safety_audit",
        "counterfactual_safety_audit",
        "counterfactual_replay_audit",
    )
    return {key: dataset.metadata.get(key) for key in keys}


def compare_counterfactual_implementations(
    legacy: MechanismDataset,
    optimized: MechanismDataset,
    *,
    legacy_elapsed_seconds: float,
    optimized_elapsed_seconds: float,
    tolerance: float = 1e-8,
    minimum_speedup: float = 1.25,
) -> dict[str, Any]:
    """Compare every model-visible value and the operational speed contract."""

    schema = {
        "feature_names": legacy.feature_names == optimized.feature_names,
        "context_names": legacy.context_names == optimized.context_names,
        "prior_names": set(legacy.priors) == set(optimized.priors),
        "target_names": set(legacy.targets) == set(optimized.targets),
        "row_count": legacy.size == optimized.size,
    }
    arrays: dict[str, Any] = {
        "features": _array_comparison(
            legacy.features, optimized.features, tolerance=tolerance
        ),
        "context": _array_comparison(
            legacy.context, optimized.context, tolerance=tolerance
        ),
        "domains": _array_comparison(
            legacy.domains, optimized.domains, tolerance=tolerance
        ),
    }
    if schema["prior_names"]:
        for name in sorted(legacy.priors):
            arrays[f"prior__{name}"] = _array_comparison(
                legacy.priors[name], optimized.priors[name], tolerance=tolerance
            )
    if schema["target_names"]:
        for name in sorted(legacy.targets):
            arrays[f"target__{name}"] = _array_comparison(
                legacy.targets[name], optimized.targets[name], tolerance=tolerance
            )

    row_metadata = {
        key: _array_comparison(
            _aligned_metadata(legacy, key),
            _aligned_metadata(optimized, key),
            tolerance=tolerance,
        )
        for key in ("action_group_ids", "row_tls", "row_times", "candidate_states")
    }
    semantic_metadata_equal = _semantic_metadata(legacy) == _semantic_metadata(
        optimized
    )

    legacy_snapshot = dict(legacy.metadata.get("state_snapshot_protocol", {}))
    optimized_snapshot = dict(optimized.metadata.get("state_snapshot_protocol", {}))
    reader = dict(optimized.metadata.get("halted_cost_reader") or {})
    optimized_path_active = bool(
        optimized_snapshot.get("reload_mode") == OPTIMIZED_RELOAD_MODE
        and optimized_snapshot.get("branch_evaluation") == OPTIMIZED_BRANCH_MODE
        and optimized_snapshot.get("halted_cost_readout")
        == "libsumo_lane_subscription_v1"
        and reader.get("protocol") == OPTIMIZED_HALTED_READER_PROTOCOL
        and int(reader.get("aggregate_reads", 0)) > 0
    )
    distinct_implementation = legacy_snapshot != optimized_snapshot

    legacy_elapsed = float(legacy_elapsed_seconds)
    optimized_elapsed = float(optimized_elapsed_seconds)
    speedup = legacy_elapsed / optimized_elapsed
    runtime = {
        "legacy_elapsed_seconds": legacy_elapsed,
        "optimized_elapsed_seconds": optimized_elapsed,
        "speedup": float(speedup),
        "minimum_speedup": float(minimum_speedup),
        "passed": bool(speedup + 1e-12 >= float(minimum_speedup)),
    }
    gates = {
        "schema": all(schema.values()),
        "all_model_arrays": all(value["passed"] for value in arrays.values()),
        "row_metadata": all(value["passed"] for value in row_metadata.values()),
        "semantic_metadata": semantic_metadata_equal,
        "distinct_implementation": distinct_implementation,
        "optimized_path_active": optimized_path_active,
        "runtime_speedup": runtime["passed"],
    }
    gates["passed"] = all(gates.values())
    return {
        "status": "PASS" if gates["passed"] else "FAIL",
        "decision": (
            "authorize_homogeneous_optimized_cache_rebuild"
            if gates["passed"]
            else "reject_optimized_cache_implementation"
        ),
        "gates": gates,
        "schema": schema,
        "row_counts": {
            "legacy": int(legacy.size),
            "optimized": int(optimized.size),
        },
        "array_comparisons": arrays,
        "row_metadata_comparisons": row_metadata,
        "snapshot_protocols": {
            "legacy": legacy_snapshot,
            "optimized": optimized_snapshot,
        },
        "optimized_halted_reader": reader,
        "runtime": runtime,
    }


def run_audit(
    *,
    legacy_cache: Path,
    optimized_cache: Path,
    legacy_summary: Path,
    optimized_summary: Path,
    tolerance: float,
    minimum_speedup: float,
) -> dict[str, Any]:
    legacy = load_mechanism_dataset(legacy_cache)
    optimized = load_mechanism_dataset(optimized_cache)
    comparison = compare_counterfactual_implementations(
        legacy,
        optimized,
        legacy_elapsed_seconds=_elapsed_seconds(_read_json(legacy_summary)),
        optimized_elapsed_seconds=_elapsed_seconds(_read_json(optimized_summary)),
        tolerance=tolerance,
        minimum_speedup=minimum_speedup,
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **comparison,
        "inputs": {
            "legacy_cache": str(Path(legacy_cache).resolve()),
            "optimized_cache": str(Path(optimized_cache).resolve()),
            "legacy_summary": str(Path(legacy_summary).resolve()),
            "optimized_summary": str(Path(optimized_summary).resolve()),
        },
        "claim_boundary": (
            "Implementation-equivalence and runtime evidence only; this audit "
            "contains no policy-efficacy evidence."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument("--optimized-cache", type=Path, required=True)
    parser.add_argument("--legacy-summary", type=Path, required=True)
    parser.add_argument("--optimized-summary", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--minimum-speedup", type=float, default=1.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite equivalence audit: {args.out}")
    result = run_audit(
        legacy_cache=args.legacy_cache,
        optimized_cache=args.optimized_cache,
        legacy_summary=args.legacy_summary,
        optimized_summary=args.optimized_summary,
        tolerance=args.absolute_tolerance,
        minimum_speedup=args.minimum_speedup,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "speedup": result["runtime"]["speedup"],
                "rows": result["row_counts"]["optimized"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
