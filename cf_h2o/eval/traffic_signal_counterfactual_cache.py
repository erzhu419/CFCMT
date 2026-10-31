"""Precompute and audit matched SUMO counterfactual caches for TSC experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    counterfactual_cost_contract_v5,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import SUMO_EXECUTION_PROTOCOL
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    COUNTERFACTUAL_CACHE_VERSION,
    MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION,
    _counterfactual_cache_path,
    collect_counterfactual_bank_v3,
)
from cf_h2o.sumo_runtime import libsumo_version
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    resolved = Path(root).resolve()
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"counterfactual input root contains no files: {resolved}")
    for path in files:
        digest.update(path.relative_to(resolved).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validated_authorization_provenance(
    audit_path: Path, *, expected_sha256: str, required_decision: str
) -> dict[str, Any]:
    resolved = Path(audit_path).resolve()
    audit_sha256 = _sha256(resolved)
    if audit_sha256 != str(expected_sha256):
        raise ValueError("counterfactual authorization audit SHA-256 mismatch")
    audit = json.loads(resolved.read_text(encoding="utf-8"))
    authorization_gate = audit.get("gate", audit.get("integrity_gate", {}))
    if (
        audit.get("status") != "PASS"
        or not bool(authorization_gate.get("passed", False))
        or audit.get("decision") != str(required_decision)
    ):
        raise ValueError("counterfactual authorization audit does not permit collection")
    return {
        "path": str(resolved),
        "sha256": audit_sha256,
        "status": audit["status"],
        "decision": audit["decision"],
    }


def precompute_counterfactual_cache(
    *,
    manifest_path: Path,
    scenarios: Sequence[str] | None,
    seeds: Sequence[int],
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    max_focal_tls: int,
    counterfactual_horizon_intervals: int,
    collection_shards: int,
    workers: int,
    cache_root: Path,
    behavior_policy: str,
    min_tls_coverage: float,
    counterfactual_cost_mode: str = "system_vehicle_load",
    expected_sumo_version: str | None = None,
    expected_input_root: Path | None = None,
    expected_input_manifest_sha256: str | None = None,
    expected_input_tree_sha256: str | None = None,
    authorization_audit: Path | None = None,
    expected_authorization_sha256: str | None = None,
    required_authorization_decision: str | None = None,
    rollout_prefix_horizons_sec: Sequence[int] | None = None,
    allowed_zero_retained_scenarios: Sequence[str] = (),
) -> dict[str, Any]:
    authorization_provenance: dict[str, Any] | None = None
    authorization_values = (
        authorization_audit,
        expected_authorization_sha256,
        required_authorization_decision,
    )
    if any(value is not None for value in authorization_values):
        if not all(value is not None for value in authorization_values):
            raise ValueError(
                "authorization audit, SHA-256, and required decision must be "
                "provided together"
            )
        authorization_provenance = _validated_authorization_provenance(
            Path(authorization_audit),
            expected_sha256=str(expected_authorization_sha256),
            required_decision=str(required_authorization_decision),
        )
    input_provenance: dict[str, Any] | None = None
    expected_values = (
        expected_input_root,
        expected_input_manifest_sha256,
        expected_input_tree_sha256,
    )
    if any(value is not None for value in expected_values):
        if not all(value is not None for value in expected_values):
            raise ValueError(
                "expected input root, manifest hash, and tree hash must be "
                "provided together"
            )
        input_root = Path(expected_input_root).resolve()
        manifest_digest = _sha256(input_root / "conversion_manifest.json")
        tree_digest = _tree_sha256(input_root)
        if manifest_digest != str(expected_input_manifest_sha256):
            raise ValueError(
                "counterfactual input manifest SHA-256 mismatch: "
                f"{manifest_digest} != {expected_input_manifest_sha256}"
            )
        if tree_digest != str(expected_input_tree_sha256):
            raise ValueError(
                "counterfactual input tree SHA-256 mismatch: "
                f"{tree_digest} != {expected_input_tree_sha256}"
            )
        input_provenance = {
            "root": str(input_root),
            "manifest_sha256": manifest_digest,
            "tree_sha256": tree_digest,
        }
    manifest = load_traffic_signal_manifest(manifest_path)
    selected = tuple(str(name) for name in (scenarios or tuple(manifest.sumocfgs)))
    unknown = set(selected) - set(manifest.sumocfgs)
    if unknown:
        raise ValueError(f"scenarios are absent from manifest: {sorted(unknown)}")
    if not selected:
        raise ValueError("at least one counterfactual scenario is required")
    zero_retained_allowed = {
        str(value) for value in allowed_zero_retained_scenarios
    }
    if zero_retained_allowed - set(selected):
        raise ValueError(
            "zero-retained exclusions are absent from the selected scenarios: "
            f"{sorted(zero_retained_allowed - set(selected))}"
        )
    source_seeds = tuple(int(seed) for seed in seeds)
    if not source_seeds:
        raise ValueError("at least one counterfactual source seed is required")
    if not 0.0 <= float(min_tls_coverage) <= 1.0:
        raise ValueError("min_tls_coverage must be in [0, 1]")
    actual_sumo_version = libsumo_version()
    if expected_sumo_version and actual_sumo_version != str(expected_sumo_version):
        raise RuntimeError(
            "counterfactual cache SUMO version mismatch: "
            f"expected {expected_sumo_version!r}, found {actual_sumo_version!r}"
        )

    resolved = {name: manifest.sumocfgs[name] for name in selected}
    if input_provenance is not None:
        input_root = Path(input_provenance["root"])
        escaped = [
            name
            for name, sumocfg in resolved.items()
            if input_root not in Path(sumocfg).resolve().parents
        ]
        if escaped:
            raise ValueError(
                "counterfactual manifest scenarios escape the frozen input "
                f"root: {escaped}"
            )
    started = time.perf_counter()
    bank = collect_counterfactual_bank_v3(
        env_root=Path("."),
        scenarios=selected,
        duration_sec=duration_sec,
        control_interval_sec=control_interval_sec,
        warmup_sec=warmup_sec,
        seeds=source_seeds,
        max_focal_tls=max_focal_tls,
        counterfactual_horizon_intervals=counterfactual_horizon_intervals,
        workers=workers,
        cache_root=cache_root,
        behavior_policy=behavior_policy,
        counterfactual_cost_mode=counterfactual_cost_mode,
        scenario_sumocfgs=resolved,
        collection_shards=collection_shards,
        rollout_prefix_horizons_sec=rollout_prefix_horizons_sec,
    )
    diagnostics = {}
    for name, dataset in bank.items():
        coverage = float(dataset.metadata.get("tls_coverage_fraction", 0.0))
        replay = dict(dataset.metadata.get("counterfactual_replay_audit", {}))
        safety = dict(dataset.metadata.get("counterfactual_safety_audit", {}))
        behavior_safety = dict(dataset.metadata.get("behavior_safety_audit", {}))
        zero_retained_complete = bool(
            name in zero_retained_allowed
            and dataset.size == 0
            and int(safety.get("candidate_groups", 0)) > 0
            and int(safety.get("retained_groups", -1)) == 0
            and int(safety.get("censored_groups", -1))
            == int(safety.get("candidate_groups", -2))
            and int(behavior_safety.get("collision_event_steps", 0)) > 0
            and bool(safety.get("no_teleport_passed", False))
            and bool(behavior_safety.get("no_teleport_passed", False))
            and bool(replay.get("passed", False))
        )
        if coverage + 1e-12 < float(min_tls_coverage):
            if not zero_retained_complete:
                raise RuntimeError(
                    f"counterfactual TLS coverage below floor for {name}: "
                    f"{coverage:.6f} < {float(min_tls_coverage):.6f}"
                )
        if not replay.get("passed", False):
            raise RuntimeError(f"counterfactual replay audit failed for {name}: {replay}")
        diagnostics[name] = {
            "city_group": manifest.city_groups[name],
            "rows": int(dataset.size),
            "groups": len(set(dataset.metadata["action_group_ids"])),
            "controllable_tls_count": int(dataset.metadata.get("controllable_tls_count", 0)),
            "covered_tls_count": int(dataset.metadata.get("covered_tls_count", 0)),
            "tls_coverage_fraction": coverage,
            "collection_admission_status": (
                "complete_but_zero_retained_groups"
                if zero_retained_complete
                else "complete_with_retained_groups"
            ),
            "counterfactual_branches": int(dataset.metadata["counterfactual_branches"]),
            "counterfactual_estimand": dataset.metadata.get(
                "counterfactual_estimand"
            ),
            "counterfactual_cost_mode": dataset.metadata.get(
                "counterfactual_cost_mode", "system_vehicle_load"
            ),
            "mechanism_output_names": dataset.metadata.get(
                "mechanism_output_names"
            ),
            "local_mechanism_horizon_sec": int(
                dataset.metadata.get("local_mechanism_horizon_sec", -1)
            ),
            "rollout_value_horizon_sec": int(
                dataset.metadata.get("rollout_value_horizon_sec", -1)
            ),
            "rollout_prefix_horizons_sec": list(
                rollout_prefix_horizons_sec or ()
            ),
            "mechanism_estimands": dataset.metadata.get("mechanism_estimands"),
            "state_restores": int(dataset.metadata["state_restores"]),
            "state_snapshot_protocol": dataset.metadata.get("state_snapshot_protocol"),
            "sumo_execution_protocol": dataset.metadata.get("sumo_execution_protocol"),
            "strict_safety_monitoring": dataset.metadata.get(
                "strict_safety_monitoring"
            ),
            "behavior_safety_audit": dataset.metadata.get(
                "behavior_safety_audit"
            ),
            "counterfactual_safety_audit": dataset.metadata.get(
                "counterfactual_safety_audit"
            ),
            "behavior_trace_sha256_by_seed": dataset.metadata.get(
                "behavior_trace_sha256_by_seed",
                {
                    str(dataset.metadata.get("seed")): dataset.metadata.get(
                        "behavior_trace_sha256"
                    )
                },
            ),
            "behavior_interval_count_by_seed": dataset.metadata.get(
                "behavior_interval_count_by_seed",
                {
                    str(dataset.metadata.get("seed")): int(
                        dataset.metadata.get("behavior_interval_count", 0)
                    )
                },
            ),
            "eligible_collection_opportunities_by_seed": dataset.metadata.get(
                "eligible_collection_opportunities_by_seed",
                {
                    str(dataset.metadata.get("seed")): int(
                        dataset.metadata.get("eligible_collection_opportunities", 0)
                    )
                },
            ),
            "focal_selection_protocol": dataset.metadata.get(
                "focal_selection_protocol"
            ),
            "focal_selection_min_count_by_seed": dataset.metadata.get(
                "focal_selection_min_count_by_seed",
                {
                    str(dataset.metadata.get("seed")): int(
                        dataset.metadata.get("focal_selection_min_count", 0)
                    )
                },
            ),
            "focal_selection_max_count_by_seed": dataset.metadata.get(
                "focal_selection_max_count_by_seed",
                {
                    str(dataset.metadata.get("seed")): int(
                        dataset.metadata.get("focal_selection_max_count", 0)
                    )
                },
            ),
            "captured_snapshot_count": int(
                dataset.metadata.get("captured_snapshot_count", 0)
            ),
            "counterfactual_replay_audit": replay,
            "cache_files": [
                str(
                    _counterfactual_cache_path(
                        cache_root=cache_root,
                        sumocfg=resolved[name],
                        scenario=name,
                        duration_sec=duration_sec,
                        control_interval_sec=control_interval_sec,
                        warmup_sec=warmup_sec,
                        seed=seed,
                        max_focal_tls=max_focal_tls,
                        counterfactual_horizon_intervals=counterfactual_horizon_intervals,
                        behavior_policy=behavior_policy,
                        counterfactual_cost_mode=counterfactual_cost_mode,
                        collection_shard_index=shard_index,
                        collection_shard_count=collection_shards,
                        rollout_prefix_horizons_sec=rollout_prefix_horizons_sec,
                    )
                )
                for seed in source_seeds
                for shard_index in range(int(collection_shards))
            ],
        }

    return {
        "experiment": "traffic_signal_counterfactual_cache_precompute",
        "runtime": _runtime_metadata(),
        "setting": {
            "manifest": manifest.to_dict(),
            "scenarios": list(selected),
            "seeds": list(source_seeds),
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "max_focal_tls": int(max_focal_tls),
            "counterfactual_horizon_intervals": int(
                counterfactual_horizon_intervals
            ),
            "collection_shards": int(collection_shards),
            "workers": int(workers),
            "cache_root": str(cache_root),
            "cache_version": (
                MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION
                if rollout_prefix_horizons_sec
                else COUNTERFACTUAL_CACHE_VERSION
            ),
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "behavior_policy": str(behavior_policy),
            "counterfactual_cost_mode": str(counterfactual_cost_mode),
            "counterfactual_cost_contract": counterfactual_cost_contract_v5(
                counterfactual_cost_mode
            ),
            "rollout_prefix_horizons_sec": [
                int(value) for value in (rollout_prefix_horizons_sec or ())
            ],
            "min_tls_coverage": float(min_tls_coverage),
            "allowed_zero_retained_scenarios": sorted(zero_retained_allowed),
            "libsumo_version": actual_sumo_version,
            "input_provenance": input_provenance,
            "authorization_provenance": authorization_provenance,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cf_h2o/config/traffic_signal_cross_city_v1.json"),
    )
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 3037, 4047])
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--max-focal-tls", type=int, default=4)
    parser.add_argument("--counterfactual-horizon-intervals", type=int, default=6)
    parser.add_argument("--collection-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cf_h2o/results/traffic_signal_resco_cfcmt_v3_cache"),
    )
    parser.add_argument(
        "--behavior-policy",
        choices=("phase_pressure", "exploratory_mixture"),
        default="phase_pressure",
    )
    parser.add_argument(
        "--counterfactual-cost-mode",
        choices=("system_vehicle_load", "halted_queue"),
        default="system_vehicle_load",
    )
    parser.add_argument("--min-tls-coverage", type=float, default=1.0)
    parser.add_argument("--allow-zero-retained-scenarios", nargs="*", default=())
    parser.add_argument(
        "--rollout-prefix-horizons-sec",
        nargs="+",
        type=int,
        help="save prefix-mean rollout costs at these horizons",
    )
    parser.add_argument("--expected-sumo-version")
    parser.add_argument("--expected-input-root", type=Path)
    parser.add_argument("--expected-input-manifest-sha256")
    parser.add_argument("--expected-input-tree-sha256")
    parser.add_argument("--authorization-audit", type=Path)
    parser.add_argument("--expected-authorization-sha256")
    parser.add_argument("--required-authorization-decision")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cf_h2o/results/traffic_signal_counterfactual_cache.json"),
    )
    args = parser.parse_args()
    result = precompute_counterfactual_cache(
        manifest_path=args.manifest,
        scenarios=args.scenarios,
        seeds=args.seeds,
        duration_sec=args.duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        max_focal_tls=args.max_focal_tls,
        counterfactual_horizon_intervals=args.counterfactual_horizon_intervals,
        collection_shards=args.collection_shards,
        workers=args.workers,
        cache_root=args.cache_root,
        behavior_policy=args.behavior_policy,
        min_tls_coverage=args.min_tls_coverage,
        counterfactual_cost_mode=args.counterfactual_cost_mode,
        expected_sumo_version=args.expected_sumo_version,
        expected_input_root=args.expected_input_root,
        expected_input_manifest_sha256=args.expected_input_manifest_sha256,
        expected_input_tree_sha256=args.expected_input_tree_sha256,
        authorization_audit=args.authorization_audit,
        expected_authorization_sha256=args.expected_authorization_sha256,
        required_authorization_decision=args.required_authorization_decision,
        rollout_prefix_horizons_sec=args.rollout_prefix_horizons_sec,
        allowed_zero_retained_scenarios=args.allow_zero_retained_scenarios,
    )
    atomic_write_json(args.out, result)
    print(
        f"cached {len(result['diagnostics'])} scenarios in "
        f"{result['elapsed_seconds']:.2f}s at {args.cache_root}"
    )


if __name__ == "__main__":
    main()
