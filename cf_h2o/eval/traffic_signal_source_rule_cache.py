"""Precompute content-addressed source pressure-rule rollouts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import SUMO_EXECUTION_PROTOCOL
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    SOURCE_RULE_CACHE_VERSION,
    _source_rule_cache_identity_v3,
    _source_rule_cache_path_v3,
    collect_source_rule_costs_v3,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import STRICT_SAFETY_MONITORING_V3
from cf_h2o.sumo_runtime import libsumo_version
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.generalized_pressure import (
    MAX_PRESSURE_SPEC,
    PHASE_PRESSURE_SPEC,
    SPILLBACK_PRESSURE_SPEC,
    generalized_pressure_grid,
)


def _policy_grid(name: str):
    return {
        "generalized": generalized_pressure_grid(),
        "legacy": (MAX_PRESSURE_SPEC, PHASE_PRESSURE_SPEC, SPILLBACK_PRESSURE_SPEC),
        "fixed_max": (MAX_PRESSURE_SPEC,),
        "fixed_phase": (PHASE_PRESSURE_SPEC,),
    }[str(name)]


def precompute_source_rule_cache(
    *,
    manifest_path: Path,
    scenarios: Sequence[str] | None,
    seeds: Sequence[int],
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    workers: int,
    source_rule_grid: str,
    cache_root: Path,
    tripinfo_root: Path,
    expected_sumo_version: str | None = None,
) -> dict[str, Any]:
    manifest = load_traffic_signal_manifest(manifest_path)
    selected = tuple(str(name) for name in (scenarios or tuple(manifest.sumocfgs)))
    unknown = set(selected) - set(manifest.sumocfgs)
    if unknown:
        raise ValueError(f"scenarios are absent from manifest: {sorted(unknown)}")
    source_seeds = tuple(int(seed) for seed in seeds)
    specs = tuple(_policy_grid(source_rule_grid))
    actual_sumo_version = libsumo_version()
    started = time.perf_counter()
    costs, rows = collect_source_rule_costs_v3(
        env_root=Path("."),
        scenarios=selected,
        seeds=source_seeds,
        duration_sec=duration_sec,
        control_interval_sec=control_interval_sec,
        warmup_sec=warmup_sec,
        workers=workers,
        policy_specs=specs,
        scenario_sumocfgs={name: manifest.sumocfgs[name] for name in selected},
        source_rule_cache_root=cache_root,
        expected_sumo_version=expected_sumo_version,
        tripinfo_root=tripinfo_root,
    )
    cache_files = []
    for scenario in selected:
        for spec in specs:
            for seed in source_seeds:
                identity = _source_rule_cache_identity_v3(
                    sumocfg=manifest.sumocfgs[scenario],
                    scenario=scenario,
                    policy_spec=spec,
                    seed=seed,
                    duration_sec=duration_sec,
                    control_interval_sec=control_interval_sec,
                    warmup_sec=warmup_sec,
                    sumo_version=actual_sumo_version,
                )
                cache_files.append(str(_source_rule_cache_path_v3(cache_root, identity)))
    missing = [path for path in cache_files if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"source rule precompute is missing {len(missing)} cache files")
    safety_totals = {
        name: int(sum(int(row["metrics"][name]) for row in rows))
        for name in (
            "collision_events",
            "collision_incidents",
            "starting_teleports",
            "ending_teleports",
        )
    }
    safety_audit = {
        "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
        "rollout_count": len(rows),
        "totals": safety_totals,
        "collision_interpretation": (
            "raw_benchmark_events_audited_not_blanket_failure"
        ),
        "passed": safety_totals["starting_teleports"] == 0
        and safety_totals["ending_teleports"] == 0,
    }
    if not safety_audit["passed"]:
        raise RuntimeError(f"source rule safety audit failed: {safety_audit}")
    return {
        "experiment": "traffic_signal_source_rule_cache_precompute",
        "runtime": _runtime_metadata(),
        "setting": {
            "manifest": manifest.to_dict(),
            "scenarios": list(selected),
            "seeds": list(source_seeds),
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "workers": int(workers),
            "source_rule_grid": str(source_rule_grid),
            "policy_specs": [spec.to_dict() for spec in specs],
            "cache_root": str(cache_root),
            "cache_version": SOURCE_RULE_CACHE_VERSION,
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
            "libsumo_version": actual_sumo_version,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "scenario_policy_costs": costs,
        "row_count": len(rows),
        "safety_audit": safety_audit,
        "cache_files": cache_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cf_h2o/config/traffic_signal_cross_city_v1.json"),
    )
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1709, 2027])
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument(
        "--source-rule-grid",
        choices=("generalized", "legacy", "fixed_max", "fixed_phase"),
        default="generalized",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--tripinfo-root", type=Path, required=True)
    parser.add_argument("--expected-sumo-version")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = precompute_source_rule_cache(
        manifest_path=args.manifest,
        scenarios=args.scenarios,
        seeds=args.seeds,
        duration_sec=args.duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        workers=args.workers,
        source_rule_grid=args.source_rule_grid,
        cache_root=args.cache_root,
        tripinfo_root=args.tripinfo_root,
        expected_sumo_version=args.expected_sumo_version,
    )
    atomic_write_json(args.out, result)
    print(
        f"cached {result['row_count']} source-rule rollouts at "
        f"{result['setting']['cache_root']}"
    )


if __name__ == "__main__":
    main()
