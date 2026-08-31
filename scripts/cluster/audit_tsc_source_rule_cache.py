#!/usr/bin/env python3
"""Audit partitioned content-addressed source pressure-rule rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_source_rule_cache import _policy_grid
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    SOURCE_RULE_CACHE_VERSION,
    _source_rule_cache_path_v3,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import STRICT_SAFETY_MONITORING_V3
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import SUMO_EXECUTION_PROTOCOL


DEFAULT_SPEC = PROJECT_ROOT / "cf_h2o/config/traffic_signal_tsc_v16_benchmark_aware_development.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_cache_sha256(file_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, file_digest in sorted(file_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest()


def audit_source_rule_cache(
    *,
    results_root: Path,
    spec_path: Path,
    expected_source_sha256: str | None = None,
    require_cache_files: bool = False,
    cache_root_override: Path | None = None,
) -> dict[str, Any]:
    spec = _load_json(spec_path)
    development = spec["development"]
    counterfactual = spec["counterfactual"]
    expected_scenarios = {
        str(scenario)
        for scenarios in spec["cache_assignments"].values()
        for scenario in scenarios
    }
    expected_seeds = tuple(int(seed) for seed in development["source_policy_seeds"])
    expected_specs = tuple(_policy_grid(development["source_rule_grid"]))
    expected_policy_keys = {item.key for item in expected_specs}
    result_files = sorted(results_root.glob("**/source_rules.json"))
    if not result_files:
        raise FileNotFoundError(f"no source_rules.json files found below {results_root}")

    errors: list[str] = []
    scenario_owner: dict[str, str] = {}
    cache_files: list[Path] = []
    source_hashes: set[str] = set()
    cache_roots: set[str] = set()
    total_rows = 0

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    for result_file in result_files:
        payload = _load_json(result_file)
        setting = payload.get("setting", {})
        runtime = payload.get("runtime", {})
        location = str(result_file)
        selected = tuple(str(value) for value in setting.get("scenarios", ()))
        require(
            setting.get("cache_version") == SOURCE_RULE_CACHE_VERSION,
            f"{location}: cache version mismatch",
        )
        require(
            setting.get("sumo_execution_protocol") == SUMO_EXECUTION_PROTOCOL,
            f"{location}: SUMO execution protocol mismatch",
        )
        require(
            setting.get("strict_safety_monitoring")
            == STRICT_SAFETY_MONITORING_V3,
            f"{location}: strict safety monitoring mismatch",
        )
        require(
            setting.get("libsumo_version") == spec["expected_sumo_version"],
            f"{location}: SUMO version mismatch",
        )
        require(
            tuple(int(value) for value in setting.get("seeds", ())) == expected_seeds,
            f"{location}: source policy seeds mismatch",
        )
        require(
            setting.get("source_rule_grid") == development["source_rule_grid"],
            f"{location}: source rule grid mismatch",
        )
        require(
            {
                json.dumps(value, sort_keys=True)
                for value in setting.get("policy_specs", ())
            }
            == {json.dumps(item.to_dict(), sort_keys=True) for item in expected_specs},
            f"{location}: pressure policy specification mismatch",
        )
        require(
            float(setting.get("duration_sec", -1.0))
            == float(counterfactual["source_duration_sec"]),
            f"{location}: source duration mismatch",
        )
        require(
            int(setting.get("control_interval_sec", -1))
            == int(counterfactual["control_interval_sec"]),
            f"{location}: control interval mismatch",
        )
        for scenario in selected:
            if scenario in scenario_owner:
                errors.append(
                    f"{location}:{scenario}: duplicate scenario also in "
                    f"{scenario_owner[scenario]}"
                )
            scenario_owner[scenario] = location
        costs = payload.get("scenario_policy_costs", {})
        require(set(costs) == set(selected), f"{location}: scenario cost keys mismatch")
        for scenario, values in costs.items():
            require(
                set(values) == expected_policy_keys,
                f"{location}:{scenario}: policy cost keys mismatch",
            )
            require(
                all(math.isfinite(float(value)) for value in values.values()),
                f"{location}:{scenario}: non-finite policy cost",
            )
        expected_rows = len(selected) * len(expected_specs) * len(expected_seeds)
        rows = int(payload.get("row_count", -1))
        require(rows == expected_rows, f"{location}: rollout row count mismatch")
        total_rows += max(rows, 0)
        safety_audit = dict(payload.get("safety_audit", {}))
        require(
            safety_audit.get("strict_safety_monitoring")
            == STRICT_SAFETY_MONITORING_V3,
            f"{location}: safety audit protocol mismatch",
        )
        require(
            int(safety_audit.get("rollout_count", -1)) == expected_rows,
            f"{location}: safety audit rollout count mismatch",
        )
        safety_totals = dict(safety_audit.get("totals", {}))
        require(
            safety_audit.get("passed") is True
            and set(safety_totals)
            == {
                "collision_events",
                "collision_incidents",
                "starting_teleports",
                "ending_teleports",
            }
            and int(safety_totals["collision_events"]) >= 0
            and 0
            <= int(safety_totals["collision_incidents"])
            <= int(safety_totals["collision_events"])
            and int(safety_totals["starting_teleports"]) == 0
            and int(safety_totals["ending_teleports"]) == 0,
            f"{location}: source rule safety audit failed",
        )
        reported_partition_files = [
            Path(value) for value in payload.get("cache_files", ())
        ]
        partition_files = [
            Path(cache_root_override) / path.name
            if cache_root_override is not None
            else path
            for path in reported_partition_files
        ]
        require(
            len(partition_files) == expected_rows,
            f"{location}: cache file count mismatch",
        )
        cache_files.extend(partition_files)
        cache_roots.add(str(setting.get("cache_root", "")))
        source_hash = str(runtime.get("source_tree_sha256", ""))
        source_hashes.add(source_hash)
        if expected_source_sha256 is not None:
            require(
                source_hash == expected_source_sha256,
                f"{location}: source tree hash mismatch",
            )

    require(
        set(scenario_owner) == expected_scenarios,
        "scenario partition mismatch: "
        f"missing={sorted(expected_scenarios - set(scenario_owner))}, "
        f"extra={sorted(set(scenario_owner) - expected_scenarios)}",
    )
    require(len(cache_roots) == 1 and "" not in cache_roots, "cache roots disagree")
    require(len(source_hashes) == 1 and "" not in source_hashes, "source hashes disagree")
    require(len(cache_files) == len(set(cache_files)), "cache file identities are not unique")

    missing: list[str] = []
    unexpected: list[str] = []
    cache_file_sha256: dict[str, str] = {}
    if require_cache_files:
        for path in cache_files:
            if not path.is_file():
                missing.append(str(path))
                continue
            cached = _load_json(path)
            cache_file_sha256[path.name] = _file_sha256(path)
            identity = cached.get("identity", {})
            row = cached.get("row", {})
            require(
                identity.get("version") == SOURCE_RULE_CACHE_VERSION,
                f"{path}: identity version mismatch",
            )
            require(
                identity.get("sumo_version") == spec["expected_sumo_version"],
                f"{path}: identity SUMO version mismatch",
            )
            require(
                identity.get("sumo_execution_protocol")
                == SUMO_EXECUTION_PROTOCOL,
                f"{path}: identity SUMO execution protocol mismatch",
            )
            require(
                identity.get("strict_safety_monitoring")
                == STRICT_SAFETY_MONITORING_V3,
                f"{path}: identity safety monitoring protocol mismatch",
            )
            require(
                str(identity.get("scenario", "")) in expected_scenarios,
                f"{path}: unknown scenario identity",
            )
            require(
                int(identity.get("seed", -1)) in expected_seeds,
                f"{path}: unknown seed identity",
            )
            require(
                _source_rule_cache_path_v3(path.parent, identity).resolve()
                == path.resolve(),
                f"{path}: content address mismatch",
            )
            require(
                row.get("metrics", {}).get("ok", False),
                f"{path}: cached rollout failed",
            )
            require(
                row.get("metrics", {}).get("sumo_execution_protocol")
                == SUMO_EXECUTION_PROTOCOL,
                f"{path}: cached rollout SUMO execution protocol mismatch",
            )
            require(
                row.get("metrics", {}).get("strict_safety_monitoring")
                == STRICT_SAFETY_MONITORING_V3,
                f"{path}: cached rollout safety monitoring protocol mismatch",
            )
            cached_metrics = dict(row.get("metrics", {}))
            collision_events = int(cached_metrics.get("collision_events", -1))
            collision_incidents = int(
                cached_metrics.get("collision_incidents", -1)
            )
            require(
                collision_events >= 0
                and 0 <= collision_incidents <= collision_events,
                f"{path}: cached rollout has invalid collision audit",
            )
            for metric in ("starting_teleports", "ending_teleports"):
                require(
                    int(cached_metrics.get(metric, -1)) == 0,
                    f"{path}: cached rollout has invalid {metric}",
                )
        require(not missing, f"missing {len(missing)} cache files")
        parents = {path.parent.resolve() for path in cache_files}
        require(len(parents) == 1, "cache files do not share one root")
        if len(parents) == 1:
            actual = set(next(iter(parents)).glob("source_rule_*.json"))
            expected = {path.resolve() for path in cache_files}
            unexpected = sorted(str(path) for path in actual - expected)
            require(not unexpected, f"cache root contains {len(unexpected)} unexpected files")

    summary = {
        "audit": "traffic_signal_source_rule_cache",
        "passed": not errors,
        "results_root": str(results_root.resolve()),
        "result_files": [str(path.resolve()) for path in result_files],
        "scenario_count": len(scenario_owner),
        "policy_count": len(expected_specs),
        "source_seeds": list(expected_seeds),
        "cache_file_count": len(cache_files),
        "cache_root": next(iter(cache_roots), ""),
        "physical_cache_root": (
            str(Path(cache_root_override).resolve())
            if cache_root_override is not None
            else next(iter(cache_roots), "")
        ),
        "cache_file_sha256": cache_file_sha256,
        "cache_aggregate_sha256": (
            _aggregate_cache_sha256(cache_file_sha256)
            if cache_file_sha256
            else None
        ),
        "source_tree_sha256": next(iter(source_hashes), ""),
        "total_rows": total_rows,
        "missing_cache_files": missing,
        "unexpected_cache_files": unexpected,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("source rule cache audit failed:\n- " + "\n- ".join(errors))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--require-cache-files", action="store_true")
    parser.add_argument("--cache-root-override", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = audit_source_rule_cache(
        results_root=args.results_root,
        spec_path=args.spec,
        expected_source_sha256=args.expected_source_sha256,
        require_cache_files=args.require_cache_files,
        cache_root_override=args.cache_root_override,
    )
    if args.out is not None:
        _atomic_write_json(args.out, summary)
    print(
        "source rule cache audit passed: "
        f"{summary['scenario_count']} scenarios, "
        f"{summary['policy_count']} policies, "
        f"{summary['cache_file_count']} files"
    )


if __name__ == "__main__":
    main()
