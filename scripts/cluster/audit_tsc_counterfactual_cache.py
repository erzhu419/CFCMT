#!/usr/bin/env python3
"""Audit a partitioned TSC counterfactual cache before model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    COUNTERFACTUAL_CACHE_VERSION,
    _validate_counterfactual_cache_dataset,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    COUNTERFACTUAL_SAFETY_PROTOCOL_V3,
    ONE_STEP_OUTPUT_NAMES_V4,
    OUTPUT_NAMES_V4,
    POLICY_CONSISTENT_ESTIMAND_PROTOCOL_V4,
    STATE_SNAPSHOT_PROTOCOL_V3,
    STRICT_SAFETY_MONITORING_V3,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import SUMO_EXECUTION_PROTOCOL
from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset


DEFAULT_SPEC = PROJECT_ROOT / "cf_h2o/config/traffic_signal_tsc_v16_benchmark_aware_development.json"
EXPECTED_SNAPSHOT_PROTOCOL = STATE_SNAPSHOT_PROTOCOL_V3
EXPECTED_FOCAL_SELECTION_PROTOCOL = "least_covered_feasible_round_robin_v1"
EXPECTED_DETERMINISM_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
}


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_cache_sha256(file_hashes: dict[str, str]) -> str:
    """Use the repository-wide name-and-binary-digest cache hash protocol."""

    digest = hashlib.sha256()
    for name, file_digest in sorted(file_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest()


def audit_cache(
    *,
    results_root: Path,
    spec_path: Path,
    expected_source_sha256: str | None = None,
    require_cache_files: bool = False,
    cache_root_override: Path | None = None,
    min_groups_per_scenario: int = 1,
) -> dict[str, Any]:
    if int(min_groups_per_scenario) < 1:
        raise ValueError("min_groups_per_scenario must be positive")
    spec = _load_json(spec_path)
    manifest_path = (PROJECT_ROOT / spec["manifest"]).resolve()
    manifest = _load_json(manifest_path)
    expected_entries = {
        str(item["scenario"]): dict(item) for item in manifest["scenarios"]
    }
    counterfactual = spec["counterfactual"]
    expected_seeds = tuple(int(seed) for seed in counterfactual["source_seeds"])
    expected_seed_keys = {str(seed) for seed in expected_seeds}
    expected_shards = int(counterfactual["collection_shards"])
    expected_cache_count = len(expected_seeds) * expected_shards
    expected_opportunities = int(
        (float(counterfactual["source_duration_sec"]) - float(counterfactual["warmup_sec"]))
        // int(counterfactual["control_interval_sec"])
    )
    expected_intervals = int(
        float(counterfactual["source_duration_sec"])
        // int(counterfactual["control_interval_sec"])
    )

    result_files = sorted(results_root.glob("**/cache.json"))
    if not result_files:
        raise FileNotFoundError(f"no cache.json files found below {results_root}")

    errors: list[str] = []
    scenario_owner: dict[str, str] = {}
    expected_cache_files: list[Path] = []
    source_hashes: set[str] = set()
    cache_roots: set[str] = set()
    total_rows = 0
    total_groups = 0
    total_branches = 0
    total_replay_checks = 0
    total_candidate_groups = 0
    total_censored_groups = 0
    total_behavior_collision_incidents = 0
    total_observed_unsafe_branches = 0
    global_focal_min_count: int | None = None
    global_focal_max_count = 0
    complete_seed_focal_coverage_pairs = 0
    total_seed_focal_coverage_pairs = 0
    scenario_group_counts: dict[str, int] = {}

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    for result_file in result_files:
        payload = _load_json(result_file)
        setting = payload.get("setting", {})
        runtime = payload.get("runtime", {})
        diagnostics = payload.get("diagnostics", {})
        location = str(result_file)

        require(
            setting.get("cache_version") == spec["counterfactual_cache_version"],
            f"{location}: cache version mismatch",
        )
        require(
            setting.get("sumo_execution_protocol") == SUMO_EXECUTION_PROTOCOL,
            f"{location}: SUMO execution protocol mismatch",
        )
        require(
            setting.get("libsumo_version") == spec["expected_sumo_version"],
            f"{location}: setting libsumo version mismatch",
        )
        require(
            runtime.get("libsumo_version") == spec["expected_sumo_version"],
            f"{location}: runtime libsumo version mismatch",
        )
        require(
            runtime.get("determinism_environment") == EXPECTED_DETERMINISM_ENVIRONMENT,
            f"{location}: deterministic environment mismatch",
        )
        source_hash = str(runtime.get("source_tree_sha256", ""))
        source_hashes.add(source_hash)
        if expected_source_sha256 is not None:
            require(
                source_hash == expected_source_sha256,
                f"{location}: source tree hash mismatch",
            )
        cache_roots.add(str(setting.get("cache_root", "")))
        require(
            tuple(int(seed) for seed in setting.get("seeds", ())) == expected_seeds,
            f"{location}: source seed list mismatch",
        )
        require(
            int(setting.get("collection_shards", -1)) == expected_shards,
            f"{location}: collection shard count mismatch",
        )
        require(
            float(setting.get("min_tls_coverage", -1.0)) == 1.0,
            f"{location}: TLS coverage floor is not one",
        )

        selected = [str(name) for name in setting.get("scenarios", ())]
        require(set(selected) == set(diagnostics), f"{location}: selected scenarios mismatch")
        for scenario, diagnostic in diagnostics.items():
            scenario_location = f"{location}:{scenario}"
            if scenario in scenario_owner:
                errors.append(
                    f"{scenario_location}: duplicate scenario also in {scenario_owner[scenario]}"
                )
            scenario_owner[scenario] = location
            require(scenario in expected_entries, f"{scenario_location}: absent from manifest")
            if scenario in expected_entries:
                require(
                    diagnostic.get("city_group") == expected_entries[scenario]["city_group"],
                    f"{scenario_location}: city group mismatch",
                )
            rows = int(diagnostic.get("rows", -1))
            groups = int(diagnostic.get("groups", -1))
            branches = int(diagnostic.get("counterfactual_branches", -1))
            require(rows > 0 and groups > 0, f"{scenario_location}: empty dataset")
            require(
                groups >= int(min_groups_per_scenario),
                f"{scenario_location}: {groups} groups below required minimum "
                f"{int(min_groups_per_scenario)}",
            )
            scenario_group_counts[str(scenario)] = groups
            require(rows == branches, f"{scenario_location}: row/branch mismatch")
            total_rows += max(rows, 0)
            total_groups += max(groups, 0)
            total_branches += max(branches, 0)

            if spec["counterfactual_cache_version"] == COUNTERFACTUAL_CACHE_VERSION:
                expected_local_horizon = int(counterfactual["control_interval_sec"])
                expected_rollout_horizon = expected_local_horizon * max(
                    int(counterfactual["horizon_intervals"]),
                    1,
                )
                estimands = dict(diagnostic.get("mechanism_estimands", {}))
                one_step = dict(estimands.get("one_step_local_state", {}))
                require(
                    diagnostic.get("counterfactual_estimand")
                    == POLICY_CONSISTENT_ESTIMAND_PROTOCOL_V4,
                    f"{scenario_location}: policy-consistent estimand mismatch",
                )
                require(
                    tuple(diagnostic.get("mechanism_output_names", ()))
                    == OUTPUT_NAMES_V4,
                    f"{scenario_location}: mechanism output schema mismatch",
                )
                require(
                    int(diagnostic.get("local_mechanism_horizon_sec", -1))
                    == expected_local_horizon
                    and int(diagnostic.get("rollout_value_horizon_sec", -1))
                    == expected_rollout_horizon,
                    f"{scenario_location}: mechanism/value horizon mismatch",
                )
                require(
                    tuple(one_step.get("output_names", ()))
                    == ONE_STEP_OUTPUT_NAMES_V4
                    and int(one_step.get("horizon_sec", -1))
                    == expected_local_horizon
                    and int(one_step.get("analytic_prior_horizon_sec", -1))
                    == expected_local_horizon,
                    f"{scenario_location}: one-step prior/label horizon mismatch",
                )

            controllable = int(diagnostic.get("controllable_tls_count", -1))
            covered = int(diagnostic.get("covered_tls_count", -1))
            require(
                controllable > 0 and covered == controllable,
                f"{scenario_location}: incomplete TLS coverage",
            )
            require(
                float(diagnostic.get("tls_coverage_fraction", -1.0)) == 1.0,
                f"{scenario_location}: TLS coverage fraction is not one",
            )
            require(
                diagnostic.get("state_snapshot_protocol") == EXPECTED_SNAPSHOT_PROTOCOL,
                f"{scenario_location}: snapshot protocol mismatch",
            )
            require(
                diagnostic.get("sumo_execution_protocol")
                == SUMO_EXECUTION_PROTOCOL,
                f"{scenario_location}: SUMO execution protocol mismatch",
            )
            require(
                diagnostic.get("strict_safety_monitoring")
                == STRICT_SAFETY_MONITORING_V3,
                f"{scenario_location}: strict safety monitoring mismatch",
            )
            behavior_safety = dict(diagnostic.get("behavior_safety_audit", {}))
            require(
                behavior_safety.get("no_teleport_passed") is True
                and int(behavior_safety.get("starting_teleports", -1)) == 0
                and int(behavior_safety.get("ending_teleports", -1)) == 0,
                f"{scenario_location}: behavior teleport audit failed",
            )
            total_behavior_collision_incidents += max(
                int(behavior_safety.get("unique_collision_incidents", -1)), 0
            )
            counterfactual_safety = dict(
                diagnostic.get("counterfactual_safety_audit", {})
            )
            candidate_groups = int(
                counterfactual_safety.get("candidate_groups", -1)
            )
            retained_groups = int(
                counterfactual_safety.get("retained_groups", -1)
            )
            censored_groups = int(
                counterfactual_safety.get("censored_groups", -1)
            )
            candidate_branches = int(
                counterfactual_safety.get("candidate_branches", -1)
            )
            censored_branches = int(
                counterfactual_safety.get("censored_branches", -1)
            )
            observed_unsafe_branches = int(
                counterfactual_safety.get("observed_unsafe_branches", -1)
            )
            require(
                counterfactual_safety.get("protocol")
                == COUNTERFACTUAL_SAFETY_PROTOCOL_V3
                and counterfactual_safety.get("no_teleport_passed") is True
                and counterfactual_safety.get(
                    "symmetric_group_censoring_passed"
                )
                is True
                and candidate_groups == retained_groups + censored_groups
                and candidate_branches == branches + censored_branches
                and retained_groups == groups
                and len(counterfactual_safety.get("censored_group_ids", ()))
                == censored_groups,
                f"{scenario_location}: symmetric counterfactual safety audit failed",
            )
            require(
                len(
                    set(
                        counterfactual_safety.get(
                            "collision_incident_signatures", ()
                        )
                    )
                )
                == int(
                    counterfactual_safety.get(
                        "unique_collision_incidents_observed", -1
                    )
                ),
                f"{scenario_location}: collision incident deduplication failed",
            )
            total_candidate_groups += max(candidate_groups, 0)
            total_censored_groups += max(censored_groups, 0)
            total_observed_unsafe_branches += max(observed_unsafe_branches, 0)

            replay = diagnostic.get("counterfactual_replay_audit", {})
            replay_checks = int(replay.get("checks", 0))
            total_replay_checks += replay_checks
            require(
                bool(replay.get("passed"))
                and replay_checks > 0
                and float(replay.get("max_abs_difference", float("inf"))) == 0.0,
                f"{scenario_location}: counterfactual replay audit failed",
            )
            traces = diagnostic.get("behavior_trace_sha256_by_seed", {})
            require(set(traces) == expected_seed_keys, f"{scenario_location}: trace seeds mismatch")
            require(
                all(
                    len(str(value)) == 64
                    and set(str(value)) <= set("0123456789abcdef")
                    for value in traces.values()
                ),
                f"{scenario_location}: invalid behavior trace hash",
            )
            interval_counts = diagnostic.get("behavior_interval_count_by_seed", {})
            require(
                set(interval_counts) == expected_seed_keys
                and all(int(value) == expected_intervals for value in interval_counts.values()),
                f"{scenario_location}: behavior interval count mismatch",
            )
            opportunity_counts = diagnostic.get(
                "eligible_collection_opportunities_by_seed", {}
            )
            require(
                set(opportunity_counts) == expected_seed_keys
                and all(
                    0 < int(value) <= expected_opportunities
                    for value in opportunity_counts.values()
                ),
                f"{scenario_location}: decision opportunity count mismatch",
            )
            require(
                diagnostic.get("focal_selection_protocol")
                == EXPECTED_FOCAL_SELECTION_PROTOCOL,
                f"{scenario_location}: focal selection protocol mismatch",
            )
            focal_min_counts = diagnostic.get(
                "focal_selection_min_count_by_seed", {}
            )
            focal_max_counts = diagnostic.get(
                "focal_selection_max_count_by_seed", {}
            )
            require(
                set(focal_min_counts) == expected_seed_keys
                and all(int(value) >= 0 for value in focal_min_counts.values()),
                f"{scenario_location}: invalid focal selection minima",
            )
            require(
                set(focal_max_counts) == expected_seed_keys
                and all(int(value) >= 1 for value in focal_max_counts.values()),
                f"{scenario_location}: invalid focal selection maxima",
            )
            if focal_min_counts:
                total_seed_focal_coverage_pairs += len(focal_min_counts)
                complete_seed_focal_coverage_pairs += sum(
                    int(value) >= 1 for value in focal_min_counts.values()
                )
                observed_min = min(int(value) for value in focal_min_counts.values())
                global_focal_min_count = (
                    observed_min
                    if global_focal_min_count is None
                    else min(global_focal_min_count, observed_min)
                )
            if focal_max_counts:
                global_focal_max_count = max(
                    global_focal_max_count,
                    *(int(value) for value in focal_max_counts.values()),
                )
            require(
                int(diagnostic.get("captured_snapshot_count", -1))
                == sum(int(value) for value in opportunity_counts.values()),
                f"{scenario_location}: captured snapshot count mismatch",
            )
            reported_cache_files = [
                Path(value) for value in diagnostic.get("cache_files", ())
            ]
            cache_files = [
                Path(cache_root_override) / path.name
                if cache_root_override is not None
                else path
                for path in reported_cache_files
            ]
            require(
                len(cache_files) == expected_cache_count,
                f"{scenario_location}: cache file count mismatch",
            )
            expected_cache_files.extend(cache_files)

    expected_scenarios = set(expected_entries)
    observed_scenarios = set(scenario_owner)
    require(
        observed_scenarios == expected_scenarios,
        "scenario partition mismatch: "
        f"missing={sorted(expected_scenarios - observed_scenarios)}, "
        f"extra={sorted(observed_scenarios - expected_scenarios)}",
    )
    require(len(source_hashes) == 1 and "" not in source_hashes, "source hashes disagree")
    require(len(cache_roots) == 1 and "" not in cache_roots, "cache roots disagree")
    require(
        len(expected_cache_files) == len(set(expected_cache_files)),
        "cache file identities are not globally unique",
    )

    missing_cache_files: list[str] = []
    unexpected_cache_files: list[str] = []
    physically_loaded_rows = 0
    cache_file_sha256: dict[str, str] = {}
    if require_cache_files:
        missing_cache_files = [str(path) for path in expected_cache_files if not path.is_file()]
        require(not missing_cache_files, f"missing {len(missing_cache_files)} cache files")
        for path in expected_cache_files:
            if not path.is_file():
                continue
            try:
                dataset = load_mechanism_dataset(path)
                identity = dict(
                    dataset.metadata.get("counterfactual_cache_identity", {})
                )
                digest = hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode("utf-8")
                ).hexdigest()[:20]
                expected_name = f"{identity.get('scenario', '')}__{digest}.npz"
                require(path.name == expected_name, f"{path}: content address mismatch")
                _validate_counterfactual_cache_dataset(dataset, identity, path)
                cache_file_sha256[path.name] = _file_sha256(path)
                physically_loaded_rows += int(dataset.size)
            except Exception as error:
                errors.append(f"{path}: physical dataset validation failed: {error}")
        parents = {path.parent.resolve() for path in expected_cache_files}
        require(len(parents) == 1, "cache files do not share one cache root")
        if len(parents) == 1:
            actual = set(next(iter(parents)).glob("*.npz"))
            expected = {path.resolve() for path in expected_cache_files}
            unexpected_cache_files = sorted(str(path) for path in actual - expected)
            require(
                not unexpected_cache_files,
                f"cache root contains {len(unexpected_cache_files)} unexpected NPZ files",
            )
        require(
            physically_loaded_rows == total_rows,
            "physical cache row total does not match partition diagnostics",
        )

    summary = {
        "audit": "traffic_signal_counterfactual_cache",
        "passed": not errors,
        "errors": errors,
        "results_root": str(results_root.resolve()),
        "result_files": [str(path.resolve()) for path in result_files],
        "scenario_count": len(observed_scenarios),
        "city_group_count": len(
            {entry["city_group"] for entry in expected_entries.values()}
        ),
        "source_seeds": list(expected_seeds),
        "collection_shards": expected_shards,
        "cache_file_count": len(expected_cache_files),
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
        "total_groups": total_groups,
        "minimum_groups_per_scenario": int(min_groups_per_scenario),
        "scenario_group_counts": scenario_group_counts,
        "total_counterfactual_branches": total_branches,
        "total_replay_checks": total_replay_checks,
        "total_candidate_groups": total_candidate_groups,
        "total_censored_groups": total_censored_groups,
        "counterfactual_group_censor_fraction": (
            total_censored_groups / max(total_candidate_groups, 1)
        ),
        "total_behavior_collision_incidents": (
            total_behavior_collision_incidents
        ),
        "total_observed_unsafe_branches": total_observed_unsafe_branches,
        "global_focal_min_count": global_focal_min_count,
        "global_focal_max_count": global_focal_max_count,
        "complete_seed_focal_coverage_pairs": complete_seed_focal_coverage_pairs,
        "total_seed_focal_coverage_pairs": total_seed_focal_coverage_pairs,
        "missing_cache_files": missing_cache_files,
        "unexpected_cache_files": unexpected_cache_files,
        "physically_loaded_rows": physically_loaded_rows,
    }
    if errors:
        raise RuntimeError("counterfactual cache audit failed:\n- " + "\n- ".join(errors))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--require-cache-files", action="store_true")
    parser.add_argument("--cache-root-override", type=Path)
    parser.add_argument("--min-groups-per-scenario", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = audit_cache(
        results_root=args.results_root,
        spec_path=args.spec,
        expected_source_sha256=args.expected_source_sha256,
        require_cache_files=args.require_cache_files,
        cache_root_override=args.cache_root_override,
        min_groups_per_scenario=args.min_groups_per_scenario,
    )
    if args.out is not None:
        _atomic_write_json(args.out, summary)
    print(
        "cache audit passed: "
        f"{summary['scenario_count']} scenarios, "
        f"{summary['cache_file_count']} files, "
        f"{summary['total_rows']} rows"
    )


if __name__ == "__main__":
    main()
