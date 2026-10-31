#!/usr/bin/env python3
"""Independently audit a partitioned TSC benchmark safety survey."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    STRICT_SAFETY_MONITORING_V3,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    SUMO_EXECUTION_PROTOCOL,
)


DEFAULT_SPEC = (
    PROJECT_ROOT
    / "cf_h2o/config/traffic_signal_tsc_v16_benchmark_aware_development.json"
)
EXPECTED_DETERMINISM_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
}
AUDITED_METRICS = (
    "collision_events",
    "collision_event_steps",
    "collision_incidents",
    "collision_rate",
    "collision_incident_rate",
    "emergency_stops",
    "starting_teleports",
    "ending_teleports",
    "mean_system_vehicles_per_controlled_lane",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot aggregate an empty safety-survey cell")
    return float(statistics.fmean(materialized))


def audit_safety_survey(
    *,
    results_root: Path,
    spec_path: Path,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    spec = _load_json(spec_path)
    manifest_path = Path(spec["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest = _load_json(manifest_path.resolve())
    manifest_entries = {
        str(item["scenario"]): dict(item) for item in manifest["scenarios"]
    }
    assignments = {
        str(node): tuple(str(scenario) for scenario in scenarios)
        for node, scenarios in spec["cache_assignments"].items()
    }
    survey = spec["safety_survey"]
    counterfactual = spec["counterfactual"]
    expected_policies = tuple(str(value) for value in survey["policies"])
    expected_seeds = tuple(int(value) for value in survey["seeds"])
    expected_scenarios = {
        scenario for scenarios in assignments.values() for scenario in scenarios
    }
    expected_identities = {
        (scenario, policy, seed)
        for scenario in expected_scenarios
        for policy in expected_policies
        for seed in expected_seeds
    }

    result_files = sorted(results_root.glob("**/safety_survey.json"))
    if not result_files:
        raise FileNotFoundError(
            f"no safety_survey.json files found below {results_root}"
        )

    errors: list[str] = []
    source_hashes: set[str] = set()
    observed_nodes: set[str] = set()
    observed_identities: set[tuple[str, str, int]] = set()
    all_rows: list[dict[str, Any]] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    for result_file in result_files:
        payload = _load_json(result_file)
        setting = dict(payload.get("setting", {}))
        runtime = dict(payload.get("runtime", {}))
        node = result_file.parent.name
        location = str(result_file)
        require(node in assignments, f"{location}: unknown node partition {node}")
        require(node not in observed_nodes, f"{location}: duplicate node partition")
        observed_nodes.add(node)
        expected_node_scenarios = assignments.get(node, ())

        require(
            payload.get("experiment")
            == "traffic_signal_benchmark_aware_safety_survey",
            f"{location}: experiment tag mismatch",
        )
        require(payload.get("passed") is True, f"{location}: shard did not pass")
        require(not payload.get("errors"), f"{location}: shard reports errors")
        require(
            tuple(str(value) for value in setting.get("scenarios", ()))
            == expected_node_scenarios,
            f"{location}: node scenario assignment mismatch",
        )
        require(
            tuple(str(value) for value in setting.get("policies", ()))
            == expected_policies,
            f"{location}: policy list mismatch",
        )
        require(
            tuple(int(value) for value in setting.get("seeds", ()))
            == expected_seeds,
            f"{location}: seed list mismatch",
        )
        require(
            float(setting.get("duration_sec", -1.0))
            == float(survey["duration_sec"]),
            f"{location}: duration mismatch",
        )
        require(
            int(setting.get("control_interval_sec", -1))
            == int(counterfactual["control_interval_sec"]),
            f"{location}: control interval mismatch",
        )
        require(
            float(setting.get("warmup_sec", -1.0))
            == float(counterfactual["warmup_sec"]),
            f"{location}: warmup mismatch",
        )
        require(
            setting.get("libsumo_version") == spec["expected_sumo_version"],
            f"{location}: setting SUMO version mismatch",
        )
        require(
            setting.get("sumo_execution_protocol") == SUMO_EXECUTION_PROTOCOL,
            f"{location}: SUMO execution protocol mismatch",
        )
        require(
            setting.get("strict_safety_monitoring")
            == STRICT_SAFETY_MONITORING_V3,
            f"{location}: safety monitoring protocol mismatch",
        )
        require(
            setting.get("collision_interpretation")
            == "raw_benchmark_events_audited_not_blanket_controller_failure",
            f"{location}: collision interpretation mismatch",
        )
        require(
            runtime.get("libsumo_version") == spec["expected_sumo_version"],
            f"{location}: runtime SUMO version mismatch",
        )
        require(
            runtime.get("determinism_environment")
            == EXPECTED_DETERMINISM_ENVIRONMENT,
            f"{location}: deterministic environment mismatch",
        )
        source_hash = str(runtime.get("source_tree_sha256", ""))
        source_hashes.add(source_hash)
        if expected_source_sha256 is not None:
            require(
                source_hash == expected_source_sha256,
                f"{location}: source tree hash mismatch",
            )

        rows = [dict(row) for row in payload.get("rows", ())]
        expected_row_count = (
            len(expected_node_scenarios)
            * len(expected_policies)
            * len(expected_seeds)
        )
        require(
            int(payload.get("expected_row_count", -1)) == expected_row_count,
            f"{location}: declared expected row count mismatch",
        )
        require(
            int(payload.get("row_count", -1)) == expected_row_count
            and len(rows) == expected_row_count,
            f"{location}: row count mismatch",
        )

        shard_collision_rollouts = 0
        shard_identities: set[tuple[str, str, int]] = set()
        for row in rows:
            scenario = str(row.get("scenario", ""))
            policy = str(row.get("policy", ""))
            seed = int(row.get("seed", -1))
            identity = (scenario, policy, seed)
            row_location = f"{location}:{scenario}/{policy}/seed={seed}"
            require(
                identity not in shard_identities,
                f"{row_location}: duplicate rollout identity in shard",
            )
            require(
                identity not in observed_identities,
                f"{row_location}: duplicate rollout identity across shards",
            )
            shard_identities.add(identity)
            observed_identities.add(identity)
            require(
                scenario in expected_node_scenarios,
                f"{row_location}: scenario is outside node assignment",
            )
            require(
                scenario in manifest_entries,
                f"{row_location}: scenario is absent from manifest",
            )
            if scenario in manifest_entries:
                require(
                    row.get("city_group")
                    == manifest_entries[scenario]["city_group"],
                    f"{row_location}: city group mismatch",
                )
            metrics = dict(row.get("metrics", {}))
            require(metrics.get("ok") is True, f"{row_location}: rollout failed")
            require(
                metrics.get("sumo_execution_protocol") == SUMO_EXECUTION_PROTOCOL,
                f"{row_location}: rollout SUMO protocol mismatch",
            )
            require(
                metrics.get("strict_safety_monitoring")
                == STRICT_SAFETY_MONITORING_V3,
                f"{row_location}: rollout safety protocol mismatch",
            )
            for metric in AUDITED_METRICS:
                value = metrics.get(metric)
                require(
                    isinstance(value, (int, float))
                    and math.isfinite(float(value)),
                    f"{row_location}: invalid metric {metric}",
                )
            collision_events = int(metrics.get("collision_events", -1))
            collision_event_steps = int(metrics.get("collision_event_steps", -1))
            collision_incidents = int(metrics.get("collision_incidents", -1))
            require(
                collision_events >= 0
                and 0 <= collision_event_steps <= collision_events
                and 0 <= collision_incidents <= collision_events,
                f"{row_location}: inconsistent collision audit",
            )
            require(
                int(metrics.get("starting_teleports", -1)) == 0
                and int(metrics.get("ending_teleports", -1)) == 0,
                f"{row_location}: teleport event",
            )
            shard_collision_rollouts += collision_incidents > 0
            all_rows.append(row)

        require(
            int(payload.get("raw_collision_rollout_count", -1))
            == shard_collision_rollouts,
            f"{location}: raw-collision rollout count mismatch",
        )
        require(
            math.isclose(
                float(payload.get("raw_collision_rollout_fraction", -1.0)),
                shard_collision_rollouts / max(expected_row_count, 1),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{location}: raw-collision rollout fraction mismatch",
        )

        reported_aggregate = dict(payload.get("aggregate", {}))
        require(
            set(reported_aggregate) == set(expected_node_scenarios),
            f"{location}: aggregate scenario keys mismatch",
        )
        for scenario in expected_node_scenarios:
            scenario_aggregate = dict(reported_aggregate.get(scenario, {}))
            require(
                set(scenario_aggregate) == set(expected_policies),
                f"{location}:{scenario}: aggregate policy keys mismatch",
            )
            for policy in expected_policies:
                reported_metrics = dict(scenario_aggregate.get(policy, {}))
                require(
                    set(reported_metrics) == set(AUDITED_METRICS),
                    f"{location}:{scenario}/{policy}: aggregate metric keys mismatch",
                )
                cell = [
                    row
                    for row in rows
                    if row.get("scenario") == scenario
                    and row.get("policy") == policy
                ]
                for metric in AUDITED_METRICS:
                    expected_value = _mean(
                        row["metrics"][metric] for row in cell
                    )
                    reported_value = reported_metrics.get(metric)
                    require(
                        isinstance(reported_value, (int, float))
                        and math.isclose(
                            float(reported_value),
                            expected_value,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        ),
                        f"{location}:{scenario}/{policy}: aggregate {metric} mismatch",
                    )

    require(
        observed_nodes == set(assignments),
        "node partition mismatch: "
        f"missing={sorted(set(assignments) - observed_nodes)}, "
        f"extra={sorted(observed_nodes - set(assignments))}",
    )
    require(
        observed_identities == expected_identities,
        "rollout identity matrix mismatch: "
        f"missing={len(expected_identities - observed_identities)}, "
        f"extra={len(observed_identities - expected_identities)}",
    )
    require(
        set(manifest_entries) == expected_scenarios,
        "spec assignments do not exactly cover the benchmark manifest",
    )
    require(
        len(source_hashes) == 1 and "" not in source_hashes,
        "source tree hashes disagree or are missing",
    )

    policy_summary: dict[str, dict[str, float | int]] = {}
    for policy in expected_policies:
        policy_rows = [row for row in all_rows if row["policy"] == policy]
        policy_summary[policy] = {
            "rollout_count": len(policy_rows),
            "collision_rollout_count": sum(
                int(row["metrics"]["collision_incidents"]) > 0
                for row in policy_rows
            ),
            "collision_events": sum(
                int(row["metrics"]["collision_events"]) for row in policy_rows
            ),
            "collision_incidents": sum(
                int(row["metrics"]["collision_incidents"])
                for row in policy_rows
            ),
            "starting_teleports": sum(
                int(row["metrics"]["starting_teleports"])
                for row in policy_rows
            ),
            "ending_teleports": sum(
                int(row["metrics"]["ending_teleports"])
                for row in policy_rows
            ),
        }

    summary = {
        "audit": "traffic_signal_benchmark_aware_safety_survey",
        "passed": not errors,
        "results_root": str(results_root.resolve()),
        "result_files": [str(path.resolve()) for path in result_files],
        "node_count": len(observed_nodes),
        "scenario_count": len(expected_scenarios),
        "policy_count": len(expected_policies),
        "seed_count": len(expected_seeds),
        "rollout_count": len(all_rows),
        "expected_rollout_count": len(expected_identities),
        "source_tree_sha256": next(iter(source_hashes), ""),
        "raw_collision_rollout_count": sum(
            int(row["metrics"]["collision_incidents"]) > 0
            for row in all_rows
        ),
        "collision_events": sum(
            int(row["metrics"]["collision_events"]) for row in all_rows
        ),
        "collision_incidents": sum(
            int(row["metrics"]["collision_incidents"]) for row in all_rows
        ),
        "starting_teleports": sum(
            int(row["metrics"]["starting_teleports"]) for row in all_rows
        ),
        "ending_teleports": sum(
            int(row["metrics"]["ending_teleports"]) for row in all_rows
        ),
        "policy_summary": policy_summary,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("safety survey audit failed:\n- " + "\n- ".join(errors))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = audit_safety_survey(
        results_root=args.results_root,
        spec_path=args.spec,
        expected_source_sha256=args.expected_source_sha256,
    )
    if args.out is not None:
        _atomic_write_json(args.out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
