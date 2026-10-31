"""Audit and aggregate the authorized v43 external closed-loop matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    METHOD_POLICY,
    POLICIES,
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import PROTOCOL
from cf_h2o.eval.traffic_signal_external_full_heldout_evaluation import (
    RESULT_PROTOCOL as OFFLINE_RESULT_PROTOCOL,
)


AGGREGATE_PROTOCOL = "tsc-v43r39-external-closed-loop-aggregate-v4"
LAUNCH_PROTOCOL = "v43r39-external-closed-loop-submission-v4"
AUTHORIZATION_DECISION = "authorize_closed_loop_external_confirmation"
PRIMARY_METRIC = "mean_tripinfo_waiting_time"
METRIC_DIRECTIONS = {
    PRIMARY_METRIC: "lower",
    "mean_tripinfo_duration": "lower",
    "mean_tripinfo_time_loss": "lower",
    "mean_completed_tripinfo_waiting_time": "lower",
    "mean_completed_tripinfo_duration": "lower",
    "tripinfo_unfinished_fraction": "lower",
    "system_vehicle_hours": "lower",
    "mean_queue_per_lane": "lower",
    "completion_ratio": "higher",
    "departure_service_ratio": "higher",
    "throughput_ratio": "higher",
    "collision_incidents": "lower",
    "collision_incident_rate": "lower",
    "emergency_stop_rate": "lower",
}
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_REPLICATES = 10_000


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _finite_number(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite closed-loop metric {label}: {value!r}")
    return number


def _expected_matrix(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], tuple[int, ...], tuple[tuple[str, int, str], ...]]:
    city_scenarios = {
        str(city): tuple(str(scenario) for scenario in scenarios)
        for city, scenarios in protocol["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    seeds = tuple(
        int(seed)
        for seed in protocol["target_protocol"]["closed_loop_seeds"]
    )
    matrix = tuple(
        (scenario, seed, policy)
        for scenarios in city_scenarios.values()
        for scenario in scenarios
        for seed in seeds
        for policy in POLICIES
    )
    if (
        len(city_scenarios) != 2
        or sum(len(value) for value in city_scenarios.values()) != 4
        or len(seeds) != 2
        or len(matrix) != 56
        or len(set(matrix)) != len(matrix)
    ):
        raise ValueError("v43 closed-loop matrix cardinality changed")
    return city_scenarios, seeds, matrix


def _launch_nodes(
    launch: Mapping[str, Any],
    matrix: Sequence[tuple[str, int, str]],
) -> dict[tuple[str, int, str], str]:
    expected = set(matrix)
    observed: dict[tuple[str, int, str], str] = {}
    for spec in launch.get("specs", ()):
        signature = str(spec.get("signature", ""))
        prefix = "CFCMT/v43r39/closed-loop-v4/"
        if not signature.startswith(prefix):
            raise ValueError(f"unexpected closed-loop launch signature: {signature}")
        suffix = signature[len(prefix) :]
        try:
            scenario, seed_text, policy = suffix.split("/", 2)
            seed = int(seed_text.removeprefix("seed-"))
        except Exception as exc:
            raise ValueError(
                f"malformed closed-loop launch signature: {signature}"
            ) from exc
        key = (scenario, seed, policy)
        if key not in expected or key in observed:
            raise ValueError(f"duplicate or unexpected launch combination: {key}")
        observed[key] = str(spec["require_node"])
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        raise ValueError(f"closed-loop launch matrix is incomplete: {missing[:3]}")
    return observed


def _scenario_to_city(
    city_scenarios: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    mapping = {
        scenario: city
        for city, scenarios in city_scenarios.items()
        for scenario in scenarios
    }
    if len(mapping) != sum(len(value) for value in city_scenarios.values()):
        raise ValueError("closed-loop scenario belongs to multiple cities")
    return mapping


def _validate_result(
    *,
    result_path: Path,
    expected_key: tuple[str, int, str],
    expected_city: str,
    expected_node: str,
    protocol_sha256: str,
    launch: Mapping[str, Any],
    authorization: Mapping[str, Any],
    joint: Mapping[str, Any],
    expected_sumo_version: str,
) -> dict[str, Any]:
    result = _read_json(result_path)
    scenario, seed, policy = expected_key
    if (
        result.get("protocol") != RESULT_PROTOCOL
        or result.get("scenario") != scenario
        or int(result.get("seed", -1)) != seed
        or result.get("policy") != policy
        or result.get("city") != expected_city
    ):
        raise ValueError(f"closed-loop result identity mismatch: {result_path}")
    expected_hashes = {
        "protocol_spec_sha256": protocol_sha256,
        "offline_authorization_sha256": launch[
            "offline_authorization_sha256"
        ],
        "joint_freeze_audit_sha256": launch["joint_freeze_audit_sha256"],
        "external_manifest_sha256": launch["external_manifest_sha256"],
        "conversion_manifest_sha256": launch[
            "conversion_manifest_sha256"
        ],
        "conversion_tree_sha256": launch["conversion_tree_sha256"],
    }
    for key, expected in expected_hashes.items():
        if result.get(key) != expected:
            raise ValueError(f"closed-loop provenance mismatch for {key}: {result_path}")
    runtime = dict(result.get("runtime", {}))
    if (
        result.get("hostname") != expected_node
        or runtime.get("hostname") != expected_node
        or runtime.get("source_tree_sha256") != launch["source_tree_sha256"]
        or result.get("sumo_version") != expected_sumo_version
        or runtime.get("libsumo_version") != expected_sumo_version
    ):
        raise ValueError(f"closed-loop runtime identity mismatch: {result_path}")
    if result.get("model_artifacts") != joint["cities"][expected_city]:
        raise ValueError(f"closed-loop frozen model identity mismatch: {result_path}")
    selected = result.get("selected_candidate")
    if policy == METHOD_POLICY:
        expected_selected = authorization["city_results"][expected_city][
            "selected_candidate"
        ]
        if selected != expected_selected:
            raise ValueError(f"closed-loop selected candidate changed: {result_path}")
        blend_audit = dict(result.get("anchored_blend_audit") or {})
        if (
            blend_audit.get("protocol")
            != "online-frozen-anchored-blend-audit-v1"
            or int(blend_audit.get("prediction_calls", 0)) <= 0
            or int(blend_audit.get("action_groups", 0)) <= 0
        ):
            raise ValueError(f"closed-loop method did not execute its frozen blend: {result_path}")
    elif selected is not None or result.get("anchored_blend_audit") is not None:
        raise ValueError(f"baseline carries method-only state: {result_path}")

    tripinfo = result_path.parent / "tripinfo.xml"
    evidence = dict(result.get("tripinfo_evidence", {}))
    if (
        evidence.get("filename") != tripinfo.name
        or not tripinfo.is_file()
        or int(evidence.get("size_bytes", -1)) != tripinfo.stat().st_size
        or evidence.get("sha256") != _sha256(tripinfo)
    ):
        raise ValueError(f"closed-loop tripinfo evidence mismatch: {result_path}")

    metrics = dict(result.get("metrics", {}))
    if not bool(metrics.get("ok", False)):
        raise ValueError(f"closed-loop rollout is not valid: {result_path}")
    finite_metrics = {
        metric: _finite_number(metrics.get(metric), label=f"{expected_key}/{metric}")
        for metric in METRIC_DIRECTIONS
    }
    teleport_keys = ("starting_teleports", "ending_teleports")
    if any(int(metrics.get(key, -1)) != 0 for key in teleport_keys):
        raise ValueError(f"closed-loop teleport violation: {result_path}")
    tripinfo_count = int(metrics.get("tripinfo_count", -1))
    completed_count = int(metrics.get("tripinfo_completed_count", -1))
    unfinished_count = int(metrics.get("tripinfo_unfinished_count", -1))
    arrived = int(metrics.get("arrived", -1))
    departed = int(metrics.get("departed", -1))
    population = int(metrics.get("demand_population_at_horizon", -1))
    if (
        tripinfo_count <= 0
        or tripinfo_count != departed
        or completed_count != arrived
        or unfinished_count != departed - arrived
        or not (0 <= arrived <= departed <= population)
    ):
        raise ValueError(f"closed-loop population accounting mismatch: {result_path}")
    return {
        "city": expected_city,
        "scenario": scenario,
        "seed": seed,
        "policy": policy,
        "node": expected_node,
        "result_sha256": _sha256(result_path),
        "tripinfo_sha256": evidence["sha256"],
        "tripinfo_size_bytes": int(evidence["size_bytes"]),
        "elapsed_sec": _finite_number(result.get("elapsed_sec"), label="elapsed_sec"),
        "metrics": {
            **finite_metrics,
            "tripinfo_count": tripinfo_count,
            "tripinfo_completed_count": completed_count,
            "tripinfo_unfinished_count": unfinished_count,
            "arrived": arrived,
            "departed": departed,
            "demand_population_at_horizon": population,
            "emergency_stops": int(metrics.get("emergency_stops", 0)),
            "collision_events": int(metrics.get("collision_events", 0)),
        },
    }


def _mean_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    return float(np.mean([float(row["metrics"][metric]) for row in rows]))


def _summaries(
    records: Sequence[Mapping[str, Any]],
    city_scenarios: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scenario_summary: dict[str, Any] = {}
    city_summary: dict[str, Any] = {}
    for city, scenarios in city_scenarios.items():
        city_summary[city] = {}
        for scenario in scenarios:
            scenario_summary[scenario] = {}
            for policy in POLICIES:
                rows = [
                    row
                    for row in records
                    if row["scenario"] == scenario and row["policy"] == policy
                ]
                scenario_summary[scenario][policy] = {
                    metric: _mean_metric(rows, metric)
                    for metric in METRIC_DIRECTIONS
                }
        for policy in POLICIES:
            city_summary[city][policy] = {
                metric: float(
                    np.mean(
                        [
                            scenario_summary[scenario][policy][metric]
                            for scenario in scenarios
                        ]
                    )
                )
                for metric in METRIC_DIRECTIONS
            }
    macro_summary = {
        policy: {
            metric: float(
                np.mean(
                    [city_summary[city][policy][metric] for city in city_scenarios]
                )
            )
            for metric in METRIC_DIRECTIONS
        }
        for policy in POLICIES
    }
    return scenario_summary, city_summary, macro_summary


def _oriented_improvement(method: float, baseline: float, direction: str) -> float:
    if direction == "lower":
        return float(baseline - method)
    return float(method - baseline)


def _bootstrap_primary_improvement(
    records: Sequence[Mapping[str, Any]],
    city_scenarios: Mapping[str, Sequence[str]],
    *,
    baseline: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    lookup = {
        (row["city"], row["scenario"], row["seed"], row["policy"]): float(
            row["metrics"][PRIMARY_METRIC]
        )
        for row in records
    }
    cities = tuple(city_scenarios)
    seeds_by_scenario = {
        scenario: tuple(
            sorted(
                {
                    int(row["seed"])
                    for row in records
                    if row["scenario"] == scenario
                }
            )
        )
        for scenarios in city_scenarios.values()
        for scenario in scenarios
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        city_values = []
        for city_index in rng.integers(0, len(cities), size=len(cities)):
            city = cities[int(city_index)]
            scenarios = tuple(city_scenarios[city])
            scenario_values = []
            for scenario_index in rng.integers(
                0, len(scenarios), size=len(scenarios)
            ):
                scenario = scenarios[int(scenario_index)]
                seeds = seeds_by_scenario[scenario]
                seed_values = []
                for seed_index in rng.integers(0, len(seeds), size=len(seeds)):
                    rollout_seed = seeds[int(seed_index)]
                    method = lookup[
                        (city, scenario, rollout_seed, METHOD_POLICY)
                    ]
                    comparator = lookup[(city, scenario, rollout_seed, baseline)]
                    seed_values.append(comparator - method)
                scenario_values.append(float(np.mean(seed_values)))
            city_values.append(float(np.mean(scenario_values)))
        draws[replicate] = float(np.mean(city_values))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "protocol": "paired-hierarchical-descriptive-bootstrap-v1",
        "replicates": int(replicates),
        "seed": int(seed),
        "sampling_units": "city_then_scenario_then_seed_with_paired_policies",
        "independent_target_city_count": len(cities),
        "ci_is_descriptive_due_to_two_target_cities": True,
        "improvement_95pct_ci": [float(lower), float(upper)],
        "bootstrap_probability_positive": float(np.mean(draws > 0.0)),
    }


def _comparisons(
    records: Sequence[Mapping[str, Any]],
    city_scenarios: Mapping[str, Sequence[str]],
    city_summary: Mapping[str, Any],
    macro_summary: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    comparisons = {}
    for index, baseline in enumerate(POLICIES):
        if baseline == METHOD_POLICY:
            continue
        metric_rows = {}
        for metric, direction in METRIC_DIRECTIONS.items():
            method = float(macro_summary[METHOD_POLICY][metric])
            comparator = float(macro_summary[baseline][metric])
            improvement = _oriented_improvement(method, comparator, direction)
            metric_rows[metric] = {
                "direction": direction,
                "method": method,
                "baseline": comparator,
                "oriented_improvement": improvement,
                "relative_improvement": float(
                    improvement / max(abs(comparator), 1e-12)
                ),
                "city_oriented_improvements": {
                    city: _oriented_improvement(
                        float(city_summary[city][METHOD_POLICY][metric]),
                        float(city_summary[city][baseline][metric]),
                        direction,
                    )
                    for city in city_scenarios
                },
            }
        metric_rows[PRIMARY_METRIC]["descriptive_bootstrap"] = (
            _bootstrap_primary_improvement(
                records,
                city_scenarios,
                baseline=baseline,
                replicates=bootstrap_replicates,
                seed=BOOTSTRAP_SEED + index,
            )
        )
        comparisons[baseline] = metric_rows
    return comparisons


def _collision_audit(
    records: Sequence[Mapping[str, Any]],
    macro_summary: Mapping[str, Any],
) -> dict[str, Any]:
    method_rows = {
        (row["scenario"], row["seed"]): int(
            row["metrics"]["collision_incidents"]
        )
        for row in records
        if row["policy"] == METHOD_POLICY
    }
    method_vs_baselines = {}
    for baseline in POLICIES:
        if baseline == METHOD_POLICY:
            continue
        paired = []
        for row in records:
            if row["policy"] != baseline:
                continue
            key = (row["scenario"], row["seed"])
            method_value = method_rows[key]
            baseline_value = int(row["metrics"]["collision_incidents"])
            paired.append(
                {
                    "scenario": key[0],
                    "seed": key[1],
                    "method": method_value,
                    "baseline": baseline_value,
                    "method_noninferior": method_value <= baseline_value,
                }
            )
        method_vs_baselines[baseline] = {
            "macro_method": float(
                macro_summary[METHOD_POLICY]["collision_incidents"]
            ),
            "macro_baseline": float(
                macro_summary[baseline]["collision_incidents"]
            ),
            "macro_noninferior": (
                float(macro_summary[METHOD_POLICY]["collision_incidents"])
                <= float(macro_summary[baseline]["collision_incidents"])
            ),
            "all_paired_rollouts_noninferior": all(
                row["method_noninferior"] for row in paired
            ),
            "paired_noninferior_fraction": float(
                np.mean([row["method_noninferior"] for row in paired])
            ),
            "paired_rollouts": paired,
        }
    return {
        "protocol": "teleport-fail-fast-collision-audit-v1",
        "teleport_validity_requirement": "zero_starting_and_ending_teleports",
        "collision_validity_requirement": "audited_not_censored",
        "total_collision_events": int(
            sum(row["metrics"]["collision_events"] for row in records)
        ),
        "total_collision_incidents": int(
            sum(row["metrics"]["collision_incidents"] for row in records)
        ),
        "rollouts_with_collision_incidents": int(
            sum(row["metrics"]["collision_incidents"] > 0 for row in records)
        ),
        "method_vs_baselines": method_vs_baselines,
    }


def aggregate_closed_loop_results(
    *,
    protocol_spec_path: Path,
    offline_authorization_path: Path,
    joint_freeze_audit_path: Path,
    launch_manifest_path: Path,
    result_root: Path,
    expected_sumo_version: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    protocol = _read_json(protocol_spec_path)
    authorization = _read_json(offline_authorization_path)
    joint = _read_json(joint_freeze_audit_path)
    launch = _read_json(launch_manifest_path)
    if (
        protocol.get("protocol") != PROTOCOL
        or authorization.get("protocol") != OFFLINE_RESULT_PROTOCOL
        or authorization.get("decision") != AUTHORIZATION_DECISION
        or not bool(
            authorization.get("offline_confirmation_gate", {}).get(
                "passed", False
            )
        )
        or joint.get("status") != "PASS"
        or launch.get("protocol") != LAUNCH_PROTOCOL
        or not bool(launch.get("submitted", False))
        or int(launch.get("task_count", -1)) != 56
        or launch.get("result_protocol") != RESULT_PROTOCOL
    ):
        raise ValueError("closed-loop aggregate lacks authorized protocol evidence")
    if (
        _sha256(offline_authorization_path)
        != launch["offline_authorization_sha256"]
        or _sha256(joint_freeze_audit_path)
        != launch["joint_freeze_audit_sha256"]
        or authorization.get("joint_freeze_audit_sha256")
        != launch["joint_freeze_audit_sha256"]
    ):
        raise ValueError("closed-loop aggregate authorization hashes changed")
    if int(bootstrap_replicates) <= 0:
        raise ValueError("bootstrap_replicates must be positive")

    city_scenarios, seeds, matrix = _expected_matrix(protocol)
    scenario_city = _scenario_to_city(city_scenarios)
    launch_nodes = _launch_nodes(launch, matrix)
    result_root = Path(result_root)
    discovered_results = tuple(sorted(result_root.glob("**/result.json")))
    discovered_tripinfos = tuple(sorted(result_root.glob("**/tripinfo.xml")))
    temporary_files = tuple(sorted(result_root.glob("**/*.tmp-*")))
    if (
        len(discovered_results) != len(matrix)
        or len(discovered_tripinfos) != len(matrix)
        or temporary_files
    ):
        raise ValueError(
            "closed-loop evidence tree is incomplete or contains temporary files"
        )

    protocol_sha256 = _sha256(protocol_spec_path)
    records = []
    expected_paths = set()
    for key in matrix:
        scenario, seed, policy = key
        result_path = result_root / scenario / f"seed_{seed}" / policy / "result.json"
        expected_paths.add(result_path.resolve())
        if not result_path.is_file():
            raise FileNotFoundError(f"missing closed-loop result: {result_path}")
        records.append(
            _validate_result(
                result_path=result_path,
                expected_key=key,
                expected_city=scenario_city[scenario],
                expected_node=launch_nodes[key],
                protocol_sha256=protocol_sha256,
                launch=launch,
                authorization=authorization,
                joint=joint,
                expected_sumo_version=expected_sumo_version,
            )
        )
    if {path.resolve() for path in discovered_results} != expected_paths:
        raise ValueError("closed-loop evidence tree contains unexpected results")

    scenario_summary, city_summary, macro_summary = _summaries(
        records, city_scenarios
    )
    comparisons = _comparisons(
        records,
        city_scenarios,
        city_summary,
        macro_summary,
        bootstrap_replicates=int(bootstrap_replicates),
    )
    collision_audit = _collision_audit(records, macro_summary)
    return {
        "protocol": AGGREGATE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validity_gate": {
            "passed": True,
            "decision": "complete_confirmatory_closed_loop_reporting",
            "performance_threshold_used": False,
            "expected_rollout_count": len(matrix),
            "valid_rollout_count": len(records),
            "expected_tripinfo_count": len(matrix),
            "valid_tripinfo_count": len(discovered_tripinfos),
            "temporary_file_count": 0,
            "teleport_count": 0,
            "collision_incidents_are_audited_not_a_validity_failure": True,
            "collision_incident_count": collision_audit[
                "total_collision_incidents"
            ],
        },
        "provenance": {
            "protocol_spec_sha256": protocol_sha256,
            "offline_authorization_sha256": _sha256(
                offline_authorization_path
            ),
            "joint_freeze_audit_sha256": _sha256(joint_freeze_audit_path),
            "launch_manifest_sha256": _sha256(launch_manifest_path),
            "source_tree_sha256": launch["source_tree_sha256"],
            "external_manifest_sha256": launch["external_manifest_sha256"],
            "conversion_manifest_sha256": launch[
                "conversion_manifest_sha256"
            ],
            "conversion_tree_sha256": launch["conversion_tree_sha256"],
            "sumo_version": expected_sumo_version,
        },
        "estimands": {
            "preregistered_primary_metric": protocol["closed_loop"][
                "primary_metric"
            ],
            "implemented_primary_metric": PRIMARY_METRIC,
            "implemented_primary_population": (
                "all departed vehicles, including write-unfinished records whose "
                "duration and waiting are accrued to the fixed 3600-second horizon"
            ),
            "fixed_horizon_burden_complement": "system_vehicle_hours",
            "city_weighting": "equal scenario weight within city",
            "macro_weighting": "equal target-city weight",
            "bootstrap_scope": (
                "descriptive paired hierarchy; only two independent target cities"
            ),
        },
        "matrix": {
            "cities": {key: list(value) for key, value in city_scenarios.items()},
            "seeds": list(seeds),
            "policies": list(POLICIES),
            "rollout_count": len(records),
        },
        "records": records,
        "scenario_summary": scenario_summary,
        "city_summary": city_summary,
        "macro_summary": macro_summary,
        "method_comparisons": comparisons,
        "collision_audit": collision_audit,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    macro = payload["macro_summary"]
    cities = tuple(payload["matrix"]["cities"])
    lines = [
        "# V43 External Closed-Loop Confirmation",
        "",
        f"Validity gate: **{'PASS' if payload['validity_gate']['passed'] else 'FAIL'}**  ",
        f"Rollouts: {payload['validity_gate']['valid_rollout_count']}/"
        f"{payload['validity_gate']['expected_rollout_count']}  ",
        "SUMO: `" + payload["provenance"]["sumo_version"] + "`  ",
        "",
        "The primary waiting-time metric covers all departed vehicles, including "
        "write-unfinished records accrued to the fixed 3,600-second horizon. "
        "Completed-only waiting, system vehicle-hours and completion ratio are "
        "reported beside it to expose horizon and throughput trade-offs.",
        "",
        "## Macro Results",
        "",
        "| Policy | All-departed wait (s) | Completed wait (s) | Duration (s) | System veh-h | Unfinished | Completion | Incidents |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        row = macro[policy]
        lines.append(
            f"| {policy} | {row[PRIMARY_METRIC]:.3f} | "
            f"{row['mean_completed_tripinfo_waiting_time']:.3f} | "
            f"{row['mean_tripinfo_duration']:.3f} | "
            f"{row['system_vehicle_hours']:.3f} | "
            f"{row['tripinfo_unfinished_fraction']:.4f} | "
            f"{row['completion_ratio']:.4f} | "
            f"{row['collision_incidents']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Method Comparisons",
            "",
            "Positive values favor CFCMT. Confidence intervals are descriptive "
            "paired hierarchical bootstrap intervals, not population-level "
            "inference, because there are only two target cities.",
            "",
            "| Baseline | Waiting gain (s) | Relative gain | 95% interval | City gains (s) | Collision noninferior |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for baseline, metrics in payload["method_comparisons"].items():
        row = metrics[PRIMARY_METRIC]
        lower, upper = row["descriptive_bootstrap"]["improvement_95pct_ci"]
        city_text = ", ".join(
            f"{city}={row['city_oriented_improvements'][city]:.3f}"
            for city in cities
        )
        safety = payload["collision_audit"]["method_vs_baselines"][baseline]
        lines.append(
            f"| {baseline} | {row['oriented_improvement']:.3f} | "
            f"{100.0 * row['relative_improvement']:.2f}% | "
            f"[{lower:.3f}, {upper:.3f}] | {city_text} | "
            f"macro={safety['macro_noninferior']}, "
            f"all-paired={safety['all_paired_rollouts_noninferior']} |"
        )
    lines.extend(
        [
            "",
            "## Protocol Note",
            "",
            "All policy comparisons are paired on the same scenario and seed. "
            "Scenarios are averaged equally within each city, then cities are "
            "averaged equally. The validity gate checks provenance, matrix "
            "completeness, source identity, SUMO version, tripinfo hashes, "
            "population accounting and zero teleports. Collisions are retained "
            "as audited safety outcomes under paired noninferiority; the validity "
            "gate does not impose a performance or collision threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--offline-authorization", type=Path, required=True)
    parser.add_argument("--joint-freeze-audit", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_json.exists() or args.out_md.exists():
        raise FileExistsError("refusing to overwrite closed-loop aggregate evidence")
    payload = aggregate_closed_loop_results(
        protocol_spec_path=args.protocol_spec,
        offline_authorization_path=args.offline_authorization,
        joint_freeze_audit_path=args.joint_freeze_audit,
        launch_manifest_path=args.launch_manifest,
        result_root=args.result_root,
        expected_sumo_version=args.expected_sumo_version,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    _atomic_json(args.out_json, payload)
    _atomic_text(args.out_md, render_markdown(payload))
    print(
        json.dumps(
            {
                "validity_gate": payload["validity_gate"],
                "macro_primary": {
                    policy: payload["macro_summary"][policy][PRIMARY_METRIC]
                    for policy in POLICIES
                },
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
