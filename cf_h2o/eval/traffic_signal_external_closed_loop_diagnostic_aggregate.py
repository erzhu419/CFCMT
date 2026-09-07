"""Audit and aggregate the post-hoc v43 deployment-mechanism diagnostics."""

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
from cf_h2o.eval.traffic_signal_external_closed_loop_aggregate import (
    AGGREGATE_PROTOCOL as CONFIRMATORY_AGGREGATE_PROTOCOL,
    METRIC_DIRECTIONS,
    PRIMARY_METRIC,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    METHOD_POLICY,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_diagnostic import (
    ANALYSIS_PROTOCOL,
    DIAGNOSTIC_SPECS,
    POLICIES as DIAGNOSTIC_POLICIES,
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import PROTOCOL
from scripts.cluster.launch_tsc_external_closed_loop_diagnostic import (
    LAUNCH_PROTOCOL,
)


AGGREGATE_PROTOCOL = "tsc-v43r39-deployment-mechanism-diagnostic-aggregate-v1"
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_REPLICATES = 10_000
REPORT_POLICIES = (
    "selected_source_prior",
    METHOD_POLICY,
    "cfcmt_selected_spatial_only",
    "cfcmt_selected_direct",
    "rigid_anchor_mpc",
    "rigid_anchor_direct",
    "simulator_only_mpc",
    "simulator_only_direct",
    "h2oplus_style_dense_residual_mpc",
    "h2oplus_style_dense_residual_direct",
    "phase_pressure",
    "max_pressure",
    "fixed_time",
)
COMPARISONS = (
    (METHOD_POLICY, "selected_source_prior"),
    ("cfcmt_selected_spatial_only", "selected_source_prior"),
    ("cfcmt_selected_direct", "selected_source_prior"),
    (METHOD_POLICY, "h2oplus_style_dense_residual_mpc"),
    ("cfcmt_selected_direct", "h2oplus_style_dense_residual_direct"),
    (METHOD_POLICY, "rigid_anchor_mpc"),
    ("cfcmt_selected_direct", "rigid_anchor_direct"),
    (METHOD_POLICY, "phase_pressure"),
    (METHOD_POLICY, "max_pressure"),
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _finite(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value for {label}: {value!r}")
    return number


def _expected_matrix(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], tuple[int, ...], tuple[tuple[str, int, str], ...]]:
    cities = {
        str(city): tuple(str(scenario) for scenario in scenarios)
        for city, scenarios in protocol["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    seeds = tuple(
        int(seed) for seed in protocol["target_protocol"]["closed_loop_seeds"]
    )
    matrix = tuple(
        (scenario, seed, policy)
        for scenarios in cities.values()
        for scenario in scenarios
        for seed in seeds
        for policy in DIAGNOSTIC_POLICIES
    )
    if len(cities) != 2 or len(matrix) != 48 or len(set(matrix)) != 48:
        raise ValueError("diagnostic matrix cardinality changed")
    return cities, seeds, matrix


def _launch_nodes(
    launch: Mapping[str, Any],
    matrix: Sequence[tuple[str, int, str]],
) -> dict[tuple[str, int, str], str]:
    expected = set(matrix)
    nodes: dict[tuple[str, int, str], str] = {}
    prefix = "CFCMT/v43r39/closed-loop-diagnostic-v1/"
    for spec in launch.get("specs", ()):
        signature = str(spec.get("signature", ""))
        if not signature.startswith(prefix):
            raise ValueError(f"unexpected diagnostic signature: {signature}")
        scenario, seed_text, policy = signature[len(prefix) :].split("/", 2)
        key = (scenario, int(seed_text.removeprefix("seed-")), policy)
        if key not in expected or key in nodes:
            raise ValueError(f"duplicate or unexpected diagnostic task: {key}")
        nodes[key] = str(spec["require_node"])
    if set(nodes) != expected:
        raise ValueError("diagnostic launch matrix is incomplete")
    return nodes


def _validate_population(metrics: Mapping[str, Any], *, label: str) -> None:
    if not bool(metrics.get("ok", False)):
        raise ValueError(f"invalid diagnostic rollout: {label}")
    if int(metrics.get("starting_teleports", -1)) != 0 or int(
        metrics.get("ending_teleports", -1)
    ) != 0:
        raise ValueError(f"teleport in diagnostic rollout: {label}")
    departed = int(metrics.get("departed", -1))
    arrived = int(metrics.get("arrived", -1))
    population = int(metrics.get("demand_population_at_horizon", -1))
    tripinfo_count = int(metrics.get("tripinfo_count", -1))
    completed = int(metrics.get("tripinfo_completed_count", -1))
    unfinished = int(metrics.get("tripinfo_unfinished_count", -1))
    if (
        tripinfo_count <= 0
        or tripinfo_count != departed
        or completed != arrived
        or unfinished != departed - arrived
        or not (0 <= arrived <= departed <= population)
    ):
        raise ValueError(f"population accounting mismatch: {label}")


def _validate_deployment_audit(
    *, policy: str, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    spec = DIAGNOSTIC_SPECS[policy]
    deployment = dict(metrics.get("residual_deployment", {}))
    guard = dict(metrics.get("guard_audit", {}))
    if policy == "selected_source_prior":
        if deployment or guard:
            raise ValueError("source-prior diagnostic unexpectedly used residual logic")
        return {}
    if deployment.get("mode") != spec.coordination_mode:
        raise ValueError(f"deployment mode mismatch for {policy}")
    proposed = int(guard.get("proposed_overrides", -1))
    accepted = int(guard.get("accepted_overrides", -1))
    cooldown_rejections = int(guard.get("rejected_cooldown", -1))
    coordination_rejections = int(guard.get("rejected_coordination", -1))
    if proposed <= 0:
        raise ValueError(f"diagnostic exercised no learned overrides: {policy}")
    if spec.coordination_mode == "direct":
        if (
            int(deployment.get("cooldown_intervals", -1)) != 0
            or deployment.get("coordination") != "none"
            or bool(deployment.get("conflict_graph_applied", True))
            or accepted != proposed
            or cooldown_rejections != 0
            or coordination_rejections != 0
        ):
            raise ValueError(f"direct execution invariant failed: {policy}")
    elif spec.coordination_mode == "spatial_only":
        if (
            int(deployment.get("cooldown_intervals", -1)) != 0
            or deployment.get("coordination")
            != "greedy_priority_independent_set"
            or not bool(deployment.get("conflict_graph_applied", False))
            or cooldown_rejections != 0
            or accepted + coordination_rejections != proposed
        ):
            raise ValueError(f"spatial-only execution invariant failed: {policy}")
    return {
        "mode": spec.coordination_mode,
        "decisions": int(guard["decisions"]),
        "prior_agreements": int(guard["prior_agreements"]),
        "proposed_overrides": proposed,
        "accepted_overrides": accepted,
        "rejected_cooldown": cooldown_rejections,
        "rejected_coordination": coordination_rejections,
    }


def _validate_result(
    *,
    path: Path,
    key: tuple[str, int, str],
    city: str,
    node: str,
    launch: Mapping[str, Any],
    joint: Mapping[str, Any],
    authorization: Mapping[str, Any],
    protocol_sha256: str,
    expected_sumo_version: str,
) -> dict[str, Any]:
    scenario, seed, policy = key
    result = _read_json(path)
    spec = DIAGNOSTIC_SPECS[policy]
    expected_hashes = {
        "protocol_spec_sha256": protocol_sha256,
        "offline_authorization_sha256": launch["offline_authorization_sha256"],
        "joint_freeze_audit_sha256": launch["joint_freeze_audit_sha256"],
        "external_manifest_sha256": launch["external_manifest_sha256"],
        "conversion_manifest_sha256": launch["conversion_manifest_sha256"],
        "conversion_tree_sha256": launch["conversion_tree_sha256"],
    }
    if (
        result.get("protocol") != RESULT_PROTOCOL
        or result.get("analysis_protocol") != ANALYSIS_PROTOCOL
        or result.get("scenario") != scenario
        or int(result.get("seed", -1)) != seed
        or result.get("policy") != policy
        or result.get("source_policy") != spec.source_policy
        or result.get("city") != city
        or result.get("post_hoc_diagnostic") != spec.to_dict()
        or any(result.get(name) != value for name, value in expected_hashes.items())
    ):
        raise ValueError(f"diagnostic identity or provenance mismatch: {path}")
    runtime = dict(result.get("runtime", {}))
    if (
        result.get("hostname") != node
        or runtime.get("hostname") != node
        or runtime.get("source_tree_sha256") != launch["source_tree_sha256"]
        or result.get("sumo_version") != expected_sumo_version
        or runtime.get("libsumo_version") != expected_sumo_version
        or result.get("model_artifacts") != joint["cities"][city]
    ):
        raise ValueError(f"diagnostic physical runtime mismatch: {path}")
    if spec.source_policy == METHOD_POLICY:
        expected_candidate = authorization["city_results"][city][
            "selected_candidate"
        ]
        blend = dict(result.get("anchored_blend_audit") or {})
        if (
            result.get("selected_candidate") != expected_candidate
            or int(blend.get("prediction_calls", 0)) <= 0
        ):
            raise ValueError(f"diagnostic did not execute frozen CFCMT blend: {path}")
    elif result.get("selected_candidate") is not None or result.get(
        "anchored_blend_audit"
    ) is not None:
        raise ValueError(f"non-CFCMT diagnostic carries method-only state: {path}")

    tripinfo = path.parent / "tripinfo.xml"
    evidence = dict(result.get("tripinfo_evidence", {}))
    if (
        not tripinfo.is_file()
        or evidence.get("filename") != tripinfo.name
        or int(evidence.get("size_bytes", -1)) != tripinfo.stat().st_size
        or evidence.get("sha256") != _sha256(tripinfo)
    ):
        raise ValueError(f"diagnostic tripinfo evidence mismatch: {path}")
    metrics = dict(result.get("metrics", {}))
    _validate_population(metrics, label=str(path))
    finite_metrics = {
        metric: _finite(metrics.get(metric), label=f"{key}/{metric}")
        for metric in METRIC_DIRECTIONS
    }
    return {
        "city": city,
        "scenario": scenario,
        "seed": seed,
        "policy": policy,
        "node": node,
        "result_sha256": _sha256(path),
        "tripinfo_sha256": evidence["sha256"],
        "metrics": finite_metrics,
        "deployment_audit": _validate_deployment_audit(
            policy=policy, metrics=metrics
        ),
    }


def _audit_confirmatory_deployment(
    *, aggregate: Mapping[str, Any], result_root: Path
) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for record in aggregate["records"]:
        policy = str(record["policy"])
        path = (
            result_root
            / str(record["scenario"])
            / f"seed_{int(record['seed'])}"
            / policy
            / "result.json"
        )
        if not path.is_file() or _sha256(path) != record["result_sha256"]:
            raise ValueError(f"confirmatory raw result changed: {path}")
        result = _read_json(path)
        deployment = dict(result["metrics"].get("residual_deployment", {}))
        guard = dict(result["metrics"].get("guard_audit", {}))
        if not deployment:
            continue
        proposed = int(guard["proposed_overrides"])
        accepted = int(guard["accepted_overrides"])
        rejected_cooldown = int(guard["rejected_cooldown"])
        rejected_coordination = int(guard["rejected_coordination"])
        if (
            deployment.get("estimand")
            != "partial_interference_first_action_then_prior_rollout"
            or int(deployment.get("cooldown_intervals", -1)) <= 0
            or proposed
            != accepted + rejected_cooldown + rejected_coordination
        ):
            raise ValueError(f"confirmatory sparse-deployment audit failed: {path}")
        row = totals.setdefault(
            policy,
            {
                "mode": "sparse",
                "decisions": 0,
                "prior_agreements": 0,
                "proposed_overrides": 0,
                "accepted_overrides": 0,
                "rejected_cooldown": 0,
                "rejected_coordination": 0,
            },
        )
        for key in tuple(row)[1:]:
            row[key] += int(guard[key])
    return totals


def _summaries(
    records: Sequence[Mapping[str, Any]],
    cities: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario_summary: dict[str, Any] = {}
    city_summary: dict[str, Any] = {}
    for city, scenarios in cities.items():
        city_summary[city] = {}
        for scenario in scenarios:
            scenario_summary[scenario] = {}
            for policy in REPORT_POLICIES:
                rows = [
                    row
                    for row in records
                    if row["scenario"] == scenario and row["policy"] == policy
                ]
                if len(rows) != 2:
                    raise ValueError(f"incomplete reporting cells: {scenario}/{policy}")
                scenario_summary[scenario][policy] = {
                    metric: float(np.mean([row["metrics"][metric] for row in rows]))
                    for metric in METRIC_DIRECTIONS
                }
        for policy in REPORT_POLICIES:
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
    macro = {
        policy: {
            metric: float(
                np.mean([city_summary[city][policy][metric] for city in cities])
            )
            for metric in METRIC_DIRECTIONS
        }
        for policy in REPORT_POLICIES
    }
    return city_summary, macro


def _paired_bootstrap(
    records: Sequence[Mapping[str, Any]],
    cities: Mapping[str, Sequence[str]],
    *,
    method: str,
    baseline: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    lookup = {
        (row["city"], row["scenario"], int(row["seed"]), row["policy"]): float(
            row["metrics"][PRIMARY_METRIC]
        )
        for row in records
    }
    city_names = tuple(cities)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_city_values = []
        for city_index in rng.integers(0, len(city_names), size=len(city_names)):
            city = city_names[int(city_index)]
            scenarios = tuple(cities[city])
            sampled_scenario_values = []
            for scenario_index in rng.integers(0, len(scenarios), size=len(scenarios)):
                scenario = scenarios[int(scenario_index)]
                seeds = tuple(
                    sorted(
                        {
                            int(row["seed"])
                            for row in records
                            if row["scenario"] == scenario
                        }
                    )
                )
                paired = []
                for seed_index in rng.integers(0, len(seeds), size=len(seeds)):
                    rollout_seed = seeds[int(seed_index)]
                    paired.append(
                        lookup[(city, scenario, rollout_seed, baseline)]
                        - lookup[(city, scenario, rollout_seed, method)]
                    )
                sampled_scenario_values.append(float(np.mean(paired)))
            sampled_city_values.append(float(np.mean(sampled_scenario_values)))
        draws[index] = float(np.mean(sampled_city_values))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "method": method,
        "baseline": baseline,
        "positive_favors_method": True,
        "macro_waiting_gain_sec": float(np.mean(draws)),
        "gain_95pct_descriptive_interval": [float(lower), float(upper)],
        "bootstrap_probability_positive": float(np.mean(draws > 0.0)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def aggregate_diagnostics(
    *,
    protocol_spec_path: Path,
    offline_authorization_path: Path,
    joint_freeze_audit_path: Path,
    launch_manifest_path: Path,
    diagnostic_result_root: Path,
    confirmatory_aggregate_path: Path,
    confirmatory_result_root: Path,
    expected_sumo_version: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    protocol = _read_json(protocol_spec_path)
    authorization = _read_json(offline_authorization_path)
    joint = _read_json(joint_freeze_audit_path)
    launch = _read_json(launch_manifest_path)
    confirmatory = _read_json(confirmatory_aggregate_path)
    if (
        protocol.get("protocol") != PROTOCOL
        or launch.get("protocol") != LAUNCH_PROTOCOL
        or launch.get("analysis_protocol") != ANALYSIS_PROTOCOL
        or launch.get("result_protocol") != RESULT_PROTOCOL
        or not bool(launch.get("submitted", False))
        or int(launch.get("task_count", -1)) != 48
        or joint.get("status") != "PASS"
        or confirmatory.get("protocol") != CONFIRMATORY_AGGREGATE_PROTOCOL
        or not bool(confirmatory.get("validity_gate", {}).get("passed", False))
        or int(bootstrap_replicates) <= 0
    ):
        raise ValueError("diagnostic aggregate lacks complete frozen evidence")
    if (
        _sha256(offline_authorization_path)
        != launch["offline_authorization_sha256"]
        or _sha256(joint_freeze_audit_path) != launch["joint_freeze_audit_sha256"]
        or authorization.get("joint_freeze_audit_sha256")
        != launch["joint_freeze_audit_sha256"]
    ):
        raise ValueError("diagnostic authorization hashes changed")

    cities, seeds, matrix = _expected_matrix(protocol)
    scenario_city = {
        scenario: city for city, scenarios in cities.items() for scenario in scenarios
    }
    nodes = _launch_nodes(launch, matrix)
    root = Path(diagnostic_result_root)
    discovered_results = tuple(sorted(root.glob("**/result.json")))
    discovered_tripinfos = tuple(sorted(root.glob("**/tripinfo.xml")))
    temporary = tuple(sorted(root.glob("**/*.tmp-*")))
    if len(discovered_results) != 48 or len(discovered_tripinfos) != 48 or temporary:
        raise ValueError("diagnostic evidence tree is incomplete or contaminated")
    protocol_sha256 = _sha256(protocol_spec_path)
    diagnostic_records = []
    expected_paths = set()
    for key in matrix:
        scenario, seed, policy = key
        path = root / scenario / f"seed_{seed}" / policy / "result.json"
        expected_paths.add(path.resolve())
        diagnostic_records.append(
            _validate_result(
                path=path,
                key=key,
                city=scenario_city[scenario],
                node=nodes[key],
                launch=launch,
                joint=joint,
                authorization=authorization,
                protocol_sha256=protocol_sha256,
                expected_sumo_version=expected_sumo_version,
            )
        )
    if {path.resolve() for path in discovered_results} != expected_paths:
        raise ValueError("diagnostic evidence tree contains unexpected results")

    confirmatory_records = list(confirmatory["records"])
    if len(confirmatory_records) != 56:
        raise ValueError("confirmatory aggregate record count changed")
    if any(
        confirmatory["provenance"].get(key) != launch.get(key)
        for key in (
            "offline_authorization_sha256",
            "joint_freeze_audit_sha256",
            "external_manifest_sha256",
            "conversion_manifest_sha256",
            "conversion_tree_sha256",
        )
    ):
        raise ValueError("diagnostic and confirmatory inputs differ")
    sparse_audits = _audit_confirmatory_deployment(
        aggregate=confirmatory,
        result_root=confirmatory_result_root,
    )
    combined = [*confirmatory_records, *diagnostic_records]
    city_summary, macro = _summaries(combined, cities)
    comparisons = {
        f"{method}_vs_{baseline}": _paired_bootstrap(
            combined,
            cities,
            method=method,
            baseline=baseline,
            replicates=int(bootstrap_replicates),
            seed=BOOTSTRAP_SEED + index,
        )
        for index, (method, baseline) in enumerate(COMPARISONS)
    }
    diagnostic_audits: dict[str, dict[str, Any]] = {}
    for row in diagnostic_records:
        if not row["deployment_audit"]:
            continue
        policy = row["policy"]
        total = diagnostic_audits.setdefault(
            policy,
            {
                "mode": row["deployment_audit"]["mode"],
                "decisions": 0,
                "prior_agreements": 0,
                "proposed_overrides": 0,
                "accepted_overrides": 0,
                "rejected_cooldown": 0,
                "rejected_coordination": 0,
            },
        )
        for key in tuple(total)[1:]:
            total[key] += int(row["deployment_audit"][key])
    deployment_audit = {**sparse_audits, **diagnostic_audits}
    for row in deployment_audit.values():
        row["accepted_fraction_of_proposals"] = float(
            row["accepted_overrides"] / max(row["proposed_overrides"], 1)
        )

    return {
        "protocol": AGGREGATE_PROTOCOL,
        "analysis_status": "post_hoc_explanatory_not_preregistered_confirmation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validity_gate": {
            "passed": True,
            "decision": "complete_deployment_mechanism_diagnostic_reporting",
            "performance_threshold_used": False,
            "diagnostic_rollouts": len(diagnostic_records),
            "confirmatory_rollouts_reused": len(confirmatory_records),
            "physical_hostname_verified": True,
            "direct_execution_invariants_verified": True,
            "spatial_only_execution_invariants_verified": True,
            "teleports": 0,
        },
        "provenance": {
            "protocol_spec_sha256": protocol_sha256,
            "offline_authorization_sha256": _sha256(offline_authorization_path),
            "joint_freeze_audit_sha256": _sha256(joint_freeze_audit_path),
            "launch_manifest_sha256": _sha256(launch_manifest_path),
            "confirmatory_aggregate_sha256": _sha256(
                confirmatory_aggregate_path
            ),
            "diagnostic_source_tree_sha256": launch["source_tree_sha256"],
            "confirmatory_source_tree_sha256": confirmatory["provenance"][
                "source_tree_sha256"
            ],
            "sumo_version": expected_sumo_version,
        },
        "matrix": {
            "cities": {city: list(scenarios) for city, scenarios in cities.items()},
            "seeds": list(seeds),
            "diagnostic_policies": list(DIAGNOSTIC_POLICIES),
            "report_policies": list(REPORT_POLICIES),
        },
        "deployment_interpretation": {
            "confirmatory_method": (
                "source-selected generalized-pressure prior with learned causal "
                "action-contrast overrides, conflict-graph coordination, and "
                "prediction-horizon cooldown"
            ),
            "spatial_only_diagnostic": (
                "same frozen CFCMT scores with conflict-graph coordination but "
                "without temporal cooldown"
            ),
            "direct_diagnostic": (
                "same frozen score model with every eligible per-signal override "
                "executed and no cooldown or cross-signal coordination"
            ),
        },
        "deployment_audit": deployment_audit,
        "city_summary": city_summary,
        "macro_summary": macro,
        "paired_descriptive_comparisons": comparisons,
        "diagnostic_records": diagnostic_records,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    macro = payload["macro_summary"]
    lines = [
        "# V43 Deployment-Mechanism Diagnostic",
        "",
        "Status: **post-hoc explanatory analysis (not preregistered confirmation)**  ",
        f"Validity gate: **{'PASS' if payload['validity_gate']['passed'] else 'FAIL'}**  ",
        f"New diagnostic rollouts: {payload['validity_gate']['diagnostic_rollouts']}  ",
        "",
        "The frozen confirmatory controller is a pressure-anchored sparse residual "
        "controller. `direct` executes every eligible learned override; `spatial_only` "
        "retains only conflict-graph coordination.",
        "",
        "## Macro Results",
        "",
        "| Policy | Wait (s) | Completed wait (s) | System veh-h | Completion | Incidents |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in REPORT_POLICIES:
        row = macro[policy]
        lines.append(
            f"| {policy} | {row[PRIMARY_METRIC]:.3f} | "
            f"{row['mean_completed_tripinfo_waiting_time']:.3f} | "
            f"{row['system_vehicle_hours']:.3f} | "
            f"{row['completion_ratio']:.4f} | "
            f"{row['collision_incidents']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Descriptive Comparisons",
            "",
            "Positive waiting-time gains favor the named method. Intervals are "
            "descriptive because there are only two independent target cities.",
            "",
            "| Comparison | Gain (s) | 95% interval | P(gain > 0) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in payload["paired_descriptive_comparisons"].items():
        interval = row["gain_95pct_descriptive_interval"]
        lines.append(
            f"| {name} | {row['macro_waiting_gain_sec']:.3f} | "
            f"[{interval[0]:.3f}, {interval[1]:.3f}] | "
            f"{row['bootstrap_probability_positive']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Deployment Audit",
            "",
            "| Policy | Mode | Proposals | Accepted | Acceptance | Cooldown rejects | Spatial rejects |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for policy, row in payload["deployment_audit"].items():
        lines.append(
            f"| {policy} | {row['mode']} | {row['proposed_overrides']} | "
            f"{row['accepted_overrides']} | "
            f"{row['accepted_fraction_of_proposals']:.4f} | "
            f"{row['rejected_cooldown']} | {row['rejected_coordination']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--offline-authorization", type=Path, required=True)
    parser.add_argument("--joint-freeze-audit", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-result-root", type=Path, required=True)
    parser.add_argument("--confirmatory-aggregate", type=Path, required=True)
    parser.add_argument("--confirmatory-result-root", type=Path, required=True)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = aggregate_diagnostics(
        protocol_spec_path=args.protocol_spec,
        offline_authorization_path=args.offline_authorization,
        joint_freeze_audit_path=args.joint_freeze_audit,
        launch_manifest_path=args.launch_manifest,
        diagnostic_result_root=args.diagnostic_result_root,
        confirmatory_aggregate_path=args.confirmatory_aggregate,
        confirmatory_result_root=args.confirmatory_result_root,
        expected_sumo_version=args.expected_sumo_version,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    _atomic_json(args.out, payload)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "diagnostic_rollouts": payload["validity_gate"][
                    "diagnostic_rollouts"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
