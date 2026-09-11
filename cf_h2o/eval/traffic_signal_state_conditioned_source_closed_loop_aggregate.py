"""Aggregate the frozen V150L closed-loop development matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import (
    POLICY_ARMS,
    RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


AGGREGATE_PROTOCOL = "tsc-v150l-state-conditioned-source-closed-loop-aggregate-v4"
CONFIG_PROTOCOL = "tsc-v150l-state-conditioned-source-closed-loop-development-v3"
OUTCOME_FIELDS = (
    "mean_tripinfo_waiting_time",
    "p90_tripinfo_waiting_time",
    "mean_queue",
    "p90_queue",
    "halted_vehicle_seconds",
    "system_vehicle_seconds",
    "departed",
    "arrived",
    "completion_ratio",
    "starting_teleports",
    "ending_teleports",
    "collision_events",
    "collision_incidents",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def expected_identities(
    config: Mapping[str, Any], *, stage: str
) -> tuple[tuple[str, str, int, str], ...]:
    if config.get("protocol") != CONFIG_PROTOCOL:
        raise ValueError("V150L aggregate configuration changed")
    if stage == "smoke":
        scenarios = dict(config["smoke"]["city_scenarios"])
        seeds = tuple(int(value) for value in config["smoke"]["seeds"])
        expected_size = int(config["smoke"]["matrix_size"])
    elif stage == "full":
        scenarios = dict(config["city_scenarios"])
        seeds = tuple(int(value) for value in config["full"]["seeds"])
        expected_size = int(config["full"]["matrix_size"])
    else:
        raise ValueError(f"unknown V150L aggregate stage: {stage!r}")
    identities = tuple(
        (str(city), str(scenario), seed, str(arm))
        for city, values in scenarios.items()
        for scenario in values
        for seed in seeds
        for arm in POLICY_ARMS
    )
    if len(identities) != expected_size:
        raise ValueError("V150L aggregate matrix size changed")
    return identities


def _identity(result: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(result.get("city")),
        str(result.get("scenario")),
        int(result.get("seed", -1)),
        str(result.get("arm")),
    )


def _cost(result: Mapping[str, Any], metric: str) -> float:
    value = float(result.get("metrics", {}).get(metric, float("nan")))
    if not np.isfinite(value):
        raise ValueError(f"V150L result has invalid {metric}: {_identity(result)}")
    return value


def _outcomes_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_metrics = dict(left.get("metrics", {}))
    right_metrics = dict(right.get("metrics", {}))
    return all(left_metrics.get(name) == right_metrics.get(name) for name in OUTCOME_FIELDS)


def aggregate_results(
    results: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    rows = tuple(dict(value) for value in results)
    expected = set(expected_identities(config, stage=stage))
    by_identity = {_identity(row): row for row in rows}
    if (
        len(rows) != len(expected)
        or set(by_identity) != expected
        or any(row.get("protocol") != RESULT_PROTOCOL for row in rows)
    ):
        raise ValueError("V150L aggregate requires the complete frozen matrix")
    primary = str(config["primary_metric"])
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        city, scenario, seed, arm = _identity(row)
        grouped.setdefault((city, scenario, seed), {})[arm] = row

    admitted = set(str(value) for value in config["offline_admitted_source_cities"])
    rejected = set(str(value) for value in config["city_scenarios"]) - admitted
    exact_fallback_failures: list[dict[str, Any]] = []
    for (city, scenario, seed), arms in grouped.items():
        if city not in rejected:
            continue
        rigid = arms["rigid_target_only"]
        for arm in (
            "selected_source_corrected_rigid",
            "same_capacity_placebo_corrected_rigid",
        ):
            candidate = arms[arm]
            changed = int(
                candidate.get("originator_diagnostics", {}).get(
                    "source_changed_rigid_action_count", -1
                )
            )
            if changed != 0 or not _outcomes_equal(rigid, candidate):
                exact_fallback_failures.append(
                    {
                        "city": city,
                        "scenario": scenario,
                        "seed": seed,
                        "arm": arm,
                        "source_changed_rigid_action_count": changed,
                    }
                )

    observed_collision_rows = [
        {
            "identity": list(_identity(row)),
            "collision_events": int(row["metrics"]["collision_events"]),
            "collision_incidents": int(row["metrics"]["collision_incidents"]),
        }
        for row in rows
        if int(row["metrics"]["collision_events"]) != 0
    ]
    teleport_failures = [
        {
            "identity": list(_identity(row)),
            "starting_teleports": int(row["metrics"]["starting_teleports"]),
            "ending_teleports": int(row["metrics"]["ending_teleports"]),
        }
        for row in rows
        if int(row["metrics"]["starting_teleports"]) != 0
        or int(row["metrics"]["ending_teleports"]) != 0
    ]
    paired_collision_audit = []
    for (city, scenario, seed), arms in sorted(grouped.items()):
        selected_incidents = int(
            arms["selected_source_corrected_rigid"]["metrics"][
                "collision_incidents"
            ]
        )
        rigid_incidents = int(
            arms["rigid_target_only"]["metrics"]["collision_incidents"]
        )
        phase_incidents = int(
            arms["phase_pressure"]["metrics"]["collision_incidents"]
        )
        paired_collision_audit.append(
            {
                "city": city,
                "scenario": scenario,
                "seed": seed,
                "selected_source_collision_incidents": selected_incidents,
                "rigid_collision_incidents": rigid_incidents,
                "phase_pressure_collision_incidents": phase_incidents,
                "noninferior_to_rigid": selected_incidents <= rigid_incidents,
                "noninferior_to_phase_pressure": selected_incidents
                <= phase_incidents,
                "passed": selected_incidents <= rigid_incidents
                and selected_incidents <= phase_incidents,
            }
        )
    paired_collision_failures = [
        row for row in paired_collision_audit if not bool(row["passed"])
    ]
    originator_failures = [
        list(_identity(row))
        for row in rows
        if row["arm"] != "phase_pressure"
        and int(
            row.get("originator_diagnostics", {}).get("decision_count", 0)
        )
        <= 0
    ]

    city_summaries: dict[str, dict[str, Any]] = {}
    for city in config["city_scenarios"]:
        city_groups = [key for key in grouped if key[0] == city]
        if not city_groups:
            continue
        arm_means = {
            arm: float(
                np.mean([_cost(grouped[key][arm], primary) for key in city_groups])
            )
            for arm in POLICY_ARMS
        }
        source_minus_rigid = (
            arm_means["selected_source_corrected_rigid"]
            - arm_means["rigid_target_only"]
        )
        source_minus_placebo = (
            arm_means["selected_source_corrected_rigid"]
            - arm_means["same_capacity_placebo_corrected_rigid"]
        )
        source_changes = sum(
            int(
                grouped[key]["selected_source_corrected_rigid"]
                .get("originator_diagnostics", {})
                .get("source_changed_rigid_action_count", 0)
            )
            for key in city_groups
        )
        city_summaries[city] = {
            "offline_source_admitted": city in admitted,
            "scenario_seed_unit_count": len(city_groups),
            "arm_mean_costs": arm_means,
            "source_minus_rigid": source_minus_rigid,
            "source_minus_matched_placebo": source_minus_placebo,
            "source_relative_change_vs_rigid": float(
                source_minus_rigid
                / max(abs(arm_means["rigid_target_only"]), 1e-12)
            ),
            "source_changed_rigid_action_count": source_changes,
        }

    admitted_rows = [city_summaries[city] for city in sorted(admitted) if city in city_summaries]
    source_differences = np.asarray(
        [row["source_minus_rigid"] for row in admitted_rows], dtype=float
    )
    placebo_differences = np.asarray(
        [row["source_minus_matched_placebo"] for row in admitted_rows],
        dtype=float,
    )
    nondegrading = int(np.count_nonzero(source_differences <= 0.0))
    source_action_changes = int(
        sum(row["source_changed_rigid_action_count"] for row in admitted_rows)
    )
    operational_checks = {
        "all_rollouts_complete": len(rows) == len(expected),
        "zero_teleports_all_arms": not teleport_failures,
        "selected_source_collision_incidents_noninferior_to_rigid_and_phase": (
            not paired_collision_failures
        ),
        "all_source_arms_entered_originator": not originator_failures,
        "exact_rigid_fallback_for_offline_rejected_cities": (
            not exact_fallback_failures
        ),
    }
    gate = dict(config["development_gate"])
    maximum_regression = float(gate["maximum_admitted_city_relative_regression"])
    performance_checks = {
        "minimum_two_admitted_cities_nondegrading": nondegrading
        >= int(gate["minimum_admitted_cities_with_nondegrading_source"]),
        "no_admitted_city_regresses_over_one_percent": all(
            row["source_relative_change_vs_rigid"] <= maximum_regression
            for row in admitted_rows
        ),
        "mean_source_gain_vs_rigid": bool(
            source_differences.size and float(np.mean(source_differences)) < 0.0
        ),
        "mean_source_gain_vs_matched_placebo": bool(
            placebo_differences.size and float(np.mean(placebo_differences)) < 0.0
        ),
        "nonzero_source_action_changes": source_action_changes > 0,
    }
    operational_passed = all(operational_checks.values())
    passed = operational_passed and all(performance_checks.values())
    if stage == "smoke":
        decision = (
            "authorize_v150l_full_development_matrix"
            if passed
            else "reject_v150l_full_matrix_after_smoke"
        )
    else:
        decision = (
            "authorize_fresh_city_source_contribution_confirmation"
            if passed
            else "retain_rigid_target_adaptation_and_reject_closed_loop_source_claim"
        )
    return {
        "protocol": AGGREGATE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": f"seven-city-{stage}-development-aggregate",
        "stage": stage,
        "rollout_count": len(rows),
        "scenario_seed_unit_count": len(grouped),
        "primary_metric": primary,
        "inference_unit": "city",
        "offline_admitted_source_cities": sorted(admitted),
        "city_summaries": city_summaries,
        "admitted_city_source_minus_rigid_mean": (
            float(np.mean(source_differences)) if source_differences.size else None
        ),
        "admitted_city_source_minus_matched_placebo_mean": (
            float(np.mean(placebo_differences))
            if placebo_differences.size
            else None
        ),
        "admitted_city_nondegrading_count": nondegrading,
        "source_changed_rigid_action_count": source_action_changes,
        "observed_collision_rows": observed_collision_rows,
        "teleport_failures": teleport_failures,
        "paired_collision_audit": paired_collision_audit,
        "paired_collision_failures": paired_collision_failures,
        "originator_failures": originator_failures,
        "exact_fallback_failures": exact_fallback_failures,
        "development_gate": {
            "operational_checks": operational_checks,
            "operational_passed": operational_passed,
            "performance_checks": performance_checks,
            "assessment_basis": (
                "post_smoke_performance_adjudication_using_existing_development_thresholds"
                if stage == "smoke"
                else "frozen_full_development_gate"
            ),
            "passed": passed,
            "decision": decision,
        },
        "claim_boundary": (
            "V150L smoke performance adjudication uses the existing development "
            "thresholds as a stop screen; the original operational-only smoke "
            "result remains historical evidence. A smoke pass can authorize the "
            "full development matrix. Only a full-stage pass can authorize "
            "separate fresh-city confirmation. Neither is independent external evidence."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150L aggregate: {args.out}")
    result_paths = sorted(Path(args.result_root).rglob("result.json"))
    result = aggregate_results(
        [_read_json(path) for path in result_paths],
        config=_read_json(args.config),
        stage=str(args.stage),
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": (
                    "PASS" if result["development_gate"]["passed"] else "REJECT"
                ),
                "stage": result["stage"],
                "rollout_count": result["rollout_count"],
                "decision": result["development_gate"]["decision"],
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
