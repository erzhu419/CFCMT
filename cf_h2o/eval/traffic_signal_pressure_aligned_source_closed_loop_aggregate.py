"""Aggregate V150M pressure-aligned closed-loop development rollouts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_pressure_aligned_source_closed_loop import (
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import (
    POLICY_ARMS,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop_aggregate import (
    OUTCOME_FIELDS,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
)


AGGREGATE_PROTOCOL = "tsc-v150m-pressure-aligned-source-closed-loop-aggregate-v1"
CONFIG_PROTOCOL = "tsc-v150m-pressure-aligned-source-closed-loop-development-v1"
SOURCE_ARM = "selected_source_corrected_rigid"
PLACEBO_ARM = "same_capacity_placebo_corrected_rigid"
RIGID_ARM = "rigid_target_only"
PHASE_ARM = "phase_pressure"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def expected_identities(
    config: Mapping[str, Any], *, stage: str
) -> tuple[tuple[str, str, int, str], ...]:
    if config.get("protocol") != CONFIG_PROTOCOL:
        raise ValueError("V150M aggregate configuration changed")
    if stage == "smoke":
        scenarios = dict(config["smoke"]["city_scenarios"])
        seeds = tuple(int(value) for value in config["smoke"]["seeds"])
        expected_size = int(config["smoke"]["matrix_size"])
    elif stage == "full":
        scenarios = dict(config["city_scenarios"])
        seeds = tuple(int(value) for value in config["full"]["seeds"])
        expected_size = int(config["full"]["matrix_size"])
    else:
        raise ValueError(f"unknown V150M aggregate stage: {stage!r}")
    identities = tuple(
        (str(city), str(scenario), seed, str(arm))
        for city, values in scenarios.items()
        for scenario in values
        for seed in seeds
        for arm in POLICY_ARMS
    )
    if len(identities) != expected_size or len(identities) != len(set(identities)):
        raise ValueError("V150M aggregate matrix size changed")
    return identities


def _identity(result: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(result.get("city")),
        str(result.get("scenario")),
        int(result.get("seed", -1)),
        str(result.get("arm")),
    )


def _metric(result: Mapping[str, Any], name: str) -> float:
    value = float(result.get("metrics", {}).get(name, float("nan")))
    if not np.isfinite(value):
        raise ValueError(f"V150M result has invalid {name}: {_identity(result)}")
    return value


def _relative(candidate: float, baseline: float) -> float:
    return float((candidate - baseline) / max(abs(baseline), 1e-12))


def _outcomes_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_metrics = dict(left.get("metrics", {}))
    right_metrics = dict(right.get("metrics", {}))
    return all(
        left_metrics.get(name) == right_metrics.get(name)
        for name in OUTCOME_FIELDS
    )


def _bootstrap_mean_ci(values: Sequence[float]) -> list[float] | None:
    samples = np.asarray(tuple(values), dtype=float)
    if samples.size < 2:
        return None
    generator = np.random.default_rng(15013)
    indices = generator.integers(0, samples.size, size=(10_000, samples.size))
    means = np.mean(samples[indices], axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


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
        raise ValueError("V150M aggregate requires the complete frozen matrix")
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        city, scenario, seed, arm = _identity(row)
        grouped.setdefault((city, scenario, seed), {})[arm] = row

    admitted = set(str(value) for value in config["offline_admitted_source_cities"])
    rejected = set(str(value) for value in config["city_scenarios"]) - admitted
    fallback_failures = []
    for (city, scenario, seed), arms in grouped.items():
        if city not in rejected:
            continue
        rigid = arms[RIGID_ARM]
        for arm in (SOURCE_ARM, PLACEBO_ARM):
            candidate = arms[arm]
            changed = int(
                candidate.get("originator_diagnostics", {}).get(
                    "source_changed_rigid_action_count", -1
                )
            )
            if changed != 0 or not _outcomes_equal(rigid, candidate):
                fallback_failures.append(
                    {
                        "city": city,
                        "scenario": scenario,
                        "seed": seed,
                        "arm": arm,
                        "source_changed_rigid_action_count": changed,
                    }
                )

    constraint_failures = []
    originator_failures = []
    for row in rows:
        if row["arm"] == PHASE_ARM:
            continue
        diagnostics = dict(row.get("originator_diagnostics", {}))
        changed = int(diagnostics.get("source_changed_rigid_action_count", -1))
        changed_to_reference = int(
            diagnostics.get("source_changed_to_reference_action_count", -1)
        )
        changed_off_reference = int(
            diagnostics.get("source_changed_off_reference_action_count", -1)
        )
        if int(diagnostics.get("decision_count", 0)) <= 0:
            originator_failures.append(list(_identity(row)))
        if (
            diagnostics.get("candidate_score_constraint")
            != PRESSURE_ALIGNED_CANDIDATE_PROTOCOL
            or changed_off_reference != 0
            or changed != changed_to_reference
        ):
            constraint_failures.append(
                {
                    "identity": list(_identity(row)),
                    "changed": changed,
                    "changed_to_reference": changed_to_reference,
                    "changed_off_reference": changed_off_reference,
                    "constraint": diagnostics.get("candidate_score_constraint"),
                }
            )

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
    observed_collision_rows = [
        {
            "identity": list(_identity(row)),
            "collision_events": int(row["metrics"]["collision_events"]),
            "collision_incidents": int(row["metrics"]["collision_incidents"]),
        }
        for row in rows
        if int(row["metrics"]["collision_events"]) != 0
    ]
    paired_collision_audit = []
    for (city, scenario, seed), arms in sorted(grouped.items()):
        rigid = int(arms[RIGID_ARM]["metrics"]["collision_incidents"])
        phase = int(arms[PHASE_ARM]["metrics"]["collision_incidents"])
        for arm in (SOURCE_ARM, PLACEBO_ARM):
            candidate = int(arms[arm]["metrics"]["collision_incidents"])
            paired_collision_audit.append(
                {
                    "city": city,
                    "scenario": scenario,
                    "seed": seed,
                    "arm": arm,
                    "candidate_collision_incidents": candidate,
                    "rigid_collision_incidents": rigid,
                    "phase_pressure_collision_incidents": phase,
                    "passed": candidate <= rigid and candidate <= phase,
                }
            )
    collision_failures = [
        row for row in paired_collision_audit if not bool(row["passed"])
    ]

    primary = str(config["primary_metric"])
    city_summaries: dict[str, dict[str, Any]] = {}
    for city in config["city_scenarios"]:
        keys = [key for key in grouped if key[0] == city]
        if not keys:
            continue
        arm_means = {
            arm: {
                name: float(np.mean([_metric(grouped[key][arm], name) for key in keys]))
                for name in (
                    primary,
                    "p90_tripinfo_waiting_time",
                    "completion_ratio",
                )
            }
            for arm in POLICY_ARMS
        }
        primary_vs_rigid = [
            _relative(
                _metric(grouped[key][SOURCE_ARM], primary),
                _metric(grouped[key][RIGID_ARM], primary),
            )
            for key in keys
        ]
        primary_vs_placebo = [
            _relative(
                _metric(grouped[key][SOURCE_ARM], primary),
                _metric(grouped[key][PLACEBO_ARM], primary),
            )
            for key in keys
        ]
        p90_vs_rigid = [
            _relative(
                _metric(grouped[key][SOURCE_ARM], "p90_tripinfo_waiting_time"),
                _metric(grouped[key][RIGID_ARM], "p90_tripinfo_waiting_time"),
            )
            for key in keys
        ]
        p90_vs_placebo = [
            _relative(
                _metric(grouped[key][SOURCE_ARM], "p90_tripinfo_waiting_time"),
                _metric(grouped[key][PLACEBO_ARM], "p90_tripinfo_waiting_time"),
            )
            for key in keys
        ]
        completion_vs_rigid = [
            _metric(grouped[key][SOURCE_ARM], "completion_ratio")
            - _metric(grouped[key][RIGID_ARM], "completion_ratio")
            for key in keys
        ]
        completion_vs_placebo = [
            _metric(grouped[key][SOURCE_ARM], "completion_ratio")
            - _metric(grouped[key][PLACEBO_ARM], "completion_ratio")
            for key in keys
        ]
        source_changes = sum(
            int(
                grouped[key][SOURCE_ARM]
                .get("originator_diagnostics", {})
                .get("source_changed_rigid_action_count", 0)
            )
            for key in keys
        )
        city_summaries[str(city)] = {
            "offline_source_admitted": city in admitted,
            "scenario_seed_unit_count": len(keys),
            "arm_metric_means": arm_means,
            "primary_relative_change_vs_rigid": float(np.mean(primary_vs_rigid)),
            "primary_relative_change_vs_matched_placebo": float(
                np.mean(primary_vs_placebo)
            ),
            "p90_relative_change_vs_rigid": float(np.mean(p90_vs_rigid)),
            "p90_relative_change_vs_matched_placebo": float(
                np.mean(p90_vs_placebo)
            ),
            "completion_ratio_change_vs_rigid": float(
                np.mean(completion_vs_rigid)
            ),
            "completion_ratio_change_vs_matched_placebo": float(
                np.mean(completion_vs_placebo)
            ),
            "source_changed_rigid_action_count": int(source_changes),
        }

    admitted_rows = [
        city_summaries[city]
        for city in sorted(admitted)
        if city in city_summaries
    ]
    effects_rigid = [
        float(row["primary_relative_change_vs_rigid"]) for row in admitted_rows
    ]
    effects_placebo = [
        float(row["primary_relative_change_vs_matched_placebo"])
        for row in admitted_rows
    ]
    source_changes = sum(
        int(row["source_changed_rigid_action_count"]) for row in admitted_rows
    )
    operational_checks = {
        "all_rollouts_complete": len(rows) == len(expected),
        "zero_teleports_all_arms": not teleport_failures,
        "source_and_placebo_collision_incidents_noninferior_to_rigid_and_phase": (
            not collision_failures
        ),
        "all_nonphase_arms_entered_originator": not originator_failures,
        "pressure_alignment_enforced_for_every_applied_change": (
            not constraint_failures
        ),
        "exact_rigid_fallback_for_offline_rejected_cities": (
            not fallback_failures
        ),
    }
    gate = dict(config["development_gate"])
    if stage == "smoke":
        performance_checks = {
            "nonzero_source_action_changes": source_changes > 0,
            "no_admitted_city_primary_regression_over_smoke_limit_vs_rigid": all(
                effect <= float(gate["smoke_maximum_primary_relative_regression"])
                for effect in effects_rigid
            ),
            "no_admitted_city_primary_regression_over_smoke_limit_vs_placebo": all(
                effect <= float(gate["smoke_maximum_primary_relative_regression"])
                for effect in effects_placebo
            ),
            "no_admitted_city_p90_regression_over_smoke_limit": all(
                float(row["p90_relative_change_vs_rigid"])
                <= float(gate["smoke_maximum_p90_relative_regression"])
                and float(row["p90_relative_change_vs_matched_placebo"])
                <= float(gate["smoke_maximum_p90_relative_regression"])
                for row in admitted_rows
            ),
            "no_admitted_city_completion_drop_over_smoke_limit": all(
                float(row["completion_ratio_change_vs_rigid"])
                >= -float(gate["smoke_maximum_completion_ratio_drop"])
                and float(row["completion_ratio_change_vs_matched_placebo"])
                >= -float(gate["smoke_maximum_completion_ratio_drop"])
                for row in admitted_rows
            ),
        }
        passed = all((*operational_checks.values(), *performance_checks.values()))
        decision = (
            "authorize_v150m_full_development_matrix"
            if passed
            else "reject_v150m_full_matrix_after_smoke"
        )
    else:
        performance_checks = {
            "minimum_two_admitted_cities_nondegrading_vs_rigid": sum(
                effect <= 0.0 for effect in effects_rigid
            )
            >= int(gate["minimum_admitted_cities_nondegrading"]),
            "minimum_two_admitted_cities_nondegrading_vs_placebo": sum(
                effect <= 0.0 for effect in effects_placebo
            )
            >= int(gate["minimum_admitted_cities_nondegrading"]),
            "city_macro_mean_gain_vs_rigid": bool(
                effects_rigid and float(np.mean(effects_rigid)) < 0.0
            ),
            "city_macro_mean_gain_vs_matched_placebo": bool(
                effects_placebo and float(np.mean(effects_placebo)) < 0.0
            ),
            "no_admitted_city_primary_regression_over_one_percent": all(
                effect <= float(gate["maximum_primary_relative_regression"])
                for effect in (*effects_rigid, *effects_placebo)
            ),
            "no_admitted_city_p90_regression_over_five_percent": all(
                float(row["p90_relative_change_vs_rigid"])
                <= float(gate["maximum_p90_relative_regression"])
                and float(row["p90_relative_change_vs_matched_placebo"])
                <= float(gate["maximum_p90_relative_regression"])
                for row in admitted_rows
            ),
            "no_admitted_city_completion_drop_over_one_point": all(
                float(row["completion_ratio_change_vs_rigid"])
                >= -float(gate["maximum_completion_ratio_drop"])
                and float(row["completion_ratio_change_vs_matched_placebo"])
                >= -float(gate["maximum_completion_ratio_drop"])
                for row in admitted_rows
            ),
            "nonzero_source_action_changes": source_changes > 0,
        }
        passed = all((*operational_checks.values(), *performance_checks.values()))
        decision = (
            "freeze_v150m_for_fresh_city_confirmation"
            if passed
            else "retain_rigid_target_adaptation_and_reject_v150m_source_claim"
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
        "effect_scale": "paired_relative_change_with_equal_city_weighting",
        "offline_admitted_source_cities": sorted(admitted),
        "city_summaries": city_summaries,
        "admitted_city_macro_primary_relative_change_vs_rigid": (
            float(np.mean(effects_rigid)) if effects_rigid else None
        ),
        "admitted_city_macro_primary_relative_change_vs_matched_placebo": (
            float(np.mean(effects_placebo)) if effects_placebo else None
        ),
        "city_bootstrap_95ci_vs_rigid": _bootstrap_mean_ci(effects_rigid),
        "city_bootstrap_95ci_vs_matched_placebo": _bootstrap_mean_ci(
            effects_placebo
        ),
        "source_changed_rigid_action_count": int(source_changes),
        "observed_collision_rows": observed_collision_rows,
        "teleport_failures": teleport_failures,
        "paired_collision_audit": paired_collision_audit,
        "paired_collision_failures": collision_failures,
        "originator_failures": originator_failures,
        "constraint_failures": constraint_failures,
        "exact_fallback_failures": fallback_failures,
        "development_gate": {
            "operational_checks": operational_checks,
            "performance_checks": performance_checks,
            "passed": passed,
            "decision": decision,
        },
        "claim_boundary": (
            "V150M is a seven-city development result. A full-stage pass can "
            "freeze the method for a separate untouched-city confirmation but "
            "is not itself independent external evidence."
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
        raise FileExistsError(f"refusing to overwrite V150M aggregate: {args.out}")
    result = aggregate_results(
        [_read_json(path) for path in sorted(args.result_root.rglob("result.json"))],
        config=_read_json(args.config),
        stage=str(args.stage),
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "PASS" if result["development_gate"]["passed"] else "REJECT",
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
