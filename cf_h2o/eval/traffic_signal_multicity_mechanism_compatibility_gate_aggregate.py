"""Aggregate the frozen V148 mechanism-compatibility gate over seven cities."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_multicity_mechanism_compatibility_gate import (
    COMPATIBILITY_FOLD_COUNT,
    RESULT_PROTOCOL as TARGET_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multicity_robust_source_gate_aggregate import (
    DESCRIPTIVE_COMPARATORS,
    PRIMARY_COMPARATORS,
    _history_summary,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_aggregate import (
    MAXIMUM_CITY_REGRESSION,
    MINIMUM_IMPROVING_CITIES,
    MINIMUM_MEAN_IMPROVEMENT,
    _city_effect_summary,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
    MECHANISM_COMPATIBILITY_NAMES,
    TARGET_BUDGET,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v148-multicity-mechanism-compatibility-aggregate-v1"
MAXIMUM_HISTORY_REGRESSION = MAXIMUM_CITY_REGRESSION / 2.0
MINIMUM_HISTORY_Q75_IMPROVEMENT = MINIMUM_MEAN_IMPROVEMENT


def _read_result(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V148 result is not a JSON object: {path}")
    return value


def _validate_target_result(row: Mapping[str, Any], target: str) -> None:
    expected_sources = set(EXPECTED_CITY_GROUPS) - {target}
    diagnostics = row.get("adaptation_mechanism_diagnostics", {})
    source_diagnostics = diagnostics.get("source_arms", {})
    information = row.get("information_budget", {})
    if (
        row.get("protocol") != TARGET_RESULT_PROTOCOL
        or int(row.get("target_budget", -1)) != TARGET_BUDGET
        or set(row.get("source_city_groups", ())) != expected_sources
        or set(row.get("source_arm_summaries", {})) != expected_sources
        or set(row.get("source_placebo_arm_summaries", {})) != expected_sources
        or set(row.get("mechanism_shrunk_source_arm_summaries", {}))
        != expected_sources
        or set(row.get("mechanism_shrunk_placebo_arm_summaries", {}))
        != expected_sources
        or information.get("evaluation_groups_used_for_fit_or_selection") is not False
        or information.get("adaptation_groups_used_for_selector_diagnostics")
        is not True
        or information.get("adaptation_mechanism_scored_group_labels_used")
        is not True
        or int(information.get("adaptation_mechanism_fold_count", -1))
        != COMPATIBILITY_FOLD_COUNT
        or diagnostics.get("evaluation_features_or_labels_used") is not False
        or diagnostics.get("scored_adaptation_labels_used") is not True
        or int(diagnostics.get("adaptation_group_count", -1)) != TARGET_BUDGET
        or int(diagnostics.get("fold_count", -1)) != COMPATIBILITY_FOLD_COUNT
        or tuple(diagnostics.get("mechanism_names", ()))
        != MECHANISM_COMPATIBILITY_NAMES
        or set(source_diagnostics) != expected_sources
    ):
        raise ValueError(f"{target}: V148 information boundary changed")
    for source, values in source_diagnostics.items():
        trusts = values.get("mechanism_trust", {})
        target_losses = values.get("target_only_losses", {})
        source_losses = values.get("source_plus_target_losses", {})
        mass = float(values.get("source_mass", np.nan))
        if (
            values.get("protocol")
            != "five-fold-oof-labeled-mechanism-compatibility-v1"
            or values.get("labels_from_scored_groups_used") is not True
            or values.get("estimator")
            != "fixed-parent-rank0-domain-balanced-residual-v1"
            or int(values.get("fold_count", -1)) != COMPATIBILITY_FOLD_COUNT
            or int(values.get("group_count", -1)) != TARGET_BUDGET
            or tuple(values.get("mechanism_names", ()))
            != MECHANISM_COMPATIBILITY_NAMES
            or set(trusts) != set(MECHANISM_COMPATIBILITY_NAMES)
            or set(target_losses) != set(MECHANISM_COMPATIBILITY_NAMES)
            or set(source_losses) != set(MECHANISM_COMPATIBILITY_NAMES)
            or not np.isfinite(mass)
            or not 0.0 <= mass <= 1.0
        ):
            raise ValueError(f"{target}: invalid V148 diagnostic for {source}")
        trust_values = np.asarray(list(trusts.values()), dtype=float)
        loss_values = np.asarray(
            [*target_losses.values(), *source_losses.values()], dtype=float
        )
        if (
            not np.all(np.isfinite(trust_values))
            or np.any(trust_values < 0.0)
            or np.any(trust_values > 1.0)
            or not np.all(np.isfinite(loss_values))
            or np.any(loss_values < 0.0)
            or not np.isclose(mass, np.mean(trust_values), rtol=0.0, atol=1e-12)
        ):
            raise ValueError(f"{target}: invalid V148 losses or mass for {source}")


def aggregate_mechanism_compatibility_gate(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = tuple(dict(value) for value in results)
    by_target = {str(row.get("target_city")): row for row in rows}
    if len(rows) != len(EXPECTED_CITY_GROUPS) or set(by_target) != set(
        EXPECTED_CITY_GROUPS
    ):
        raise ValueError("V148 requires exactly one result per target city")
    shared_inputs = None
    target_effects: dict[str, dict[str, float]] = {}
    placebo_effects: dict[str, dict[str, float]] = {}
    for target in EXPECTED_CITY_GROUPS:
        row = by_target[target]
        _validate_target_result(row, target)
        identities = dict(row.get("inputs", {}))
        if shared_inputs is None:
            shared_inputs = identities
        elif identities != shared_inputs:
            raise ValueError("V148 target results do not share input identities")
        target_only = float(row["arm_summaries"]["target_only_causal"]["mean"])
        target_effects[target] = {
            source: float(summary["mean"]) - target_only
            for source, summary in row[
                "mechanism_shrunk_source_arm_summaries"
            ].items()
        }
        placebo_effects[target] = {
            source: float(
                row["mechanism_shrunk_source_arm_summaries"][source]["mean"]
            )
            - float(
                row["mechanism_shrunk_placebo_arm_summaries"][source]["mean"]
            )
            for source in row["mechanism_shrunk_source_arm_summaries"]
        }

    decisions = {}
    for target in EXPECTED_CITY_GROUPS:
        meta_cities = tuple(city for city in EXPECTED_CITY_GROUPS if city != target)
        histories = {}
        admitted_sources = []
        diagnostics = by_target[target]["adaptation_mechanism_diagnostics"][
            "source_arms"
        ]
        for source in meta_cities:
            history_targets = tuple(city for city in meta_cities if city != source)
            target_history = _history_summary(
                [target_effects[city][source] for city in history_targets]
            )
            placebo_history = _history_summary(
                [placebo_effects[city][source] for city in history_targets]
            )
            admitted = bool(
                target_history["maximum"] <= MAXIMUM_HISTORY_REGRESSION
                and target_history["q75"]
                <= -MINIMUM_HISTORY_Q75_IMPROVEMENT
                and placebo_history["maximum"] <= MAXIMUM_HISTORY_REGRESSION
                and placebo_history["q75"]
                <= -MINIMUM_HISTORY_Q75_IMPROVEMENT
            )
            histories[source] = {
                "history_targets": list(history_targets),
                "target_only_effect": target_history,
                "matched_placebo_effect": placebo_history,
                "admitted": admitted,
            }
            if admitted:
                admitted_sources.append(source)
        selected_source = None
        selection_stage = "source_null_no_robust_history"
        if admitted_sources:
            selected_source = min(
                admitted_sources,
                key=lambda source: (
                    -float(diagnostics[source]["source_mass"]),
                    float(histories[source]["target_only_effect"]["mean"]),
                    source,
                ),
            )
            selection_stage = "robust_history_max_mechanism_mass"
        decisions[target] = {
            "selected_source": selected_source,
            "selection_stage": selection_stage,
            "admitted_sources": admitted_sources,
            "selected_mechanism_mass": (
                0.0
                if selected_source is None
                else float(diagnostics[selected_source]["source_mass"])
            ),
            "meta_label_cities": list(meta_cities),
            "heldout_city_absent_as_source_and_target": target,
            "source_histories": histories,
        }

    city_arm_means = {}
    effects = {
        comparator: {}
        for comparator in (*PRIMARY_COMPARATORS, *DESCRIPTIVE_COMPARATORS)
    }
    for target in EXPECTED_CITY_GROUPS:
        row = by_target[target]
        arms = row["arm_summaries"]
        selected_source = decisions[target]["selected_source"]
        if selected_source is None:
            selected_mean = float(arms["target_only_causal"]["mean"])
            placebo_mean = selected_mean
        else:
            selected_mean = float(
                row["mechanism_shrunk_source_arm_summaries"][selected_source][
                    "mean"
                ]
            )
            placebo_mean = float(
                row["mechanism_shrunk_placebo_arm_summaries"][selected_source][
                    "mean"
                ]
            )
        comparator_means = {
            "target_only_causal": float(arms["target_only_causal"]["mean"]),
            "pooled_h2oplus": float(arms["pooled_h2oplus"]["mean"]),
            "matched_source_placebo": placebo_mean,
            "uniform_cfcmt": float(arms["uniform_cfcmt"]["mean"]),
            "phase_pressure": float(arms["phase_pressure"]["mean"]),
        }
        city_arm_means[target] = {
            "selected_source": selected_source,
            "selected_mechanism_mass": decisions[target][
                "selected_mechanism_mass"
            ],
            "selected_gate_mean": selected_mean,
            **comparator_means,
        }
        for comparator, comparator_mean in comparator_means.items():
            effects[comparator][target] = selected_mean - comparator_mean

    summaries = {
        comparator: _city_effect_summary(values)
        for comparator, values in effects.items()
    }
    comparator_pass = {}
    for comparator in PRIMARY_COMPARATORS:
        summary = summaries[comparator]
        comparator_pass[comparator] = bool(
            float(summary["mean"]) <= -MINIMUM_MEAN_IMPROVEMENT
            and float(summary["upper_95_one_sided"]) < 0.0
            and int(summary["improving_city_count"]) >= MINIMUM_IMPROVING_CITIES
            and float(summary["maximum_city_regression"])
            <= MAXIMUM_CITY_REGRESSION
        )
    passed = all(comparator_pass.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "seven-city-mechanism-compatibility-development-aggregate",
        "target_city_groups": list(EXPECTED_CITY_GROUPS),
        "target_budget": TARGET_BUDGET,
        "selection_rule": {
            "cross_fit_folds": COMPATIBILITY_FOLD_COUNT,
            "maximum_history_regression": MAXIMUM_HISTORY_REGRESSION,
            "minimum_history_q75_improvement": (
                MINIMUM_HISTORY_Q75_IMPROVEMENT
            ),
            "source_rank": "maximum mechanism mass then history mean then name",
            "source_mass": "equal mean of five labeled OOF mechanism trusts",
            "source_null": "no source passing both robust history comparisons",
        },
        "city_decisions": decisions,
        "city_arm_means": city_arm_means,
        "paired_city_effects": {
            f"mechanism_gate_minus_{comparator}": summary
            for comparator, summary in summaries.items()
        },
        "development_gate": {
            "comparator_pass": comparator_pass,
            "passed": passed,
            "decision": (
                "authorize_preregistered_v149_boston_confirmation"
                if passed
                else "close_v148_mechanism_compatibility_gate"
            ),
            "minimum_mean_improvement": MINIMUM_MEAN_IMPROVEMENT,
            "minimum_improving_cities": MINIMUM_IMPROVING_CITIES,
            "maximum_city_regression": MAXIMUM_CITY_REGRESSION,
        },
        "inputs": shared_inputs,
        "claim_boundary": (
            "V148 is seven-city target-offline few-shot development. B5 "
            "next-state labels estimate mechanism source mass; evaluation groups "
            "remain excluded from each target's gate. Mechanisms do not select "
            "actions, and a pass only authorizes preregistered V149."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V148 aggregate: {args.out}")
    result = aggregate_mechanism_compatibility_gate(
        [_read_result(path) for path in args.results]
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "PASS" if result["development_gate"]["passed"] else "REJECT",
                "protocol": result["protocol"],
                "development_gate": result["development_gate"],
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
