"""Aggregate the frozen V145 robust source gate over seven target cities."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_multicity_robust_source_gate import (
    RESULT_PROTOCOL as TARGET_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_aggregate import (
    MAXIMUM_CITY_REGRESSION,
    MINIMUM_IMPROVING_CITIES,
    MINIMUM_MEAN_IMPROVEMENT,
    _city_effect_summary,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
    TARGET_BUDGET,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v145-multicity-robust-source-gate-aggregate-v1"
MAXIMUM_HISTORY_REGRESSION = MAXIMUM_CITY_REGRESSION / 2.0
MINIMUM_HISTORY_Q75_IMPROVEMENT = MINIMUM_MEAN_IMPROVEMENT
MAXIMUM_ADAPTATION_INTERVENTION_FRACTION = 0.90
PRIMARY_COMPARATORS = (
    "target_only_causal",
    "pooled_h2oplus",
    "matched_source_placebo",
)
DESCRIPTIVE_COMPARATORS = ("uniform_cfcmt", "phase_pressure")


def _read_result(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V145 result is not a JSON object: {path}")
    return value


def _history_summary(values: Sequence[float]) -> dict[str, float]:
    rows = np.asarray(values, dtype=float)
    if rows.size < 2 or not np.all(np.isfinite(rows)):
        raise ValueError("V145 source history is incomplete")
    return {
        "mean": float(np.mean(rows)),
        "q75": float(np.quantile(rows, 0.75)),
        "maximum": float(np.max(rows)),
    }


def _validate_target_result(row: Mapping[str, Any], target: str) -> None:
    expected_sources = set(EXPECTED_CITY_GROUPS) - {target}
    diagnostics = row.get("adaptation_selector_diagnostics", {})
    diagnostic_arms = diagnostics.get("arms", {})
    if (
        row.get("protocol") != TARGET_RESULT_PROTOCOL
        or int(row.get("target_budget", -1)) != TARGET_BUDGET
        or set(row.get("source_city_groups", ())) != expected_sources
        or set(row.get("source_arm_summaries", {})) != expected_sources
        or set(row.get("source_placebo_arm_summaries", {})) != expected_sources
        or row.get("information_budget", {}).get(
            "evaluation_groups_used_for_fit_or_selection"
        )
        is not False
        or row.get("information_budget", {}).get(
            "adaptation_groups_used_for_selector_diagnostics"
        )
        is not True
        or diagnostics.get("evaluation_features_or_labels_used") is not False
        or int(diagnostics.get("adaptation_group_count", -1)) != TARGET_BUDGET
        or set(diagnostic_arms) != {"target_only", "h2oplus", *expected_sources}
    ):
        raise ValueError(f"{target}: V145 information boundary changed")
    for arm, values in diagnostic_arms.items():
        fraction = float(values.get("intervention_fraction", np.nan))
        if (
            values.get("protocol")
            != "adaptation-action-reference-intervention-v1"
            or values.get("information_boundary")
            != "fitted_b25_model_predictions_and_phase_pressure_reference_only"
            or int(values.get("group_count", -1)) != TARGET_BUDGET
            or not 0.0 <= fraction <= 1.0
        ):
            raise ValueError(f"{target}: invalid V145 selector diagnostic for {arm}")


def aggregate_robust_source_gate(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = tuple(dict(value) for value in results)
    by_target = {str(row.get("target_city")): row for row in rows}
    if len(rows) != len(EXPECTED_CITY_GROUPS) or set(by_target) != set(
        EXPECTED_CITY_GROUPS
    ):
        raise ValueError("V145 requires exactly one result per target city")
    shared_inputs = None
    target_effects: dict[str, dict[str, float]] = {}
    placebo_effects: dict[str, dict[str, float]] = {}
    intervention_fractions: dict[str, dict[str, float]] = {}
    for target in EXPECTED_CITY_GROUPS:
        row = by_target[target]
        _validate_target_result(row, target)
        identities = dict(row.get("inputs", {}))
        if shared_inputs is None:
            shared_inputs = identities
        elif identities != shared_inputs:
            raise ValueError("V145 target results do not share input identities")
        target_only = float(row["arm_summaries"]["target_only_causal"]["mean"])
        target_effects[target] = {
            source: float(summary["mean"]) - target_only
            for source, summary in row["source_arm_summaries"].items()
        }
        placebo_effects[target] = {
            source: float(row["source_arm_summaries"][source]["mean"])
            - float(row["source_placebo_arm_summaries"][source]["mean"])
            for source in row["source_arm_summaries"]
        }
        intervention_fractions[target] = {
            source: float(values["intervention_fraction"])
            for source, values in row["adaptation_selector_diagnostics"][
                "arms"
            ].items()
            if source in target_effects[target]
        }

    decisions = {}
    for target in EXPECTED_CITY_GROUPS:
        meta_cities = tuple(city for city in EXPECTED_CITY_GROUPS if city != target)
        minimum_intervention_source = min(
            meta_cities,
            key=lambda source: (
                intervention_fractions[target][source],
                source,
            ),
        )
        histories = {}
        robust_sources = {}
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
                robust_sources[source] = target_history["mean"]
        minimum_intervention = float(
            intervention_fractions[target][minimum_intervention_source]
        )
        if minimum_intervention > MAXIMUM_ADAPTATION_INTERVENTION_FRACTION:
            selected_source = None
            selection_stage = "source_null_high_intervention_veto"
        elif robust_sources:
            selected_source = min(
                robust_sources,
                key=lambda source: (robust_sources[source], source),
            )
            selection_stage = "robust_history"
        else:
            selected_source = minimum_intervention_source
            selection_stage = "minimum_intervention_fallback"
        decisions[target] = {
            "selected_source": selected_source,
            "selection_stage": selection_stage,
            "minimum_intervention_source": minimum_intervention_source,
            "minimum_adaptation_intervention_fraction": minimum_intervention,
            "meta_label_cities": list(meta_cities),
            "heldout_city_absent_as_source_and_target": target,
            "source_histories": histories,
        }

    city_arm_means = {}
    effects = {
        comparator: {} for comparator in (*PRIMARY_COMPARATORS, *DESCRIPTIVE_COMPARATORS)
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
                row["source_arm_summaries"][selected_source]["mean"]
            )
            placebo_mean = float(
                row["source_placebo_arm_summaries"][selected_source]["mean"]
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
        "scientific_status": "seven-city-source-gate-development-aggregate",
        "target_city_groups": list(EXPECTED_CITY_GROUPS),
        "target_budget": TARGET_BUDGET,
        "selection_rule": {
            "maximum_history_regression": MAXIMUM_HISTORY_REGRESSION,
            "minimum_history_q75_improvement": MINIMUM_HISTORY_Q75_IMPROVEMENT,
            "maximum_adaptation_intervention_fraction": (
                MAXIMUM_ADAPTATION_INTERVENTION_FRACTION
            ),
            "fallback": "minimum_adaptation_intervention_then_source_null_veto",
        },
        "city_decisions": decisions,
        "city_arm_means": city_arm_means,
        "paired_city_effects": {
            f"robust_gate_minus_{comparator}": summary
            for comparator, summary in summaries.items()
        },
        "development_gate": {
            "comparator_pass": comparator_pass,
            "passed": passed,
            "decision": (
                "authorize_separately_frozen_untouched_city_protocol"
                if passed
                else "close_v145_robust_source_gate"
            ),
            "minimum_mean_improvement": MINIMUM_MEAN_IMPROVEMENT,
            "minimum_improving_cities": MINIMUM_IMPROVING_CITIES,
            "maximum_city_regression": MAXIMUM_CITY_REGRESSION,
        },
        "inputs": shared_inputs,
        "claim_boundary": (
            "V145 is post-V144 seven-city method development. A pass only "
            "authorizes a frozen untouched-city protocol. It is neither "
            "zero-shot evidence nor untouched-city confirmation."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V145 aggregate: {args.out}")
    result = aggregate_robust_source_gate([_read_result(path) for path in args.results])
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
