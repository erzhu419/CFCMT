"""Aggregate the frozen V146 OOF compatibility gate over seven cities."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_multicity_oof_compatibility_gate import (
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
    TARGET_BUDGET,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v146-multicity-oof-compatibility-gate-aggregate-v1"
MINIMUM_HISTORY_MEAN_IMPROVEMENT = MINIMUM_MEAN_IMPROVEMENT


def _read_result(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V146 result is not a JSON object: {path}")
    return value


def _validate_target_result(row: Mapping[str, Any], target: str) -> None:
    expected_sources = set(EXPECTED_CITY_GROUPS) - {target}
    diagnostics = row.get("adaptation_compatibility_diagnostics", {})
    source_diagnostics = diagnostics.get("source_arms", {})
    information = row.get("information_budget", {})
    if (
        row.get("protocol") != TARGET_RESULT_PROTOCOL
        or int(row.get("target_budget", -1)) != TARGET_BUDGET
        or set(row.get("source_city_groups", ())) != expected_sources
        or set(row.get("source_arm_summaries", {})) != expected_sources
        or set(row.get("source_placebo_arm_summaries", {})) != expected_sources
        or set(row.get("compatibility_shrunk_source_arm_summaries", {}))
        != expected_sources
        or set(row.get("compatibility_shrunk_placebo_arm_summaries", {}))
        != expected_sources
        or information.get("evaluation_groups_used_for_fit_or_selection") is not False
        or information.get("adaptation_groups_used_for_selector_diagnostics")
        is not True
        or information.get("adaptation_compatibility_scored_group_labels_used")
        is not False
        or int(information.get("adaptation_compatibility_fold_count", -1))
        != COMPATIBILITY_FOLD_COUNT
        or diagnostics.get("evaluation_features_or_labels_used") is not False
        or diagnostics.get("scored_adaptation_labels_used") is not False
        or int(diagnostics.get("adaptation_group_count", -1)) != TARGET_BUDGET
        or int(diagnostics.get("fold_count", -1)) != COMPATIBILITY_FOLD_COUNT
        or set(source_diagnostics) != expected_sources
    ):
        raise ValueError(f"{target}: V146 information boundary changed")
    for source, values in source_diagnostics.items():
        if (
            values.get("protocol") != "five-fold-oof-policy-compatibility-v1"
            or values.get("labels_from_scored_groups_used") is not False
            or int(values.get("fold_count", -1)) != COMPATIBILITY_FOLD_COUNT
            or int(values.get("group_count", -1)) != TARGET_BUDGET
        ):
            raise ValueError(f"{target}: invalid V146 diagnostic for {source}")
        for field in (
            "action_agreement_fraction",
            "chance_action_agreement",
            "chance_corrected_trust_mass",
            "pairwise_rank_agreement_fraction",
            "mean_normalized_target_regret",
            "q90_normalized_target_regret",
        ):
            value = float(values.get(field, np.nan))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{target}: invalid V146 {field} for {source}")


def aggregate_oof_compatibility_gate(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = tuple(dict(value) for value in results)
    by_target = {str(row.get("target_city")): row for row in rows}
    if len(rows) != len(EXPECTED_CITY_GROUPS) or set(by_target) != set(
        EXPECTED_CITY_GROUPS
    ):
        raise ValueError("V146 requires exactly one result per target city")
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
            raise ValueError("V146 target results do not share input identities")
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

    decisions = {}
    for target in EXPECTED_CITY_GROUPS:
        meta_cities = tuple(city for city in EXPECTED_CITY_GROUPS if city != target)
        histories = {}
        eligible_sources = []
        diagnostics = by_target[target]["adaptation_compatibility_diagnostics"][
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
            eligible = bool(
                target_history["mean"] <= -MINIMUM_HISTORY_MEAN_IMPROVEMENT
                and placebo_history["mean"] <= -MINIMUM_HISTORY_MEAN_IMPROVEMENT
            )
            histories[source] = {
                "history_targets": list(history_targets),
                "target_only_effect": target_history,
                "matched_placebo_effect": placebo_history,
                "eligible": eligible,
            }
            if eligible:
                eligible_sources.append(source)
        selected_source = None
        selection_stage = "source_null_no_positive_history"
        if eligible_sources:
            candidate = min(
                eligible_sources,
                key=lambda source: (
                    -float(diagnostics[source]["chance_corrected_trust_mass"]),
                    float(diagnostics[source]["mean_normalized_target_regret"]),
                    float(histories[source]["target_only_effect"]["mean"]),
                    source,
                ),
            )
            if float(diagnostics[candidate]["chance_corrected_trust_mass"]) > 0.0:
                selected_source = candidate
                selection_stage = "positive_history_max_oof_compatibility"
            else:
                selection_stage = "source_null_zero_oof_compatibility"
        decisions[target] = {
            "selected_source": selected_source,
            "selection_stage": selection_stage,
            "eligible_sources": eligible_sources,
            "selected_trust_mass": (
                0.0
                if selected_source is None
                else float(
                    diagnostics[selected_source]["chance_corrected_trust_mass"]
                )
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
                row["compatibility_shrunk_source_arm_summaries"][selected_source][
                    "mean"
                ]
            )
            placebo_mean = float(
                row["compatibility_shrunk_placebo_arm_summaries"][selected_source][
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
            "selected_trust_mass": decisions[target]["selected_trust_mass"],
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
        "scientific_status": "seven-city-oof-compatibility-development-aggregate",
        "target_city_groups": list(EXPECTED_CITY_GROUPS),
        "target_budget": TARGET_BUDGET,
        "selection_rule": {
            "cross_fit_folds": COMPATIBILITY_FOLD_COUNT,
            "source_history_gate": (
                "leave-one-target-and-source mean effects <= -0.0005 versus "
                "target-only and matched placebo"
            ),
            "source_rank": (
                "maximum chance-corrected OOF policy compatibility, then "
                "minimum OOF target-model regret, then history mean"
            ),
            "source_mass": "chance-corrected OOF action agreement",
            "source_null": "no positive-history source or zero trust mass",
        },
        "city_decisions": decisions,
        "city_arm_means": city_arm_means,
        "paired_city_effects": {
            f"oof_gate_minus_{comparator}": summary
            for comparator, summary in summaries.items()
        },
        "development_gate": {
            "comparator_pass": comparator_pass,
            "passed": passed,
            "decision": (
                "authorize_separately_frozen_untouched_city_protocol"
                if passed
                else "close_v146_oof_compatibility_gate"
            ),
            "minimum_mean_improvement": MINIMUM_MEAN_IMPROVEMENT,
            "minimum_improving_cities": MINIMUM_IMPROVING_CITIES,
            "maximum_city_regression": MAXIMUM_CITY_REGRESSION,
        },
        "inputs": shared_inputs,
        "claim_boundary": (
            "V146 is post-V145 seven-city method development. Selection uses "
            "five-fold adaptation OOF predictions and leave-one-city source "
            "history; no evaluated target labels enter its own gate. A pass "
            "only authorizes a separately frozen untouched-city protocol."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V146 aggregate: {args.out}")
    result = aggregate_oof_compatibility_gate(
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
