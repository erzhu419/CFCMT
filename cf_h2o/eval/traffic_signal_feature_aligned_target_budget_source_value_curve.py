"""Adjudicate the correction-only V157A replay against legacy V123."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    ARTIFACT_PROTOCOL as V123_ARTIFACT_PROTOCOL,
    RESULT_PROTOCOL as V123_RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = (
    "tsc-v157a-feature-aligned-target-budget-source-value-curve-aggregate-v1"
)
LAUNCH_PROTOCOL = (
    "tsc-v157a-feature-aligned-target-budget-source-value-curve-launch-v1"
)
CONFIG_PROTOCOL = (
    "tsc-v157a-feature-aligned-target-budget-source-value-curve-v1"
)
SIGNATURE = "CFCMT/v157a/feature-aligned-v123-source-value-curve-v1"
LEGACY_RESULT_SHA256 = (
    "cc59ab48768901a0e73d91cde878b0a764b6a8e16f8a929df4a2406efd80d944"
)
MINIMUM_SOURCE_CONTRIBUTION = 0.0005


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_contract(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in dict(row).items() if key != "elapsed_seconds"}
        for row in rows
    ]


def _budget_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    nested = dict(row["nested_source_selection"])
    bootstrap = dict(nested["paired_source_minus_target_bootstrap"])
    mean = float(bootstrap["mean"])
    upper = float(bootstrap["upper_95"])
    source_vs_pp = float(nested["mean_source_policy_delta"])
    target_vs_pp = float(nested["mean_target_only_policy_delta"])
    relative_pass = mean <= -MINIMUM_SOURCE_CONTRIBUTION and upper < 0.0
    return {
        "mean_source_minus_architecture_matched_target": mean,
        "paired_95_lower": float(bootstrap["lower_95"]),
        "paired_95_upper": upper,
        "improving_seed_fraction": float(bootstrap["improving_seed_fraction"]),
        "source_policy_delta_vs_phase_pressure": source_vs_pp,
        "target_only_delta_vs_phase_pressure": target_vs_pp,
        "selected_candidate_counts": deepcopy(nested["selected_candidate_counts"]),
        "relative_source_contribution_passed": relative_pass,
        "v123_budget_level_absolute_conditions_passed": (
            relative_pass and source_vs_pp < 0.0
        ),
    }


def aggregate_results(
    corrected: Mapping[str, Any],
    legacy: Mapping[str, Any],
    launch: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    identities = dict(config.get("frozen_identities", {}))
    if (
        corrected.get("protocol") != V123_RESULT_PROTOCOL
        or legacy.get("protocol") != V123_RESULT_PROTOCOL
        or corrected.get("artifact", {}).get("protocol") != V123_ARTIFACT_PROTOCOL
        or config.get("protocol") != CONFIG_PROTOCOL
        or launch.get("protocol") != LAUNCH_PROTOCOL
        or launch.get("submitted") is not True
        or launch.get("signature") != SIGNATURE
        or launch.get("snapshot_sha256")
        != identities.get("derived_snapshot_sha256")
        or launch.get("source_tree_sha256")
        != identities.get("derived_source_tree_sha256")
        or launch.get("snapshot_patch", {}).get("parent_sha256")
        != identities.get("parent_action_ranker_sha256")
        or launch.get("snapshot_patch", {}).get("patched_sha256")
        != identities.get("patched_action_ranker_sha256")
    ):
        raise ValueError("V157A protocol or launch identity changed")

    invariant_fields = (
        "city",
        "estimand",
        "target_name",
        "target_budgets",
        "selector_scenario",
        "selector_seed_count",
        "selector_group_count",
        "selector_row_count",
        "selection_audits",
        "input_audits",
        "inputs",
    )
    invariants = {
        field: corrected.get(field) == legacy.get(field) for field in invariant_fields
    }
    invariants["fit_contract"] = _fit_contract(corrected.get("fit_diagnostics", ())) == (
        _fit_contract(legacy.get("fit_diagnostics", ()))
    )
    if not all(invariants.values()):
        raise ValueError(f"V157A changed more than feature binding: {invariants}")

    corrected_budgets = dict(corrected["budget_results"])
    legacy_budgets = dict(legacy["budget_results"])
    positive = tuple(str(value) for value in config["target_budgets"] if int(value) > 0)
    if set(corrected_budgets) != set(legacy_budgets) or any(
        budget not in corrected_budgets for budget in positive
    ):
        raise ValueError("V157A budget coverage changed")

    comparison = {}
    for budget in positive:
        new = _budget_summary(corrected_budgets[budget])
        old = _budget_summary(legacy_budgets[budget])
        comparison[budget] = {
            "legacy": old,
            "corrected": new,
            "corrected_minus_legacy_source_contribution": (
                new["mean_source_minus_architecture_matched_target"]
                - old["mean_source_minus_architecture_matched_target"]
            ),
        }

    b100 = comparison["100"]["corrected"]
    branch_passed = bool(b100["relative_source_contribution_passed"])
    return {
        "protocol": RESULT_PROTOCOL,
        "scientific_status": (
            "feature_aligned_b100_relative_source_contribution_retained"
            if branch_passed
            else "feature_aligned_b100_relative_source_contribution_rejected"
        ),
        "correction_only_contract": {
            "parent_protocol": V123_RESULT_PROTOCOL,
            "feature_binding": "stored_feature_names",
            "parent_snapshot_sha256": identities["parent_snapshot_sha256"],
            "derived_snapshot_sha256": identities["derived_snapshot_sha256"],
            "patched_action_ranker_sha256": identities[
                "patched_action_ranker_sha256"
            ],
            "all_input_split_and_fit_invariants_passed": True,
            "invariant_checks": invariants,
            "counterfactual_costs_reused": True,
            "new_sumo_simulations": 0,
        },
        "budget_comparison": comparison,
        "corrected_original_v123_selection": deepcopy(corrected["selection"]),
        "legacy_original_v123_selection": deepcopy(legacy["selection"]),
        "b100_branch_authorization": {
            "minimum_mean_source_contribution": MINIMUM_SOURCE_CONTRIBUTION,
            "mean_must_be_at_most": -MINIMUM_SOURCE_CONTRIBUTION,
            "paired_95_upper_must_be_below": 0.0,
            "passed": branch_passed,
            "decision": (
                "authorize_freezing_separate_same_state_450_second_branch_protocol"
                if branch_passed
                else "do_not_run_branch_and_close_v123_source_value_premise"
            ),
            "does_not_authorize_full_closed_loop_or_external_city": True,
        },
        "provenance": {
            "legacy_result_sha256": LEGACY_RESULT_SHA256,
            "corrected_prediction_artifact": deepcopy(corrected["artifact"]),
            "launch_protocol": LAUNCH_PROTOCOL,
            "task_id": json.loads(launch["scheduler_stdout"])["submitted"][0]["id"],
            "execution_node": launch["execution_node"],
            "snapshot_sha256": launch["snapshot_sha256"],
            "source_tree_sha256": launch["source_tree_sha256"],
        },
        "claim_boundary": (
            "Correction-only replay on reused V123 Jinan development data; "
            "not fresh-seed, closed-loop, unseen-city or PhasePressure-superiority evidence."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V157A aggregate")
    if _sha256(args.legacy) != LEGACY_RESULT_SHA256:
        raise ValueError("legacy V123 result identity changed")
    launch = _read_json(args.launch)
    if _sha256(args.config) != launch.get("config_sha256"):
        raise ValueError("V157A config differs from the submitted launch")
    result = aggregate_results(
        _read_json(args.corrected),
        _read_json(args.legacy),
        launch,
        _read_json(args.config),
    )
    result["provenance"]["corrected_result_sha256"] = _sha256(args.corrected)
    result["provenance"]["launch_sha256"] = _sha256(args.launch)
    result["provenance"]["config_sha256"] = _sha256(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
