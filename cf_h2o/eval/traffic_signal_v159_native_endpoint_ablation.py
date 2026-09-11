"""Target-only ablation of direct action-endpoint information on V158 B100."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn

from cf_h2o.eval import traffic_signal_v158_native_prefix_ranking_calibration as v158
from cf_h2o.traffic_signal.action_ranker import (
    CausalReferenceEndpointResidualRegressor,
    CausalReferenceResidualRegressor,
    PAIRWISE_CAUSAL_ACTION_PARENTS,
    PAIRWISE_CAUSAL_STATE_PARENTS,
    PairwisePreferenceConfig,
)
from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


PROTOCOL = "tsc-v159-jinan-native-endpoint-ablation-v1"
RESULT_PROTOCOL = "tsc-v159-jinan-native-endpoint-ablation-result-v1"
REPRESENTATIONS = ("delta_target", "endpoint_target")
ACTION_DIFFERENCE_FEATURES = (
    *PAIRWISE_CAUSAL_ACTION_PARENTS,
    *v158.META_ACTION_FEATURES,
)
PHYSICAL_MIDPOINT_FEATURES = tuple(PAIRWISE_CAUSAL_ACTION_PARENTS)
TOLERANCE = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected_model = {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 10.0,
        "uncertainty_quantile": 0.9,
        "random_state": 20260803,
    }
    expected_gate = {
        "comparison": "endpoint_target_minus_delta_target",
        "mean_difference_maximum": -0.0005,
        "paired_bootstrap_upper_95_maximum": 0.0,
        "minimum_improved_seeds": 7,
        "require_every_demand_noninferior": True,
        "demand_mean_difference_maximum": 0.0,
        "bootstrap_draws": 10000,
        "bootstrap_seed": 20260911,
    }
    checks = (
        protocol.get("protocol") == PROTOCOL,
        protocol.get("scientific_status") == "prepared_not_run",
        protocol.get("parent_v158", {}).get("protocol") == v158.PROTOCOL,
        protocol.get("parent_v158", {}).get("fit_result_protocol")
        == v158.FIT_RESULT_PROTOCOL,
        protocol.get("parent_v158", {}).get("fit_result_sha256")
        == "45a236690fe7a18f76057b4f8c45ebb3ba4fd03571dd3624b3741aea1d4438d3",
        protocol.get("parent_v158", {}).get("native_bank_sha256")
        == "b484411781d8bf9386cc636542bcf5ef57ae9c09e83aaa5443c870379010039b",
        int(protocol.get("parent_v158", {}).get("native_bank_size_bytes", -1))
        == 236663,
        tuple(map(int, protocol.get("development", {}).get("seeds", ())))
        == v158.DEVELOPMENT_SEEDS,
        tuple(protocol.get("development", {}).get("scenarios", ()))
        == ("jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"),
        int(protocol.get("development", {}).get("action_groups", -1)) == 100,
        int(protocol.get("development", {}).get("candidate_rows", -1)) == 800,
        int(protocol.get("development", {}).get("candidate_actions_per_group", -1))
        == 8,
        int(protocol.get("development", {}).get("new_simulator_trajectories", -1))
        == 0,
        tuple(
            tuple(map(int, fold))
            for fold in protocol.get("crossfit", {}).get("folds", ())
        )
        == v158.CROSSFIT_FOLDS,
        protocol.get("crossfit", {}).get("unit") == "whole_seed",
        int(protocol.get("crossfit", {}).get("fold_count", -1)) == 5,
        protocol.get("model", {}).get("family")
        == "HistGradientBoostingRegressor",
        protocol.get("model", {}).get("config") == expected_model,
        tuple(protocol.get("model", {}).get("state_feature_names", ()))
        == PAIRWISE_CAUSAL_STATE_PARENTS,
        tuple(protocol.get("model", {}).get("action_difference_feature_names", ()))
        == ACTION_DIFFERENCE_FEATURES,
        tuple(protocol.get("model", {}).get("physical_midpoint_feature_names", ()))
        == PHYSICAL_MIDPOINT_FEATURES,
        protocol.get("model", {}).get("representations")
        == {
            "delta_target": "state_plus_action_difference",
            "endpoint_target": (
                "state_plus_action_difference_plus_physical_action_midpoint"
            ),
        },
        protocol.get("model", {}).get("target_base_prediction_only") is True,
        protocol.get("model", {}).get("source_prediction_columns_used") is False,
        float(protocol.get("action_selection", {}).get("threshold", np.nan))
        == 0.0,
        protocol.get("action_selection", {}).get("no_stay_override_of_pp_switch")
        is True,
        protocol.get("action_selection", {}).get("reference_policy")
        == "phase_pressure",
        protocol.get("development_gate") == expected_gate,
    )
    if not all(checks):
        raise ValueError("V159 frozen protocol changed")


def _validate_inputs(
    protocol: Mapping[str, Any],
    *,
    parent_result_path: Path,
    bank_path: Path,
) -> tuple[dict[str, Any], MechanismDataset]:
    parent_spec = protocol["parent_v158"]
    if _sha256(parent_result_path) != parent_spec["fit_result_sha256"]:
        raise ValueError("V159 parent V158 result identity changed")
    parent = _read_json(parent_result_path)
    if (
        parent.get("protocol") != v158.FIT_RESULT_PROTOCOL
        or parent.get("status") != "OOF_GATE_FAIL"
        or parent.get("reserve_authorized") is not False
        or parent.get("artifacts", {}).get("native_b100", {}).get("sha256")
        != parent_spec["native_bank_sha256"]
        or int(
            parent.get("artifacts", {})
            .get("native_b100", {})
            .get("size_bytes", -1)
        )
        != int(parent_spec["native_bank_size_bytes"])
    ):
        raise ValueError("V159 parent V158 result contract changed")
    if bank_path.stat().st_size != int(parent_spec["native_bank_size_bytes"]):
        raise ValueError("V159 native bank size changed")
    if _sha256(bank_path) != parent_spec["native_bank_sha256"]:
        raise ValueError("V159 native bank identity changed")

    data = load_mechanism_dataset(bank_path)
    groups = np.asarray(data.metadata.get("action_group_ids", ()), dtype=str)
    references = np.asarray(data.metadata.get("is_reference", ()), dtype=bool)
    seeds = np.asarray(data.metadata.get("row_seeds", ()), dtype=int)
    scenarios = np.asarray(data.metadata.get("row_scenarios", ()), dtype=str)
    required_features = {
        *PAIRWISE_CAUSAL_STATE_PARENTS,
        *PAIRWISE_CAUSAL_ACTION_PARENTS,
        *v158.RAW_PREDICTION_FEATURES,
    }
    unique_groups = tuple(dict.fromkeys(groups.tolist()))
    if (
        data.size != 800
        or groups.shape != (data.size,)
        or references.shape != (data.size,)
        or seeds.shape != (data.size,)
        or scenarios.shape != (data.size,)
        or len(unique_groups) != 100
        or any(np.count_nonzero(groups == group) != 8 for group in unique_groups)
        or any(np.count_nonzero(references[groups == group]) != 1 for group in unique_groups)
        or tuple(sorted(set(seeds.tolist())))
        != tuple(sorted(map(int, protocol["development"]["seeds"])))
        or tuple(sorted(set(scenarios.tolist())))
        != tuple(sorted(protocol["development"]["scenarios"]))
        or not required_features.issubset(data.feature_names)
        or data.metadata.get("protocol") != v158.PROTOCOL
        or data.metadata.get("native_prefix") is not True
        or data.metadata.get("snapshot_restore_used") is not False
        or not np.all(np.isfinite(data.targets["interval_cost"]))
        or not np.all(np.isfinite(data.targets["absolute_cost"]))
    ):
        raise ValueError("V159 native B100 contract changed")
    if np.any(np.abs(np.asarray(data.targets["interval_cost"])[references]) > TOLERANCE):
        raise ValueError("V159 PP reference labels are not zero")
    return parent, data


def _model_config(protocol: Mapping[str, Any]) -> PairwisePreferenceConfig:
    return PairwisePreferenceConfig(**dict(protocol["model"]["config"]))


def _target_dataset(dataset: MechanismDataset) -> MechanismDataset:
    result = v158._arm_dataset(dataset, "target_native")
    index = {name: position for position, name in enumerate(result.feature_names)}
    missing = [
        name
        for name in (
            *PAIRWISE_CAUSAL_STATE_PARENTS,
            *ACTION_DIFFERENCE_FEATURES,
        )
        if name not in index
    ]
    if missing:
        raise KeyError(f"V159 target-only design is missing features: {missing}")
    auxiliary = result.features[
        :, [index[name] for name in v158.AUX_META_FEATURES]
    ]
    if np.any(np.abs(auxiliary) > TOLERANCE):
        raise ValueError("V159 target-only arm contains nonzero auxiliary predictions")
    selected = result.features[
        :,
        [
            index[name]
            for name in (
                *PAIRWISE_CAUSAL_STATE_PARENTS,
                *ACTION_DIFFERENCE_FEATURES,
            )
        ],
    ]
    if not np.all(np.isfinite(selected)):
        raise ValueError("V159 target-only design contains nonfinite values")
    return result


def _fit_representation(
    protocol: Mapping[str, Any],
    dataset: MechanismDataset,
    representation: str,
):
    target_data = _target_dataset(dataset)
    common = {
        "config": _model_config(protocol),
        "state_feature_names": PAIRWISE_CAUSAL_STATE_PARENTS,
        "action_feature_names": ACTION_DIFFERENCE_FEATURES,
    }
    if representation == "delta_target":
        model = CausalReferenceResidualRegressor(**common)
    elif representation == "endpoint_target":
        model = CausalReferenceEndpointResidualRegressor(
            **common,
            endpoint_feature_names=PHYSICAL_MIDPOINT_FEATURES,
        )
    else:
        raise ValueError(f"unknown V159 representation: {representation}")
    diagnostics = model.fit(target_data)
    diagnostics["representation"] = representation
    diagnostics["design_feature_count"] = (
        len(PAIRWISE_CAUSAL_STATE_PARENTS)
        + len(ACTION_DIFFERENCE_FEATURES)
        + (len(PHYSICAL_MIDPOINT_FEATURES) if representation == "endpoint_target" else 0)
    )
    return model, diagnostics, target_data


def _policy_records(
    model: Any,
    dataset: MechanismDataset,
) -> list[dict[str, Any]]:
    target_data = _target_dataset(dataset)
    scores = np.asarray(
        model.predict(target_data)["control_cost"]["mean"], dtype=float
    )
    groups = np.asarray(target_data.metadata["action_group_ids"], dtype=str)
    references = np.asarray(target_data.metadata["is_reference"], dtype=bool)
    seeds = np.asarray(target_data.metadata["row_seeds"], dtype=int)
    scenarios = np.asarray(target_data.metadata["row_scenarios"], dtype=str)
    states = np.asarray(target_data.metadata["candidate_states"], dtype=str)
    absolute = np.asarray(target_data.targets["absolute_cost"], dtype=float)
    switch_index = target_data.feature_names.index("switch_indicator")
    records = []
    for group in dict.fromkeys(groups.tolist()):
        rows = np.flatnonzero(groups == group)
        reference_rows = rows[references[rows]]
        if reference_rows.size != 1:
            raise ValueError("V159 group does not contain exactly one PP row")
        reference = int(reference_rows[0])
        selected = v158._select_row(
            target_data,
            scores,
            rows,
            prohibit_stay_override=True,
        )
        oracle = int(rows[np.argmin(absolute[rows])])
        candidates = rows[~references[rows]]
        actual_advantages = absolute[reference] - absolute[candidates]
        predicted_advantages = -scores[candidates]
        candidate_sign_correct = int(
            np.count_nonzero(
                (predicted_advantages > TOLERANCE)
                == (actual_advantages > TOLERANCE)
            )
        )
        pairwise_correct = 0
        pairwise_total = 0
        for left_offset in range(len(rows) - 1):
            for right_offset in range(left_offset + 1, len(rows)):
                left = int(rows[left_offset])
                right = int(rows[right_offset])
                actual_difference = float(absolute[left] - absolute[right])
                if abs(actual_difference) <= TOLERANCE:
                    continue
                predicted_difference = float(scores[left] - scores[right])
                pairwise_total += 1
                pairwise_correct += int(
                    np.sign(predicted_difference) == np.sign(actual_difference)
                )
        selected_cost = float(absolute[selected])
        pp_cost = float(absolute[reference])
        oracle_cost = float(absolute[oracle])
        actual_advantage = pp_cost - selected_cost
        beneficial_switch_available = bool(
            np.any(
                (target_data.features[candidates, switch_index] > TOLERANCE)
                & (actual_advantages > TOLERANCE)
            )
        )
        records.append(
            {
                "group_id": str(group),
                "seed": int(seeds[rows[0]]),
                "scenario": str(scenarios[rows[0]]),
                "selected_row": selected,
                "reference_row": reference,
                "oracle_row": oracle,
                "selected_state": str(states[selected]),
                "reference_state": str(states[reference]),
                "oracle_state": str(states[oracle]),
                "different_action": selected != reference,
                "selected_switch": bool(
                    target_data.features[selected, switch_index] > TOLERANCE
                ),
                "beneficial_switch_available": beneficial_switch_available,
                "predicted_advantage": max(float(-scores[selected]), 0.0),
                "actual_advantage": actual_advantage,
                "selected_cost": selected_cost,
                "phase_pressure_cost": pp_cost,
                "oracle_cost": oracle_cost,
                "oracle_regret": selected_cost - oracle_cost,
                "harmful_intervention_regret": max(selected_cost - pp_cost, 0.0),
                "missed_benefit_regret": max(min(selected_cost, pp_cost) - oracle_cost, 0.0),
                "candidate_actual_advantages": actual_advantages.tolist(),
                "candidate_predicted_advantages": predicted_advantages.tolist(),
                "candidate_sign_correct": candidate_sign_correct,
                "candidate_sign_total": int(candidates.size),
                "pairwise_correct": pairwise_correct,
                "pairwise_total": pairwise_total,
            }
        )
    return records


def _costs_by_scenario(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scenarios = sorted({str(row["scenario"]) for row in records})
    return {
        scenario: float(
            np.mean(
                [
                    float(row["selected_cost"])
                    for row in records
                    if str(row["scenario"]) == scenario
                ]
            )
        )
        for scenario in scenarios
    }


def _pp_costs_by_scenario(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scenarios = sorted({str(row["scenario"]) for row in records})
    return {
        scenario: float(
            np.mean(
                [
                    float(row["phase_pressure_cost"])
                    for row in records
                    if str(row["scenario"]) == scenario
                ]
            )
        )
        for scenario in scenarios
    }


def _behavior_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overrides = [row for row in records if bool(row["different_action"])]
    actual = np.asarray([float(row["actual_advantage"]) for row in overrides])
    predicted = np.concatenate(
        [np.asarray(row["candidate_predicted_advantages"], dtype=float) for row in records]
    )
    candidate_actual = np.concatenate(
        [np.asarray(row["candidate_actual_advantages"], dtype=float) for row in records]
    )
    correlation = (
        float(np.corrcoef(predicted, candidate_actual)[0, 1])
        if float(np.std(predicted)) > TOLERANCE
        and float(np.std(candidate_actual)) > TOLERANCE
        else None
    )
    sign_correct = sum(int(row["candidate_sign_correct"]) for row in records)
    sign_total = sum(int(row["candidate_sign_total"]) for row in records)
    pairwise_correct = sum(int(row["pairwise_correct"]) for row in records)
    pairwise_total = sum(int(row["pairwise_total"]) for row in records)
    return {
        "groups": len(records),
        "overrides": len(overrides),
        "beneficial_overrides": int(np.count_nonzero(actual > TOLERANCE)),
        "harmful_overrides": int(np.count_nonzero(actual < -TOLERANCE)),
        "equal_overrides": int(np.count_nonzero(np.abs(actual) <= TOLERANCE)),
        "total_realized_gain": float(np.sum(np.maximum(actual, 0.0))),
        "total_realized_harm": float(np.sum(np.maximum(-actual, 0.0))),
        "mean_oracle_regret": float(
            np.mean([float(row["oracle_regret"]) for row in records])
        ),
        "mean_harmful_intervention_regret": float(
            np.mean([float(row["harmful_intervention_regret"]) for row in records])
        ),
        "mean_missed_benefit_regret": float(
            np.mean([float(row["missed_benefit_regret"]) for row in records])
        ),
        "oracle_actions_selected": int(
            np.count_nonzero(
                [float(row["oracle_regret"]) <= TOLERANCE for row in records]
            )
        ),
        "harmful_switch_overrides": sum(
            bool(row["different_action"])
            and bool(row["selected_switch"])
            and float(row["actual_advantage"]) < -TOLERANCE
            for row in records
        ),
        "missed_beneficial_switch_groups": sum(
            bool(row["beneficial_switch_available"])
            and not (
                bool(row["selected_switch"])
                and float(row["actual_advantage"]) > TOLERANCE
            )
            for row in records
        ),
        "candidate_benefit_sign_accuracy": float(sign_correct / sign_total),
        "groupwise_pairwise_ordering_accuracy": float(
            pairwise_correct / pairwise_total
        ),
        "predicted_realized_candidate_advantage_correlation": correlation,
    }


def _gate(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    demand_differences: Mapping[str, float],
) -> dict[str, Any]:
    spec = protocol["development_gate"]
    demand_checks = {
        scenario: float(value) <= float(spec["demand_mean_difference_maximum"])
        for scenario, value in demand_differences.items()
    }
    checks = {
        "mean_passed": float(comparison["mean_difference"])
        <= float(spec["mean_difference_maximum"]),
        "bootstrap_upper_passed": float(comparison["paired_bootstrap_95"][1])
        < float(spec["paired_bootstrap_upper_95_maximum"]),
        "seed_wins_passed": int(comparison["improved_seeds"])
        >= int(spec["minimum_improved_seeds"]),
        "every_demand_noninferior_passed": all(demand_checks.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "demand_checks": demand_checks,
        "specification": dict(spec),
    }


def run_ablation(
    *,
    protocol_path: Path,
    parent_result_path: Path,
    bank_path: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol = _read_json(protocol_path)
    validate_protocol(protocol)
    if output.exists():
        raise FileExistsError("refusing to overwrite V159 output")
    parent, data = _validate_inputs(
        protocol,
        parent_result_path=parent_result_path,
        bank_path=bank_path,
    )
    groups = np.asarray(data.metadata["action_group_ids"], dtype=str)
    row_seeds = np.asarray(data.metadata["row_seeds"], dtype=int)
    unique_groups = tuple(dict.fromkeys(groups.tolist()))
    group_seed = {}
    for group in unique_groups:
        values = np.unique(row_seeds[groups == group])
        if values.size != 1:
            raise ValueError("V159 action group crosses seeds")
        group_seed[group] = int(values[0])

    oof_records = {representation: [] for representation in REPRESENTATIONS}
    folds = []
    all_groups = set(unique_groups)
    for fold_index, heldout_seeds in enumerate(protocol["crossfit"]["folds"]):
        heldout_seed_set = set(map(int, heldout_seeds))
        heldout = {
            group for group, seed in group_seed.items() if seed in heldout_seed_set
        }
        training = all_groups - heldout
        if len(heldout) != 20 or len(training) != 80:
            raise ValueError("V159 whole-seed fold changed")
        train_data = v158._subset_groups(data, training)
        heldout_data = v158._subset_groups(data, heldout)
        diagnostics = {}
        for representation in REPRESENTATIONS:
            model, diagnostics[representation], _ = _fit_representation(
                protocol, train_data, representation
            )
            records = _policy_records(model, heldout_data)
            if len(records) != 20:
                raise ValueError("V159 OOF representation coverage changed")
            oof_records[representation].extend(records)
        folds.append(
            {
                "fold": fold_index,
                "heldout_seeds": list(map(int, heldout_seeds)),
                "training_group_count": len(training),
                "heldout_group_count": len(heldout),
                "fit_diagnostics": diagnostics,
            }
        )
    if any(len(records) != 100 for records in oof_records.values()):
        raise ValueError("V159 OOF coverage is incomplete")

    seed_costs = {
        representation: v158._seed_costs(records)
        for representation, records in oof_records.items()
    }
    pp_costs = v158._pp_seed_costs(oof_records["delta_target"])
    parent_target_costs = {
        int(seed): float(value)
        for seed, value in parent["oof"]["seed_costs"]["target_native"].items()
    }
    reproduction_differences = {
        seed: float(seed_costs["delta_target"][seed] - parent_target_costs[seed])
        for seed in sorted(parent_target_costs)
    }
    parent_reproduction = {
        "maximum_absolute_seed_cost_difference": float(
            max(map(abs, reproduction_differences.values()))
        ),
        "per_seed_difference": {
            str(seed): value for seed, value in reproduction_differences.items()
        },
    }
    parent_reproduction["passed"] = (
        parent_reproduction["maximum_absolute_seed_cost_difference"] <= TOLERANCE
    )
    if not parent_reproduction["passed"]:
        raise ValueError("V159 delta arm did not reproduce the V158 target arm")

    gate_spec = protocol["development_gate"]
    comparisons = {
        "endpoint_target_minus_delta_target": v158._paired_summary(
            seed_costs["endpoint_target"],
            seed_costs["delta_target"],
            draws=int(gate_spec["bootstrap_draws"]),
            seed=int(gate_spec["bootstrap_seed"]),
        ),
        "delta_target_minus_phase_pressure": v158._paired_summary(
            seed_costs["delta_target"],
            pp_costs,
            draws=int(gate_spec["bootstrap_draws"]),
            seed=int(gate_spec["bootstrap_seed"]),
        ),
        "endpoint_target_minus_phase_pressure": v158._paired_summary(
            seed_costs["endpoint_target"],
            pp_costs,
            draws=int(gate_spec["bootstrap_draws"]),
            seed=int(gate_spec["bootstrap_seed"]),
        ),
    }
    scenario_costs = {
        representation: _costs_by_scenario(records)
        for representation, records in oof_records.items()
    }
    pp_scenario_costs = _pp_costs_by_scenario(oof_records["delta_target"])
    demand_differences = {
        scenario: float(
            scenario_costs["endpoint_target"][scenario]
            - scenario_costs["delta_target"][scenario]
        )
        for scenario in protocol["development"]["scenarios"]
    }
    gate = _gate(
        protocol,
        comparisons["endpoint_target_minus_delta_target"],
        demand_differences,
    )
    behavior = {
        representation: _behavior_summary(records)
        for representation, records in oof_records.items()
    }

    output.mkdir(parents=True)
    oof_path = output / "oof_records.json"
    _write_json(oof_path, {"records": oof_records, "folds": folds})
    result = {
        "protocol": RESULT_PROTOCOL,
        "status": "PASS" if gate["passed"] else "OOF_GATE_FAIL",
        "scientific_status": (
            "endpoint_source_followup_authorized"
            if gate["passed"]
            else "direct_endpoint_hypothesis_closed"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol": v158.PROTOCOL,
        "parent_fit_result_protocol": v158.FIT_RESULT_PROTOCOL,
        "parent_execution_snapshot": parent.get("execution_snapshot"),
        "input_bank": {
            "path": str(bank_path.resolve()),
            "sha256": _sha256(bank_path),
            "size_bytes": bank_path.stat().st_size,
            "action_groups": 100,
            "candidate_rows": 800,
            "seeds": list(map(int, protocol["development"]["seeds"])),
            "scenarios": list(protocol["development"]["scenarios"]),
            "new_simulator_trajectories": 0,
        },
        "frozen_design": {
            "representations": dict(protocol["model"]["representations"]),
            "state_feature_names": list(PAIRWISE_CAUSAL_STATE_PARENTS),
            "action_difference_feature_names": list(ACTION_DIFFERENCE_FEATURES),
            "physical_midpoint_feature_names": list(PHYSICAL_MIDPOINT_FEATURES),
            "model_config": dict(protocol["model"]["config"]),
            "folds": [list(map(int, fold)) for fold in protocol["crossfit"]["folds"]],
            "threshold": 0.0,
            "no_stay_override_of_pp_switch": True,
            "target_base_prediction_only": True,
            "source_prediction_columns_used": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "parent_reproduction": parent_reproduction,
        "oof": {
            "fold_count": len(folds),
            "unit": "whole_seed",
            "coverage": {representation: len(rows) for representation, rows in oof_records.items()},
            "mean_costs": {
                **{
                    representation: float(np.mean(list(costs.values())))
                    for representation, costs in seed_costs.items()
                },
                "phase_pressure": float(np.mean(list(pp_costs.values()))),
            },
            "seed_costs": {
                **{
                    representation: {
                        str(seed): value for seed, value in costs.items()
                    }
                    for representation, costs in seed_costs.items()
                },
                "phase_pressure": {
                    str(seed): value for seed, value in pp_costs.items()
                },
            },
            "scenario_costs": {
                **scenario_costs,
                "phase_pressure": pp_scenario_costs,
            },
            "endpoint_minus_delta_by_scenario": demand_differences,
            "comparisons": comparisons,
            "gate": gate,
        },
        "behavior": behavior,
        "decision": {
            "followup_authorized": bool(gate["passed"]),
            "deployment_authorized": False,
            "reserve_executed": False,
        },
        "artifacts": {
            "oof_records": {
                "path": str(oof_path.resolve()),
                "sha256": _sha256(oof_path),
                "size_bytes": oof_path.stat().st_size,
            }
        },
        "claim_boundary": protocol["claim_boundary"],
        "elapsed_sec": time.monotonic() - started,
    }
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "protocol": RESULT_PROTOCOL,
                "status": result["status"],
                "followup_authorized": result["decision"]["followup_authorized"],
                "mean_costs": result["oof"]["mean_costs"],
                "endpoint_minus_delta": comparisons[
                    "endpoint_target_minus_delta_target"
                ],
                "demand_differences": demand_differences,
                "elapsed_sec": result["elapsed_sec"],
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v158-fit-result", type=Path, required=True)
    parser.add_argument("--native-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_ablation(
        protocol_path=args.protocol,
        parent_result_path=args.v158_fit_result,
        bank_path=args.native_bank,
        output=args.output,
    )
    return 0 if result["status"] in {"PASS", "OOF_GATE_FAIL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
