"""Seed-blocked offline screen for the equal-information arrival rule."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    parse_action_group_seed,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_scaling import action_group_range
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.generalized_pressure import (
    PHASE_PRESSURE_SPEC,
    pressure_scores_from_feature_matrix,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.movement_arrival_timeline import (
    build_movement_arrival_timeline_context,
)


RESULT_PROTOCOL = "tsc-v112-movement-arrival-offline-rule-screen-v1"
DEFAULT_COEFFICIENTS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
BOOTSTRAP_SEED = 112001


def _source_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1]
        / "traffic_signal"
        / "movement_arrival_timeline.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def score_arrival_rule_grid(
    *,
    base_scores: np.ndarray,
    arrival_scores: np.ndarray,
    target: np.ndarray,
    groups: Sequence[str],
    coefficients: Sequence[float],
) -> list[dict[str, Any]]:
    base_scores = np.asarray(base_scores, dtype=float).reshape(-1)
    arrival_scores = np.asarray(arrival_scores, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    group_values = np.asarray(groups, dtype=str)
    if not (
        base_scores.shape
        == arrival_scores.shape
        == target.shape
        == group_values.shape
    ):
        raise ValueError("arrival-rule inputs are not row aligned")
    if not np.isfinite(
        np.column_stack([base_scores, arrival_scores, target])
    ).all():
        raise ValueError("arrival-rule inputs contain non-finite values")
    rows: list[dict[str, Any]] = []
    for group in _ordered_unique(group_values.tolist()):
        indices = np.flatnonzero(group_values == group)
        if indices.size < 2:
            raise ValueError(f"arrival-rule action group has fewer than two actions: {group}")
        reference = int(indices[int(np.argmax(base_scores[indices]))])
        scale = action_group_range(target[indices])
        seed = parse_action_group_seed(group)
        scenario = str(group).split(":", 1)[0]
        for coefficient in coefficients:
            coefficient = float(coefficient)
            selected = int(
                indices[
                    int(
                        np.argmax(
                            base_scores[indices]
                            + coefficient * arrival_scores[indices]
                        )
                    )
                ]
            )
            raw_delta = float(target[selected] - target[reference])
            rows.append(
                {
                    "group_id": group,
                    "scenario": scenario,
                    "simulator_seed": seed,
                    "coefficient": coefficient,
                    "changed": bool(selected != reference),
                    "raw_delta": raw_delta,
                    "normalized_delta": raw_delta / scale,
                    "selected_regret": float(
                        (target[selected] - np.min(target[indices])) / scale
                    ),
                    "reference_regret": float(
                        (target[reference] - np.min(target[indices])) / scale
                    ),
                }
            )
    return rows


def _coefficient_summary(
    rows: Sequence[Mapping[str, Any]], coefficient: float
) -> dict[str, Any]:
    selected = [
        row for row in rows if float(row["coefficient"]) == float(coefficient)
    ]
    seeds = sorted({int(row["simulator_seed"]) for row in selected})
    seed_means = {
        str(seed): float(
            np.mean(
                [
                    float(row["normalized_delta"])
                    for row in selected
                    if int(row["simulator_seed"]) == seed
                ]
            )
        )
        for seed in seeds
    }
    return {
        "coefficient": float(coefficient),
        "group_count": len(selected),
        "seed_count": len(seeds),
        "changed_group_count": sum(bool(row["changed"]) for row in selected),
        "changed_group_fraction": float(
            np.mean([bool(row["changed"]) for row in selected])
        ),
        "improved_group_count": sum(
            float(row["normalized_delta"]) < 0.0 for row in selected
        ),
        "worsened_group_count": sum(
            float(row["normalized_delta"]) > 0.0 for row in selected
        ),
        "equal_seed_mean_normalized_delta": float(np.mean(list(seed_means.values()))),
        "mean_selected_regret": float(
            np.mean([float(row["selected_regret"]) for row in selected])
        ),
        "mean_reference_regret": float(
            np.mean([float(row["reference_regret"]) for row in selected])
        ),
        "seed_mean_normalized_delta": seed_means,
    }


def leave_one_seed_out_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    coefficients: Sequence[float],
    minimum_training_improvement: float,
) -> dict[str, Any]:
    coefficients = tuple(float(value) for value in coefficients)
    if not coefficients or coefficients[0] != 0.0 or len(set(coefficients)) != len(
        coefficients
    ):
        raise ValueError("arrival coefficients must be unique and begin with zero")
    seeds = sorted({int(row["simulator_seed"]) for row in rows})
    if len(seeds) < 2:
        raise ValueError("seed-blocked selection requires at least two seeds")
    heldout_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for heldout_seed in seeds:
        training = [
            row for row in rows if int(row["simulator_seed"]) != heldout_seed
        ]
        training_means = {
            coefficient: _coefficient_summary(training, coefficient)[
                "equal_seed_mean_normalized_delta"
            ]
            for coefficient in coefficients
        }
        nonzero = min(
            coefficients[1:],
            key=lambda value: (training_means[value], value),
            default=0.0,
        )
        selected_coefficient = (
            nonzero
            if nonzero != 0.0
            and training_means[nonzero]
            <= training_means[0.0] - float(minimum_training_improvement)
            else 0.0
        )
        validation = [
            dict(row)
            for row in rows
            if int(row["simulator_seed"]) == heldout_seed
            and float(row["coefficient"]) == selected_coefficient
        ]
        heldout_rows.extend(validation)
        selections.append(
            {
                "heldout_seed": heldout_seed,
                "selected_coefficient": selected_coefficient,
                "training_equal_seed_mean_normalized_delta": {
                    str(value): float(training_means[value]) for value in coefficients
                },
                "validation_group_count": len(validation),
                "validation_mean_normalized_delta": float(
                    np.mean([float(row["normalized_delta"]) for row in validation])
                ),
            }
        )
    heldout_seed_means = np.asarray(
        [row["validation_mean_normalized_delta"] for row in selections], dtype=float
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(
        heldout_seed_means,
        size=(10000, heldout_seed_means.size),
        replace=True,
    ).mean(axis=1)
    changed = sum(bool(row["changed"]) for row in heldout_rows)
    return {
        "protocol": "leave-one-simulator-seed-out-coefficient-selection-v1",
        "minimum_training_improvement": float(minimum_training_improvement),
        "seed_count": len(seeds),
        "group_count": len(heldout_rows),
        "nonzero_selected_seed_count": sum(
            float(row["selected_coefficient"]) > 0.0 for row in selections
        ),
        "changed_group_count": changed,
        "changed_group_fraction": changed / max(len(heldout_rows), 1),
        "equal_seed_mean_normalized_delta": float(np.mean(heldout_seed_means)),
        "paired_seed_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": 10000,
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "probability_nonnegative": float(np.mean(draws >= 0.0)),
        },
        "selections": selections,
    }


def run_screen(
    *,
    cache_root: Path,
    manifest_path: Path,
    conversion_root: Path,
    scenario: str,
    seeds: Sequence[int],
    collection_shards: int,
    target_name: str,
    coefficients: Sequence[float],
    cache_workers: int,
    minimum_training_improvement: float,
) -> dict[str, Any]:
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected = tuple(
        value for value in full_manifest.scenarios if value.scenario == scenario
    )
    if len(selected) != 1:
        raise ValueError("movement-arrival scenario is not unique in the manifest")
    manifest = replace(full_manifest, scenarios=selected)
    bank, cache_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=tuple(int(value) for value in seeds),
        collection_shards=int(collection_shards),
        workers=int(cache_workers),
    )
    dataset: MechanismDataset = bank[scenario]
    if (
        dataset.metadata.get("counterfactual_cost_mode") != "halted_queue"
        or target_name not in dataset.targets
    ):
        raise ValueError("movement-arrival screen requires waiting-aligned target cache")
    sumocfg = Path(conversion_root) / scenario / f"{scenario}.sumocfg"
    context = build_movement_arrival_timeline_context(sumocfg)
    row_tls = list(dataset.metadata.get("row_tls", ()))
    row_times = list(dataset.metadata.get("row_times", ()))
    candidate_states = list(dataset.metadata.get("candidate_states", ()))
    if len(row_tls) != dataset.size or len(row_times) != dataset.size or len(
        candidate_states
    ) != dataset.size:
        raise ValueError("waiting cache lacks aligned candidate metadata")
    arrival_scores = np.asarray(
        [
            context.arrival_pressure_score(tls, time_sec, candidate_state)
            for tls, time_sec, candidate_state in zip(
                row_tls, row_times, candidate_states, strict=True
            )
        ],
        dtype=float,
    )
    base_scores = pressure_scores_from_feature_matrix(
        dataset.features, dataset.feature_names, PHASE_PRESSURE_SPEC
    )
    rows = score_arrival_rule_grid(
        base_scores=base_scores,
        arrival_scores=arrival_scores,
        target=np.asarray(dataset.targets[target_name], dtype=float),
        groups=dataset.metadata["action_group_ids"],
        coefficients=coefficients,
    )
    grid = {
        str(float(value)): _coefficient_summary(rows, float(value))
        for value in coefficients
    }
    oof = leave_one_seed_out_selection(
        rows,
        coefficients=coefficients,
        minimum_training_improvement=minimum_training_improvement,
    )
    gate_passed = bool(
        oof["nonzero_selected_seed_count"] > 0
        and oof["equal_seed_mean_normalized_delta"] < 0.0
        and oof["paired_seed_bootstrap"]["ci95"][1] < 0.0
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "scientific_status": "disclosed_post-v111_development_gate_b_equal_information_control",
        "source_sha256": _source_sha256(),
        "scenario": scenario,
        "seeds": [int(value) for value in seeds],
        "collection_shards": int(collection_shards),
        "target_name": target_name,
        "target_estimand": "candidate_action_then_phase_pressure_waiting_aligned_rollout",
        "information_budget": (
            "target_static_network_route_schedule_plus_candidate_state_no_labels"
        ),
        "cache_audit": cache_audit,
        "movement_arrival_context": context.to_dict(),
        "coefficient_grid": [float(value) for value in coefficients],
        "grid": grid,
        "leave_one_seed_out": oof,
        "equal_information_rule_gate_passed": gate_passed,
        "decision": (
            "retain_arrival_rule_as_equal_information_baseline"
            if gate_passed
            else "retain_exact_phase_pressure_as_equal_information_rule"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--collection-shards", type=int, required=True)
    parser.add_argument("--target-name", default="prefix_mean_cost_450s")
    parser.add_argument("--coefficients", type=float, nargs="+", default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--cache-workers", type=int, default=32)
    parser.add_argument("--minimum-training-improvement", type=float, default=0.002)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_screen(
        cache_root=args.cache_root,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        scenario=args.scenario,
        seeds=args.seeds,
        collection_shards=args.collection_shards,
        target_name=args.target_name,
        coefficients=args.coefficients,
        cache_workers=args.cache_workers,
        minimum_training_improvement=args.minimum_training_improvement,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "decision": result["decision"],
                "oof_delta": result["leave_one_seed_out"][
                    "equal_seed_mean_normalized_delta"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
