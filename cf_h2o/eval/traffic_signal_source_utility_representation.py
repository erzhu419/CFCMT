"""Test whether deploy-observable right-of-way context identifies source utility."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cf_h2o.eval.traffic_signal_right_of_way_signature import (
    RESULT_PROTOCOL as SIGNATURE_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v150b-source-utility-representation-diagnostic-v1"
V144_PROTOCOL = "tsc-v144-multicity-source-compatibility-inventory-target-v1"
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)

BASE_CONTEXT_FEATURES = (
    "demand_per_tls",
    "demand_per_lane",
    "candidate_count_mean_norm",
    "green_ratio_mean",
    "phase_duration_mean_norm",
    "incoming_lane_cv",
    "controlled_link_cv",
    "signal_degree_mean_norm",
    "signal_graph_density",
    "signal_graph_reciprocity",
    "route_entropy_norm",
    "route_max_share",
)

BASE_DYNAMIC_FEATURES = (
    "green_link_ratio",
    "green_lane_ratio",
    "current_phase_overlap",
    "current_green_elapsed_norm",
    "queue_concentration",
    "graph_in_degree_norm",
    "graph_out_degree_norm",
    "graph_focal_to_neighbor_q_ratio",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _selected(values: Sequence[float], names: Sequence[str], selected: Sequence[str]) -> list[float]:
    lookup = {str(name): float(value) for name, value in zip(names, values, strict=True)}
    missing = [name for name in selected if name not in lookup]
    if missing:
        raise KeyError(f"city signature is missing features: {missing}")
    return [lookup[name] for name in selected]


def _base_city_vectors(
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    reference = next(iter(results.values())).get("city_covariate_signatures", {})
    cities = tuple(sorted(results))
    if set(reference) != set(cities):
        raise ValueError("V144 city covariate coverage changed")
    for row in results.values():
        if row.get("city_covariate_signatures") != reference:
            raise ValueError("V144 targets do not share city covariate signatures")
    feature_names = (
        *(f"context_mean:{name}" for name in BASE_CONTEXT_FEATURES),
        *(f"feature_mean:{name}" for name in BASE_DYNAMIC_FEATURES),
        *(f"feature_std:{name}" for name in BASE_DYNAMIC_FEATURES),
    )
    vectors = {}
    for city in cities:
        signature = reference[city]
        values = [
            *_selected(
                signature["context_mean"],
                signature["context_names"],
                BASE_CONTEXT_FEATURES,
            ),
            *_selected(
                signature["feature_mean"],
                signature["feature_names"],
                BASE_DYNAMIC_FEATURES,
            ),
            *_selected(
                signature["feature_std"],
                signature["feature_names"],
                BASE_DYNAMIC_FEATURES,
            ),
        ]
        vectors[city] = np.asarray(values, dtype=float)
    return vectors, tuple(feature_names)


def _right_of_way_city_vectors(
    signature: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    if signature.get("protocol") != SIGNATURE_PROTOCOL:
        raise ValueError("V150B right-of-way signature protocol changed")
    names = tuple(str(value) for value in signature.get("feature_names", ()))
    cities = signature.get("city_signatures", {})
    if not names or not cities:
        raise ValueError("V150B right-of-way signature is incomplete")
    feature_names = (
        *(f"right_of_way_mean:{name}" for name in names),
        *(f"right_of_way_std:{name}" for name in names),
    )
    vectors = {
        str(city): np.asarray([*row["mean"], *row["std"]], dtype=float)
        for city, row in cities.items()
    }
    if any(vector.shape != (len(feature_names),) for vector in vectors.values()):
        raise ValueError("V150B right-of-way city vector width changed")
    return vectors, tuple(feature_names)


def _effects(
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], list[float]]]:
    pair_means: dict[tuple[str, str], float] = {}
    pair_seeds: dict[tuple[str, str], list[float]] = {}
    cities = set(results)
    for target, row in results.items():
        if row.get("protocol") != V144_PROTOCOL or row.get("target_city") != target:
            raise ValueError(f"invalid V144 target result: {target}")
        sources = row.get("source_arm_summaries", {})
        if set(sources) != cities - {target}:
            raise ValueError(f"V144 source coverage changed for {target}")
        target_values = row["arm_summaries"]["target_only_causal"]["seed_values"]
        for source, source_row in sources.items():
            source_values = source_row["seed_values"]
            if set(source_values) != set(target_values):
                raise ValueError(f"V144 seed alignment changed for {target}/{source}")
            values = [
                float(source_values[seed]) - float(target_values[seed])
                for seed in sorted(target_values, key=int)
            ]
            pair_seeds[(target, source)] = values
            pair_means[(target, source)] = float(np.mean(values))
    return pair_means, pair_seeds


def _pair_vector(
    city_vectors: Mapping[str, np.ndarray], target: str, source: str
) -> np.ndarray:
    target_values = np.asarray(city_vectors[target], dtype=float)
    source_values = np.asarray(city_vectors[source], dtype=float)
    return np.concatenate(
        (target_values, source_values - target_values, np.abs(source_values - target_values))
    )


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=float(alpha), fit_intercept=True).fit(
        scaler.transform(train_x), train_y
    )
    return np.asarray(model.predict(scaler.transform(test_x)), dtype=float)


def _inner_alpha(
    *,
    cities: Sequence[str],
    vectors: Mapping[str, np.ndarray],
    effects: Mapping[tuple[str, str], float],
) -> float:
    scores = []
    for alpha in RIDGE_ALPHAS:
        errors = []
        for heldout in cities:
            train_pairs = [
                pair
                for pair in effects
                if heldout not in pair and pair[0] in cities and pair[1] in cities
            ]
            test_pairs = [
                (heldout, source) for source in cities if source != heldout
            ]
            if not train_pairs or not test_pairs:
                raise ValueError("nested city exclusion produced an empty fold")
            predictions = _fit_predict(
                np.vstack([_pair_vector(vectors, *pair) for pair in train_pairs]),
                np.asarray([effects[pair] for pair in train_pairs], dtype=float),
                np.vstack([_pair_vector(vectors, *pair) for pair in test_pairs]),
                alpha=alpha,
            )
            errors.extend(
                abs(float(prediction) - effects[pair])
                for pair, prediction in zip(test_pairs, predictions, strict=True)
            )
        scores.append((float(np.mean(errors)), float(alpha)))
    return min(scores)[1]


def _evaluate_representation(
    *,
    name: str,
    vectors: Mapping[str, np.ndarray],
    effects: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    cities = tuple(sorted(vectors))
    predictions: dict[tuple[str, str], float] = {}
    folds = []
    for heldout in cities:
        training_cities = tuple(city for city in cities if city != heldout)
        train_pairs = [pair for pair in effects if heldout not in pair]
        test_pairs = [(heldout, source) for source in training_cities]
        alpha = _inner_alpha(
            cities=training_cities, vectors=vectors, effects=effects
        )
        predicted = _fit_predict(
            np.vstack([_pair_vector(vectors, *pair) for pair in train_pairs]),
            np.asarray([effects[pair] for pair in train_pairs], dtype=float),
            np.vstack([_pair_vector(vectors, *pair) for pair in test_pairs]),
            alpha=alpha,
        )
        for pair, value in zip(test_pairs, predicted, strict=True):
            predictions[pair] = float(value)
        folds.append(
            {
                "heldout_city": heldout,
                "heldout_city_absent_as_source_and_target": True,
                "training_city_count": len(training_cities),
                "training_pair_count": len(train_pairs),
                "evaluation_pair_count": len(test_pairs),
                "selected_ridge_alpha": alpha,
            }
        )
    ordered_pairs = sorted(effects)
    actual = np.asarray([effects[pair] for pair in ordered_pairs], dtype=float)
    predicted = np.asarray([predictions[pair] for pair in ordered_pairs], dtype=float)
    city_rows = {}
    selected_values = []
    null_aware_values = []
    rank_values = []
    for target in cities:
        sources = tuple(city for city in cities if city != target)
        target_actual = np.asarray([effects[(target, source)] for source in sources])
        target_predicted = np.asarray(
            [predictions[(target, source)] for source in sources]
        )
        rank = float(spearmanr(target_actual, target_predicted).statistic)
        if not np.isfinite(rank):
            rank = 0.0
        selected_index = int(np.argmin(target_predicted))
        selected_source = sources[selected_index]
        selected_effect = float(target_actual[selected_index])
        null_source = selected_source if target_predicted[selected_index] < 0.0 else None
        null_effect = selected_effect if null_source is not None else 0.0
        selected_values.append(selected_effect)
        null_aware_values.append(null_effect)
        rank_values.append(rank)
        city_rows[target] = {
            "source_rank_spearman": rank,
            "selected_source": selected_source,
            "selected_predicted_effect": float(target_predicted[selected_index]),
            "selected_actual_effect": selected_effect,
            "null_aware_selected_source": null_source,
            "null_aware_actual_effect": null_effect,
            "source_predictions": {
                source: {
                    "predicted_effect": float(target_predicted[index]),
                    "actual_effect": float(target_actual[index]),
                }
                for index, source in enumerate(sources)
            },
        }
    correlation = float(np.corrcoef(actual, predicted)[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "name": name,
        "feature_width": int(next(iter(vectors.values())).size),
        "pair_design_width": int(3 * next(iter(vectors.values())).size),
        "outer_fold_count": len(cities),
        "pair_count": len(ordered_pairs),
        "pair_mae": float(np.mean(np.abs(predicted - actual))),
        "pair_pearson": correlation,
        "pair_sign_accuracy": float(np.mean((predicted < 0.0) == (actual < 0.0))),
        "mean_source_rank_spearman": float(np.mean(rank_values)),
        "median_source_rank_spearman": float(np.median(rank_values)),
        "forced_selected_mean_effect": float(np.mean(selected_values)),
        "forced_selected_improving_city_count": int(
            np.count_nonzero(np.asarray(selected_values) < 0.0)
        ),
        "null_aware_mean_effect": float(np.mean(null_aware_values)),
        "null_aware_improving_city_count": int(
            np.count_nonzero(np.asarray(null_aware_values) < 0.0)
        ),
        "folds": folds,
        "cities": city_rows,
    }


def run_source_utility_representation_diagnostic(
    *,
    v144_results: Sequence[Mapping[str, Any]],
    right_of_way_signature: Mapping[str, Any],
) -> dict[str, Any]:
    by_target = {str(row.get("target_city")): dict(row) for row in v144_results}
    if len(by_target) != 7:
        raise ValueError("V150B requires exactly seven V144 target results")
    base_vectors, base_names = _base_city_vectors(by_target)
    right_vectors, right_names = _right_of_way_city_vectors(
        right_of_way_signature
    )
    if set(base_vectors) != set(right_vectors):
        raise ValueError("base and right-of-way city coverage differ")
    effects, seed_effects = _effects(by_target)
    enriched_vectors = {
        city: np.concatenate((base_vectors[city], right_vectors[city]))
        for city in base_vectors
    }
    baseline = _evaluate_representation(
        name="existing_deploy_observable_covariates",
        vectors=base_vectors,
        effects=effects,
    )
    enriched = _evaluate_representation(
        name="existing_plus_right_of_way_and_neighbor_execution",
        vectors=enriched_vectors,
        effects=effects,
    )
    cities = tuple(sorted(base_vectors))
    placebos = []
    for shift in range(1, len(cities)):
        permuted = {
            city: right_vectors[cities[(index + shift) % len(cities)]]
            for index, city in enumerate(cities)
        }
        vectors = {
            city: np.concatenate((base_vectors[city], permuted[city]))
            for city in cities
        }
        placebos.append(
            _evaluate_representation(
                name=f"cyclic_right_of_way_placebo_shift_{shift}",
                vectors=vectors,
                effects=effects,
            )
        )
    placebo_rank = float(
        np.median([row["mean_source_rank_spearman"] for row in placebos])
    )
    gate_checks = {
        "lower_pair_mae_than_existing": (
            enriched["pair_mae"] < baseline["pair_mae"]
        ),
        "source_rank_gain_at_least_0_10": (
            enriched["mean_source_rank_spearman"]
            >= baseline["mean_source_rank_spearman"] + 0.10
        ),
        "source_rank_above_placebo_median_by_0_10": (
            enriched["mean_source_rank_spearman"] >= placebo_rank + 0.10
        ),
        "null_aware_mean_improvement_at_least_0_0005": (
            enriched["null_aware_mean_effect"] <= -0.0005
        ),
        "null_aware_improves_at_least_five_cities": (
            enriched["null_aware_improving_city_count"] >= 5
        ),
    }
    passed = all(gate_checks.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "post-v150a-seven-city-development-diagnostic",
        "estimand": "v144_source_arm_minus_same_architecture_target_only_normalized_cost",
        "information_boundary": {
            "outer_evaluation_city_labels_used_for_fit": False,
            "outer_evaluation_city_used_as_source": False,
            "right_of_way_inputs": (
                "network_topology_candidate_states_and_same-time-observed-neighbor_"
                "execution_only"
            ),
            "future_or_outcome_features_used": False,
            "v144_outcomes_used_as_training_labels_on_other_cities": True,
            "independent_confirmation": False,
        },
        "city_groups": list(cities),
        "pair_count": len(effects),
        "seed_effects_per_pair": {
            f"{target}->{source}": values
            for (target, source), values in sorted(seed_effects.items())
        },
        "representations": {
            "existing": baseline,
            "right_of_way_enriched": enriched,
        },
        "matched_cyclic_placebos": placebos,
        "placebo_median_mean_source_rank_spearman": placebo_rank,
        "feature_schema": {
            "existing_city_features": list(base_names),
            "added_city_features": list(right_names),
            "pair_transform": "target_concat_source_minus_target_concat_absolute_difference",
            "ridge_alphas_nested_selected": list(RIDGE_ALPHAS),
        },
        "development_gate": {
            "checks": gate_checks,
            "passed": passed,
            "decision": (
                "promote_right_of_way_representation_to_mechanism_prior_development"
                if passed
                else "do_not_promote_right_of_way_representation"
            ),
        },
        "claim_boundary": (
            "V150B tests whether a fixed low-complexity representation predicts "
            "historical V144 source utility under city exclusion. It is neither "
            "closed-loop evidence nor fresh-city confirmation."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v144-results", nargs=7, type=Path, required=True)
    parser.add_argument("--right-of-way-signature", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150B result: {args.out}")
    result = run_source_utility_representation_diagnostic(
        v144_results=[_read_json(path) for path in args.v144_results],
        right_of_way_signature=_read_json(args.right_of_way_signature),
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
