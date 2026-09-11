"""Diagnose whether V144 source benefits repeat across evaluation seeds.

This module is deliberately diagnostic.  It uses outcomes from the other
evaluation seeds to choose a source and therefore does not define a deployable
target-city gate.  Its purpose is to distinguish reproducible source headroom
from a single-seed descriptive oracle before collecting new development data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_multicity_source_compatibility_inventory import (
    RESULT_PROTOCOL as V144_TARGET_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
    TARGET_BUDGET,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v150a-source-benefit-repeatability-diagnostic-v1"
EXPECTED_SEED_COUNT = 3


def _read_result(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V150A input is not a JSON object: {path}")
    return value


def _seed_values(
    summary: Mapping[str, Any],
    *,
    expected_seeds: tuple[str, ...] | None = None,
) -> dict[str, float]:
    raw = summary.get("seed_values")
    if not isinstance(raw, Mapping):
        raise ValueError("V150A requires per-seed policy values")
    values = {str(seed): float(value) for seed, value in raw.items()}
    seeds = tuple(sorted(values))
    if (
        len(seeds) != EXPECTED_SEED_COUNT
        or any(not np.isfinite(value) for value in values.values())
        or (expected_seeds is not None and seeds != expected_seeds)
    ):
        raise ValueError("V150A seed-value coverage changed")
    return values


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError("rank input must be a nonempty finite vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _sign_agreement(values: Sequence[float]) -> float:
    signs = np.sign(np.asarray(values, dtype=float))
    pairs = [
        float(signs[left] == signs[right])
        for left in range(signs.size)
        for right in range(left + 1, signs.size)
    ]
    return float(np.mean(pairs)) if pairs else 1.0


def _effect_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError("effect summary requires a nonempty finite vector")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "improving_fraction": float(np.mean(array < 0.0)),
        "nondegrading_fraction": float(np.mean(array <= 0.0)),
    }


def _target_repeatability(row: Mapping[str, Any]) -> dict[str, Any]:
    target = str(row.get("target_city"))
    expected_sources = tuple(
        sorted(set(EXPECTED_CITY_GROUPS) - {target})
    )
    if (
        row.get("protocol") != V144_TARGET_PROTOCOL
        or int(row.get("target_budget", -1)) != TARGET_BUDGET
        or tuple(sorted(str(value) for value in row.get("source_city_groups", ())))
        != expected_sources
        or set(row.get("source_arm_summaries", {})) != set(expected_sources)
        or set(row.get("source_placebo_arm_summaries", {}))
        != set(expected_sources)
        or row.get("information_budget", {}).get(
            "evaluation_groups_used_for_fit_or_selection"
        )
        is not False
    ):
        raise ValueError(f"{target}: V150A input boundary changed")

    target_values = _seed_values(
        row["arm_summaries"]["target_only_causal"]
    )
    seeds = tuple(sorted(target_values))
    source_effects: dict[str, dict[str, float]] = {}
    placebo_effects: dict[str, dict[str, float]] = {}
    source_rows: dict[str, Any] = {}
    for source in expected_sources:
        source_values = _seed_values(
            row["source_arm_summaries"][source],
            expected_seeds=seeds,
        )
        placebo_values = _seed_values(
            row["source_placebo_arm_summaries"][source],
            expected_seeds=seeds,
        )
        effects = {
            seed: source_values[seed] - target_values[seed] for seed in seeds
        }
        placebo = {
            seed: source_values[seed] - placebo_values[seed] for seed in seeds
        }
        source_effects[source] = effects
        placebo_effects[source] = placebo
        values = [effects[seed] for seed in seeds]
        source_rows[source] = {
            "seed_effects_vs_target_only": effects,
            "seed_effects_vs_matched_placebo": placebo,
            "effect_vs_target_only": _effect_summary(values),
            "seed_sign_agreement": _sign_agreement(values),
            "same_strict_sign_all_seeds": bool(
                np.all(np.asarray(values) < 0.0)
                or np.all(np.asarray(values) > 0.0)
            ),
        }

    rank_correlations = []
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            correlation = _spearman(
                [source_effects[source][left_seed] for source in expected_sources],
                [source_effects[source][right_seed] for source in expected_sources],
            )
            rank_correlations.append(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "spearman": correlation,
                }
            )

    forced_rows = []
    null_aware_rows = []
    for heldout_seed in seeds:
        training_seeds = tuple(seed for seed in seeds if seed != heldout_seed)
        training_means = {
            source: float(
                np.mean(
                    [
                        source_effects[source][seed]
                        for seed in training_seeds
                    ]
                )
            )
            for source in expected_sources
        }
        selected = min(
            expected_sources,
            key=lambda source: (training_means[source], source),
        )
        heldout_target_effect = float(
            source_effects[selected][heldout_seed]
        )
        heldout_placebo_effect = float(
            placebo_effects[selected][heldout_seed]
        )
        common = {
            "heldout_seed": heldout_seed,
            "training_seeds": list(training_seeds),
            "selected_source": selected,
            "training_mean_effect": training_means[selected],
            "heldout_effect_vs_target_only": heldout_target_effect,
            "heldout_effect_vs_matched_placebo": heldout_placebo_effect,
        }
        forced_rows.append(common)
        admitted = training_means[selected] < 0.0
        null_aware_rows.append(
            {
                **common,
                "source_admitted": bool(admitted),
                "effective_source": selected if admitted else None,
                "heldout_effect_vs_target_only": (
                    heldout_target_effect if admitted else 0.0
                ),
                "heldout_effect_vs_matched_placebo": (
                    heldout_placebo_effect if admitted else 0.0
                ),
            }
        )

    return {
        "target_city": target,
        "evaluation_seeds": list(seeds),
        "sources": source_rows,
        "rank_correlations": rank_correlations,
        "mean_seed_rank_correlation": float(
            np.mean([row["spearman"] for row in rank_correlations])
        ),
        "forced_source_leave_one_seed_out": forced_rows,
        "source_null_aware_leave_one_seed_out": null_aware_rows,
    }


def _aggregate_cross_validated_rows(
    targets: Mapping[str, Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    all_target_effects = []
    all_placebo_effects = []
    city_target_effects = {}
    city_placebo_effects = {}
    for target in EXPECTED_CITY_GROUPS:
        rows = list(targets[target][key])
        target_values = [
            float(row["heldout_effect_vs_target_only"]) for row in rows
        ]
        placebo_values = [
            float(row["heldout_effect_vs_matched_placebo"]) for row in rows
        ]
        all_target_effects.extend(target_values)
        all_placebo_effects.extend(placebo_values)
        city_target_effects[target] = float(np.mean(target_values))
        city_placebo_effects[target] = float(np.mean(placebo_values))
    return {
        "effect_vs_target_only": _effect_summary(all_target_effects),
        "effect_vs_matched_placebo": _effect_summary(all_placebo_effects),
        "city_mean_effects_vs_target_only": city_target_effects,
        "city_mean_effects_vs_matched_placebo": city_placebo_effects,
        "improving_city_count_vs_target_only": int(
            sum(value < 0.0 for value in city_target_effects.values())
        ),
        "improving_city_count_vs_matched_placebo": int(
            sum(value < 0.0 for value in city_placebo_effects.values())
        ),
    }


def aggregate_source_benefit_repeatability(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    by_target = {str(row.get("target_city")): row for row in rows}
    if (
        len(rows) != len(EXPECTED_CITY_GROUPS)
        or set(by_target) != set(EXPECTED_CITY_GROUPS)
    ):
        raise ValueError("V150A requires exactly one V144 result per target city")
    targets = {
        target: _target_repeatability(by_target[target])
        for target in EXPECTED_CITY_GROUPS
    }
    rank_correlations = [
        float(row["spearman"])
        for target in targets.values()
        for row in target["rank_correlations"]
    ]
    sign_agreements = [
        float(source["seed_sign_agreement"])
        for target in targets.values()
        for source in target["sources"].values()
    ]
    stable_pairs = [
        bool(source["same_strict_sign_all_seeds"])
        for target in targets.values()
        for source in target["sources"].values()
    ]
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "post-v148-development-diagnostic-not-deployment-or-confirmation"
        ),
        "predecessor_protocol": V144_TARGET_PROTOCOL,
        "target_budget": TARGET_BUDGET,
        "target_city_groups": list(EXPECTED_CITY_GROUPS),
        "evaluation_seed_count_per_city": EXPECTED_SEED_COUNT,
        "target_diagnostics": targets,
        "repeatability_summary": {
            "mean_within_target_source_rank_correlation": float(
                np.mean(rank_correlations)
            ),
            "median_within_target_source_rank_correlation": float(
                np.median(rank_correlations)
            ),
            "mean_ordered_pair_seed_sign_agreement": float(
                np.mean(sign_agreements)
            ),
            "fully_sign_stable_ordered_pair_fraction": float(
                np.mean(stable_pairs)
            ),
            "forced_source_leave_one_seed_out": (
                _aggregate_cross_validated_rows(
                    targets, "forced_source_leave_one_seed_out"
                )
            ),
            "source_null_aware_leave_one_seed_out": (
                _aggregate_cross_validated_rows(
                    targets, "source_null_aware_leave_one_seed_out"
                )
            ),
        },
        "information_boundary": {
            "evaluation_outcomes_used_for_cross_validated_source_choice": True,
            "heldout_seed_outcome_used_for_its_own_source_choice": False,
            "deployable_target_gate": False,
            "fresh_city_confirmation": False,
        },
        "claim_boundary": (
            "V150A measures whether V144 source effects repeat across the three "
            "existing evaluation seeds. Source choices use outcomes from the "
            "other evaluation seeds, information unavailable to a deployed B25 "
            "target. The result may justify fresh development collection but "
            "cannot establish a deployable source-selection method."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150A result: {args.out}")
    result = aggregate_source_benefit_repeatability(
        [_read_result(path) for path in args.results]
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "DONE",
                "protocol": result["protocol"],
                "repeatability_summary": result["repeatability_summary"],
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
