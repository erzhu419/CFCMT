#!/usr/bin/env python3
"""Audit the frozen V93 source selector on disjoint confirmation seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from scripts.cluster.audit_tsc_source_city_component_screen import (
    SCENARIOS_BY_CITY,
    _load_rows,
)
from scripts.cluster.freeze_tsc_source_city_confirmation import PROTOCOL as FREEZE_PROTOCOL
from scripts.cluster.launch_tsc_source_city_component_screen import LAUNCH_PROTOCOL
from scripts.cluster.run_tsc_source_city_component_shard import (
    SCENARIOS,
    load_candidate_specs,
)


AUDIT_PROTOCOL = "tsc-v93-causal-source-selector-disjoint-confirmation-audit-v1"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 9_324_017
FINAL_METHOD = "frozen_selector"
COMPARATORS = (
    "target_only",
    "equal_all_sources",
    "h2oplus",
    "phase_pressure",
)
BASELINE_CONTRACTS = {
    "h2oplus": {
        "shard_protocol": "tsc-v92-external-frozen-policy-process-shard-v1",
        "row_policy": "h2oplus_style_dense_residual_mpc_v92_diagnostic",
    },
    "phase_pressure": {
        "shard_protocol": "tsc-v92-external-phase-pressure-process-shard-v1",
        "row_policy": "phase_pressure_v92_diagnostic",
    },
}


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (float(p_values[key]), key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, float(p_values[key]) * (total - rank))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _paired_summary(
    relative_deltas: Sequence[float],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values = np.asarray(relative_deltas, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired confirmation deltas are invalid")
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(
        0,
        values.size,
        size=(BOOTSTRAP_REPLICATES, values.size),
    )
    bootstrap_means = np.mean(values[indices], axis=1)
    if np.allclose(values, 0.0, rtol=0.0, atol=1e-15):
        statistic = 0.0
        p_value = 1.0
        exact_tie = True
    else:
        test = wilcoxon(
            values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
        exact_tie = False
    return {
        "n": int(values.size),
        "mean_relative_delta": float(np.mean(values)),
        "median_relative_delta": float(np.median(values)),
        "std_relative_delta": float(np.std(values, ddof=1)),
        "bootstrap_mean_ci95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "improved_seed_count": int(np.sum(values < 0.0)),
        "tied_seed_count": int(np.sum(np.isclose(values, 0.0, atol=1e-15))),
        "worsened_seed_count": int(np.sum(values > 0.0)),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_two_sided": p_value,
        "exact_tie": exact_tie,
        "relative_deltas": values.tolist(),
    }


def _load_baseline_rows(
    *,
    shards_root: Path,
    method: str,
    confirmation_seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    contract = BASELINE_CONTRACTS[method]
    paths = sorted(Path(shards_root).glob("shard_*.json"))
    if len(paths) != 6:
        raise ValueError(f"{method} baseline shard count changed")
    all_rows = []
    shard_hashes = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not (
            payload.get("protocol") == contract["shard_protocol"]
            and payload.get("passed") is True
            and int(payload.get("task_count", -1)) == len(payload.get("rows", ()))
        ):
            raise ValueError(f"invalid {method} baseline shard: {path}")
        all_rows.extend(dict(row) for row in payload["rows"])
        shard_hashes[path.name] = _sha256(path)
    allowed = set(int(value) for value in confirmation_seeds)
    rows = []
    for row in all_rows:
        if int(row["seed"]) not in allowed:
            continue
        if str(row.get("policy")) != contract["row_policy"]:
            raise ValueError(f"{method} baseline policy changed")
        normalized = dict(row)
        normalized["candidate_key"] = method
        rows.append(normalized)
    observed = {
        (str(row["scenario"]), int(row["seed"]))
        for row in rows
    }
    expected = {
        (scenario, int(seed)) for scenario in SCENARIOS for seed in confirmation_seeds
    }
    if len(rows) != len(observed) or observed != expected:
        raise ValueError(f"{method} baseline confirmation coverage failed")
    return rows, shard_hashes


def _aggregate_methods(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    methods: Sequence[str],
) -> dict[str, dict[int, dict[str, dict[str, float | int]]]]:
    by_identity = {
        (str(row["scenario"]), int(row["seed"]), str(row["candidate_key"])): row
        for row in rows
    }
    expected = {
        (scenario, int(seed), method)
        for scenario in SCENARIOS
        for seed in seeds
        for method in methods
    }
    if len(by_identity) != len(rows) or set(by_identity) != expected:
        raise ValueError("confirmation method matrix is incomplete")
    aggregate: dict[str, dict[int, dict[str, dict[str, float | int]]]] = {}
    for city, scenarios in SCENARIOS_BY_CITY.items():
        aggregate[city] = {}
        for seed in seeds:
            aggregate[city][int(seed)] = {}
            for method in methods:
                scenario_rows = [
                    by_identity[(scenario, int(seed), method)]
                    for scenario in scenarios
                ]
                metrics = [row["metrics"] for row in scenario_rows]
                aggregate[city][int(seed)][method] = {
                    "waiting": float(
                        np.mean(
                            [
                                float(row["mean_tripinfo_waiting_time"])
                                for row in metrics
                            ]
                        )
                    ),
                    "collision_incidents": int(
                        sum(int(row.get("collision_incidents", 0)) for row in metrics)
                    ),
                    "teleports": int(
                        sum(
                            int(row.get("starting_teleports", 0))
                            + int(row.get("ending_teleports", 0))
                            for row in metrics
                        )
                    ),
                }
    return aggregate


def build_confirmation_statistics(
    aggregate: Mapping[str, Mapping[int, Mapping[str, Mapping[str, float | int]]]],
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    raw_p = {}
    comparison_index = 0
    for comparator in COMPARATORS:
        city_deltas = {}
        city_payload = {}
        for city in SCENARIOS_BY_CITY:
            deltas = [
                (
                    float(aggregate[city][int(seed)][FINAL_METHOD]["waiting"])
                    - float(aggregate[city][int(seed)][comparator]["waiting"])
                )
                / max(float(aggregate[city][int(seed)][comparator]["waiting"]), 1e-9)
                for seed in seeds
            ]
            city_deltas[city] = deltas
            summary = _paired_summary(
                deltas,
                bootstrap_seed=BOOTSTRAP_SEED + comparison_index,
            )
            comparison_index += 1
            safety = {}
            for metric in ("collision_incidents", "teleports"):
                final_total = int(
                    sum(
                        int(aggregate[city][int(seed)][FINAL_METHOD][metric])
                        for seed in seeds
                    )
                )
                comparator_total = int(
                    sum(
                        int(aggregate[city][int(seed)][comparator][metric])
                        for seed in seeds
                    )
                )
                safety[metric] = {
                    "frozen_selector": final_total,
                    "comparator": comparator_total,
                    "delta": final_total - comparator_total,
                }
            summary["safety"] = safety
            city_payload[city] = summary
            raw_p[f"{comparator}:{city}"] = summary["wilcoxon_p_two_sided"]
        macro_deltas = [
            float(
                np.mean(
                    [city_deltas[city][index] for city in SCENARIOS_BY_CITY]
                )
            )
            for index in range(len(seeds))
        ]
        macro = _paired_summary(
            macro_deltas,
            bootstrap_seed=BOOTSTRAP_SEED + comparison_index,
        )
        comparison_index += 1
        raw_p[f"{comparator}:macro"] = macro["wilcoxon_p_two_sided"]
        comparisons[comparator] = {"by_city": city_payload, "macro": macro}
    adjusted = _holm_adjust(raw_p)
    for comparator, payload in comparisons.items():
        for city, summary in payload["by_city"].items():
            summary["wilcoxon_p_holm"] = adjusted[f"{comparator}:{city}"]
        payload["macro"]["wilcoxon_p_holm"] = adjusted[f"{comparator}:macro"]
    return {
        "final_method": FINAL_METHOD,
        "comparators": comparisons,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "multiplicity": "Holm adjustment over 12 two-sided Wilcoxon tests",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-spec", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--h2oplus-shards-root", type=Path, required=True)
    parser.add_argument("--phase-pressure-shards-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite confirmation audit: {args.out}")

    candidate_payload = json.loads(args.candidate_spec.read_text(encoding="utf-8"))
    candidate_sha = _sha256(args.candidate_spec)
    if not (
        candidate_payload.get("freeze_protocol") == FREEZE_PROTOCOL
        and candidate_payload.get("scientific_status")
        == "frozen-before-disjoint-seed-confirmation"
        and candidate_payload.get("analysis_contract", {}).get(
            "no_post_confirmation_retuning"
        )
        is True
    ):
        raise ValueError("confirmation freeze contract changed")
    candidates = load_candidate_specs(args.candidate_spec)
    candidate_keys = tuple(str(row["key"]) for row in candidates)
    if candidate_keys != ("target_only", "equal_all_sources", "frozen_selector"):
        raise ValueError("confirmation candidate order changed")
    seeds = tuple(int(value) for value in candidate_payload["confirmation_seeds"])
    if set(seeds) & set(int(value) for value in candidate_payload["development_seeds"]):
        raise ValueError("development and confirmation seeds overlap")

    launch_path = args.results_root / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if not (
        launch.get("protocol") == LAUNCH_PROTOCOL
        and launch.get("complete") is True
        and launch.get("candidate_spec_sha256") == candidate_sha
        and tuple(int(value) for value in launch.get("seeds", ())) == seeds
    ):
        raise ValueError("confirmation launch contract changed")
    shard_count = len(tuple(launch.get("shard_indices", ())))
    component_rows = _load_rows(
        shards_root=args.results_root / "_shards",
        expected_shard_count=shard_count,
        candidate_spec_sha256=candidate_sha,
    )
    expected_component = len(SCENARIOS) * len(seeds) * len(candidate_keys)
    if len(component_rows) != expected_component:
        raise ValueError("confirmation component matrix size changed")
    candidate_by_key = {str(row["key"]): row for row in candidates}
    for row in component_rows:
        expected_guard = candidate_by_key[str(row["candidate_key"])][
            "guard_by_city"
        ][str(row["city"])]
        if row.get("guard_config") != expected_guard:
            raise ValueError("confirmation row used the wrong city guard")

    h2oplus_rows, h2oplus_hashes = _load_baseline_rows(
        shards_root=args.h2oplus_shards_root,
        method="h2oplus",
        confirmation_seeds=seeds,
    )
    phase_rows, phase_hashes = _load_baseline_rows(
        shards_root=args.phase_pressure_shards_root,
        method="phase_pressure",
        confirmation_seeds=seeds,
    )
    rows = [*component_rows, *h2oplus_rows, *phase_rows]
    methods = (*candidate_keys, "h2oplus", "phase_pressure")
    aggregate = _aggregate_methods(rows, seeds=seeds, methods=methods)
    statistics = build_confirmation_statistics(aggregate, seeds=seeds)

    fallback_exact = all(
        aggregate["los_angeles"][seed]["frozen_selector"]
        == aggregate["los_angeles"][seed]["target_only"]
        for seed in seeds
    )
    target = statistics["comparators"]["target_only"]
    equal = statistics["comparators"]["equal_all_sources"]
    conclusions = {
        "jinan_source_transfer_confirmed": bool(
            target["by_city"]["jinan"]["bootstrap_mean_ci95"][1] < 0.0
            and target["by_city"]["jinan"]["wilcoxon_p_holm"] < 0.05
            and target["by_city"]["jinan"]["safety"]["collision_incidents"][
                "delta"
            ]
            <= 0
            and target["by_city"]["jinan"]["safety"]["teleports"]["delta"]
            <= 0
        ),
        "los_angeles_fallback_exact": bool(fallback_exact),
        "macro_source_transfer_confirmed": bool(
            target["macro"]["bootstrap_mean_ci95"][1] < 0.0
            and target["macro"]["wilcoxon_p_holm"] < 0.05
        ),
        "negative_transfer_suppression_confirmed": bool(
            equal["macro"]["bootstrap_mean_ci95"][1] < 0.0
            and equal["macro"]["wilcoxon_p_holm"] < 0.05
        ),
    }
    payload = {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "disjoint-seed-confirmation",
        "passed_integrity_audit": True,
        "candidate_spec_sha256": candidate_sha,
        "launch_sha256": _sha256(launch_path),
        "confirmation_seed_count": len(seeds),
        "scenario_count": len(SCENARIOS),
        "method_count": len(methods),
        "matrix_size": len(rows),
        "baseline_shard_sha256": {
            "h2oplus": h2oplus_hashes,
            "phase_pressure": phase_hashes,
        },
        "statistics": statistics,
        "confirmation_conclusions": conclusions,
        "claim_boundary": (
            "The confirmation supports only the frozen LA/Jinan source-selector "
            "contract. New-city transportability remains a separate experiment."
        ),
    }
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "matrix_size": payload["matrix_size"],
                "confirmation_conclusions": conclusions,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
