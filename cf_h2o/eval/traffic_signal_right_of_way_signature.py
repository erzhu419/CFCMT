"""Build label-free city signatures for the V150B representation diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.right_of_way_context import (
    RIGHT_OF_WAY_CONTEXT_FEATURES,
    augment_dataset_with_right_of_way_context,
    net_file_from_sumocfg,
    read_network_right_of_way_context,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v150b-label-free-right-of-way-city-signature-v1"


def _read_manifest(path: Path) -> tuple[dict[str, str], dict[str, Path]]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("scenarios", ())
    city_by_scenario: dict[str, str] = {}
    sumocfg_by_scenario: dict[str, Path] = {}
    for row in rows:
        scenario = str(row["scenario"])
        city = str(row["city_group"])
        sumocfg = (path.parent / str(row["sumocfg"])).resolve()
        if scenario in city_by_scenario:
            raise ValueError(f"duplicate scenario in source manifest: {scenario}")
        if not sumocfg.is_file():
            raise FileNotFoundError(sumocfg)
        city_by_scenario[scenario] = city
        sumocfg_by_scenario[scenario] = sumocfg
    if not city_by_scenario:
        raise ValueError("source manifest contains no scenarios")
    return city_by_scenario, sumocfg_by_scenario


def _load_observable_dataset(path: Path) -> MechanismDataset:
    """Load only deploy-observable arrays; target and prior arrays stay unread."""

    with np.load(path, allow_pickle=False) as archive:
        feature_names = tuple(
            str(value) for value in archive["feature_names"].tolist()
        )
        features = np.asarray(archive["features"], dtype=float)
        context_names = tuple(
            str(value) for value in archive["context_names"].tolist()
        )
        context = np.asarray(archive["context"], dtype=float)
        domains = np.asarray(archive["domains"])
        metadata = json.loads(str(archive["metadata_json"].item()))
    return MechanismDataset(
        feature_names=feature_names,
        features=features,
        context_names=context_names,
        context=context,
        priors={},
        targets={},
        domains=domains,
        metadata=metadata,
    )


def _summary(matrix: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("right-of-way signature requires at least one row")
    if not np.all(np.isfinite(values)):
        raise ValueError("right-of-way signature contains non-finite values")
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "q25": np.quantile(values, 0.25, axis=0).tolist(),
        "q75": np.quantile(values, 0.75, axis=0).tolist(),
    }


def build_right_of_way_signatures(
    *, source_manifest: Path, source_cache_root: Path
) -> dict[str, Any]:
    city_by_scenario, sumocfg_by_scenario = _read_manifest(source_manifest)
    cache_paths = sorted(Path(source_cache_root).glob("*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"no cache archives under {source_cache_root}")
    contexts = {
        scenario: read_network_right_of_way_context(
            net_file_from_sumocfg(sumocfg)
        )
        for scenario, sumocfg in sumocfg_by_scenario.items()
    }
    rows_by_scenario: dict[str, list[np.ndarray]] = defaultdict(list)
    files_by_scenario: dict[str, int] = defaultdict(int)
    empty_files_by_scenario: dict[str, int] = defaultdict(int)
    for cache_path in cache_paths:
        dataset = _load_observable_dataset(cache_path)
        scenario = str(dataset.metadata.get("scenario", ""))
        if scenario not in contexts:
            raise ValueError(f"cache uses an undeclared scenario: {scenario!r}")
        files_by_scenario[scenario] += 1
        if dataset.size == 0:
            empty_files_by_scenario[scenario] += 1
            continue
        augmented = augment_dataset_with_right_of_way_context(
            dataset, {scenario: contexts[scenario]}
        )
        start = len(dataset.feature_names)
        rows_by_scenario[scenario].append(
            np.asarray(augmented.features[:, start:], dtype=float)
        )
    extra = sorted(set(rows_by_scenario) - set(city_by_scenario))
    if extra:
        raise ValueError(f"cache contains undeclared scenarios: {extra}")
    missing = sorted(set(city_by_scenario) - set(rows_by_scenario))

    scenario_signatures: dict[str, dict[str, Any]] = {}
    for scenario in sorted(rows_by_scenario):
        values = np.vstack(rows_by_scenario[scenario])
        scenario_signatures[scenario] = {
            "city_group": city_by_scenario[scenario],
            "cache_file_count": files_by_scenario[scenario],
            "empty_cache_file_count": empty_files_by_scenario[scenario],
            "row_count": int(values.shape[0]),
            **_summary(values),
        }

    cities = sorted(set(city_by_scenario.values()))
    city_signatures: dict[str, dict[str, Any]] = {}
    for city in cities:
        scenarios = tuple(
            scenario
            for scenario in sorted(scenario_signatures)
            if city_by_scenario[scenario] == city
        )
        if not scenarios:
            raise ValueError(f"admitted cache contains no nonempty scenario for {city}")
        metric_rows = {
            metric: np.vstack(
                [scenario_signatures[scenario][metric] for scenario in scenarios]
            )
            for metric in ("mean", "std", "q25", "q75")
        }
        city_signatures[city] = {
            "scenarios": list(scenarios),
            "scenario_count": len(scenarios),
            "row_count": sum(
                int(scenario_signatures[scenario]["row_count"])
                for scenario in scenarios
            ),
            **{
                metric: np.mean(values, axis=0).tolist()
                for metric, values in metric_rows.items()
            },
        }
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "information_boundary": (
            "cached_candidate_features_network_topology_and_matched_neighbor_"
            "execution_only_no_prior_or_outcome_arrays_read"
        ),
        "feature_names": list(RIGHT_OF_WAY_CONTEXT_FEATURES),
        "cache_file_count": len(cache_paths),
        "empty_cache_file_count": sum(empty_files_by_scenario.values()),
        "declared_scenario_count": len(city_by_scenario),
        "declared_scenarios_without_admitted_cache": missing,
        "scenario_count": len(scenario_signatures),
        "city_count": len(city_signatures),
        "scenario_signatures": scenario_signatures,
        "city_signatures": city_signatures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150B signature: {args.out}")
    result = build_right_of_way_signatures(
        source_manifest=args.source_manifest,
        source_cache_root=args.source_cache_root,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "DONE",
                "protocol": result["protocol"],
                "cache_file_count": result["cache_file_count"],
                "city_count": result["city_count"],
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
