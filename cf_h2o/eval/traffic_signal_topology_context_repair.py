"""Repair cached static topology context without rerunning counterfactuals."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTEXT_NAMES_V3,
    _scenario_context_v3,
    policy_consistent_mechanism_priors_v4,
    rollout_prefix_target_name,
    uncalibrated_mechanism_priors_v3,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    _start_sumo,
    _tls_phase_infos,
)
from cf_h2o.sumo_runtime import load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import TrafficSignalBenchmarkManifest
from cf_h2o.traffic_signal.dataset_cache import (
    load_mechanism_dataset,
    save_mechanism_dataset,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


REPAIR_PROTOCOL = "static-topology-multihop-context-repair-v1"


def corrected_scenario_contexts(
    manifest: TrafficSignalBenchmarkManifest,
    *,
    control_interval_sec: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Compute one label-free context per scenario with live TLS semantics."""

    result: dict[str, np.ndarray] = {}
    for scenario, sumocfg in manifest.sumocfgs.items():
        sumo_api = load_libsumo()
        _start_sumo(sumo_api, sumocfg, int(seed))
        try:
            infos = _tls_phase_infos(sumo_api)
            result[scenario] = _scenario_context_v3(
                sumocfg=sumocfg,
                infos=infos,
                control_interval_sec=int(control_interval_sec),
            )
        finally:
            sumo_api.close()
    return result


def repair_dataset_topology_context(
    dataset: MechanismDataset,
    corrected_context: np.ndarray,
) -> MechanismDataset:
    """Replace static context and every analytic prior derived from it."""

    if tuple(dataset.context_names) != tuple(CONTEXT_NAMES_V3):
        raise ValueError("topology repair requires the V3 context schema")
    corrected = np.asarray(corrected_context, dtype=float)
    if corrected.shape != (len(CONTEXT_NAMES_V3),) or not np.all(
        np.isfinite(corrected)
    ):
        raise ValueError("corrected scenario context has an invalid shape")
    if dataset.size and not np.allclose(
        dataset.context,
        dataset.context[0][None, :],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("cached scenario context is not static")

    context = np.repeat(corrected[None, :], dataset.size, axis=0)
    control_interval_sec = int(dataset.metadata["control_interval_sec"])
    horizon_intervals = int(
        dataset.metadata["counterfactual_horizon_intervals"]
    )
    repaired_priors = policy_consistent_mechanism_priors_v4(
        np.asarray(dataset.features, dtype=float),
        context,
        control_interval_sec=control_interval_sec,
        counterfactual_horizon_intervals=horizon_intervals,
    )
    for horizon in dataset.metadata.get("rollout_prefix_horizons_sec", ()):
        target_name = rollout_prefix_target_name(int(horizon))
        repaired_priors[target_name] = np.asarray(
            uncalibrated_mechanism_priors_v3(
                np.asarray(dataset.features, dtype=float),
                context,
                control_interval_sec=int(horizon),
            )["interval_cost"],
            dtype=float,
        ).copy()
    missing = set(dataset.priors) - set(repaired_priors)
    if missing:
        raise ValueError(f"topology repair cannot reconstruct priors: {sorted(missing)}")

    old_context = (
        np.asarray(dataset.context[0], dtype=float)
        if dataset.size
        else corrected.copy()
    )
    return replace(
        dataset,
        context=context,
        priors={
            name: np.asarray(repaired_priors[name], dtype=float)
            for name in dataset.priors
        },
        metadata={
            **dict(dataset.metadata),
            "static_topology_context_repair": {
                "protocol": REPAIR_PROTOCOL,
                "label_free": True,
                "counterfactual_features_unchanged": True,
                "counterfactual_targets_unchanged": True,
                "analytic_priors_recomputed": True,
                "maximum_absolute_context_change": float(
                    np.max(np.abs(corrected - old_context))
                ),
            },
        },
    )


def repair_cache_tree(
    *,
    source_root: Path,
    destination_root: Path,
    contexts: Mapping[str, np.ndarray],
    workers: int = 8,
) -> dict[str, Any]:
    """Write a repaired cache tree while preserving parent cache identities."""

    source_root = Path(source_root)
    destination_root = Path(destination_root)
    files = tuple(sorted(source_root.glob("*.npz")))
    if not files:
        raise ValueError(f"counterfactual cache is empty: {source_root}")
    if destination_root.exists() and any(destination_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite repaired cache: {destination_root}"
        )
    destination_root.mkdir(parents=True, exist_ok=True)

    def repair_one(path: Path) -> dict[str, Any]:
        dataset = load_mechanism_dataset(path)
        scenario = str(dataset.metadata.get("scenario", ""))
        if scenario not in contexts:
            raise ValueError(f"cache scenario is absent from manifest: {scenario}")
        repaired = repair_dataset_topology_context(dataset, contexts[scenario])
        if not np.array_equal(repaired.features, dataset.features) or any(
            not np.array_equal(repaired.targets[name], dataset.targets[name])
            for name in dataset.targets
        ):
            raise AssertionError("topology repair changed counterfactual evidence")
        output = destination_root / path.name
        save_mechanism_dataset(output, repaired)
        prior_change = max(
            (
                float(
                    np.max(
                        np.abs(
                            np.asarray(repaired.priors[name], dtype=float)
                            - np.asarray(dataset.priors[name], dtype=float)
                        )
                    )
                )
                for name in dataset.priors
                if dataset.size
            ),
            default=0.0,
        )
        context_change = float(
            repaired.metadata["static_topology_context_repair"][
                "maximum_absolute_context_change"
            ]
        )
        return {
            "scenario": scenario,
            "rows": int(dataset.size),
            "maximum_absolute_context_change": context_change,
            "maximum_absolute_prior_change": prior_change,
        }

    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as pool:
        records = list(pool.map(repair_one, files))
    scenario_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        scenario_records.setdefault(str(record["scenario"]), []).append(record)
    return {
        "protocol": REPAIR_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.resolve()),
        "destination_root": str(destination_root.resolve()),
        "workers": max(int(workers), 1),
        "file_count": len(records),
        "row_count": sum(int(record["rows"]) for record in records),
        "scenarios": {
            scenario: {
                "file_count": len(values),
                "row_count": sum(int(value["rows"]) for value in values),
                "maximum_absolute_context_change": max(
                    float(value["maximum_absolute_context_change"])
                    for value in values
                ),
                "maximum_absolute_prior_change": max(
                    float(value["maximum_absolute_prior_change"])
                    for value in values
                ),
            }
            for scenario, values in sorted(scenario_records.items())
        },
    }
