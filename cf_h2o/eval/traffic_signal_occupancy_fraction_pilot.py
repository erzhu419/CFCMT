"""Fresh Cologne cache and ranker check for the occupancy fraction contract."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _collect_worker,
    _counterfactual_cache_identity,
    _counterfactual_cache_path,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.action_ranker import PairwiseActionAdvantageRegressor
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL


PROTOCOL = "tsc-occupancy-fraction-fresh-cache-ranker-pilot-v1"


def run_pilot(*, sumocfg: Path, cache_root: Path, out: Path) -> dict:
    started = time.monotonic()
    settings = {
        "sumocfg": Path(sumocfg),
        "scenario": "cologne1",
        "duration_sec": 300.0,
        "control_interval_sec": 30,
        "warmup_sec": 120.0,
        "seed": 41242,
        "max_focal_tls": 1,
        "counterfactual_horizon_intervals": 1,
        "behavior_policy": "phase_pressure",
        "counterfactual_cost_mode": "halted_queue",
        "collection_shard_index": 0,
        "collection_shard_count": 1,
    }
    identity = _counterfactual_cache_identity(**settings)
    cache_path = _counterfactual_cache_path(cache_root=cache_root, **settings)
    if cache_path.exists():
        raise ValueError("occupancy pilot requires a fresh cache output directory")
    payload = {**settings, "cache_identity": identity, "cache_path": str(cache_path)}
    _, _, _, fresh = _collect_worker(payload)
    _, _, _, loaded = _collect_worker(payload)
    arrays_equal = (
        fresh.feature_names == loaded.feature_names
        and fresh.metadata == loaded.metadata
        and np.array_equal(fresh.features, loaded.features)
        and np.array_equal(fresh.context, loaded.context)
        and all(np.array_equal(fresh.priors[name], loaded.priors[name]) for name in fresh.priors)
        and all(np.array_equal(fresh.targets[name], loaded.targets[name]) for name in fresh.targets)
    )
    if not arrays_equal or loaded.size == 0:
        raise ValueError("fresh occupancy cache did not preserve nonempty labelled rows")
    if loaded.metadata.get("occupancy_equation_protocol") != OCCUPANCY_EQUATION_PROTOCOL:
        raise ValueError("fresh occupancy cache has the wrong equation contract")
    contrast = build_action_contrast_dataset(
        loaded, reference_policy="phase_pressure", contrast_features=CONTRAST_FEATURES_V3
    )
    model = PairwiseActionAdvantageRegressor(causal=True)
    fit_diagnostics = model.fit(contrast)
    prediction = model.predict(contrast)["control_cost"]["mean"]
    model_path = Path(out).with_name("fresh_ranker.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "protocol": PROTOCOL,
        "occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
        "feature_binding": "stored_feature_names",
        "model": model,
    }
    model_path.write_bytes(pickle.dumps(model_payload, protocol=pickle.HIGHEST_PROTOCOL))
    restored = pickle.loads(model_path.read_bytes())
    restored_prediction = restored["model"].predict(contrast)["control_cost"]["mean"]
    if not np.all(np.isfinite(prediction)) or not np.array_equal(prediction, restored_prediction):
        raise ValueError("fresh fitted ranker did not preserve finite predictions")
    index = {name: offset for offset, name in enumerate(loaded.feature_names)}
    occupancy = loaded.features[:, index["green_down_occ"]]
    expected_service = np.maximum(
        loaded.features[:, index["green_q"]] - loaded.features[:, index["green_down_q"]],
        0.0,
    ) * np.maximum(0.0, 1.0 - occupancy / 1.2)
    service_error = float(np.max(np.abs(
        loaded.features[:, index["service_pressure"]] - expected_service
    )))
    if service_error > 1e-12:
        raise ValueError("fresh service pressure differs from the fraction equation")
    result = {
        "protocol": PROTOCOL,
        "occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
        "status": "PASS",
        "scientific_status": "operational_cache_and_ranker_check_only",
        "settings": {**settings, "sumocfg": str(sumocfg)},
        "prediction_horizon_sec": 30,
        "fresh_labels_collected": True,
        "training_set_prediction_only": True,
        "cache_round_trip_equal": arrays_equal,
        "cache_path": str(cache_path),
        "cache_version": identity["version"],
        "row_count": loaded.size,
        "action_group_count": len(set(loaded.metadata["action_group_ids"])),
        "retained_sample_times_sec": sorted(set(loaded.metadata["row_times"])),
        "retained_candidates_by_group": {
            group: loaded.metadata["action_group_ids"].count(group)
            for group in sorted(set(loaded.metadata["action_group_ids"]))
        },
        "maximum_downstream_occupancy_fraction": float(np.max(occupancy)),
        "service_pressure_max_abs_error": service_error,
        "fresh_ranker_model_path": str(model_path),
        "ranker_used_constant_prediction": model.model is None,
        "ranker_fit": fit_diagnostics,
        "restored_ranker_predictions_equal": True,
        "complete_v150k_utility_payload_tested": False,
        "runtime_seconds": float(time.monotonic() - started),
    }
    atomic_write_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumocfg", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_pilot(sumocfg=args.sumocfg, cache_root=args.cache_root, out=args.out)
    print(f"{result['status']}: {result['action_group_count']} fresh groups; operational check only")


if __name__ == "__main__":
    main()
