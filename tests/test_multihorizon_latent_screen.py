from __future__ import annotations

from pathlib import Path

import numpy as np

from cf_h2o.eval.traffic_signal_multihorizon_latent_screen import (
    _atomic_oof_npz,
    _fit_fold,
)
from cf_h2o.eval.traffic_signal_state_conditioned_latent_diagnostic import (
    BASE_LATENT_MODEL_FAMILY,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    TLS_TIME_PHASE_PAIR_SCOPE,
)


def _contrast() -> MechanismDataset:
    features = []
    groups = []
    references = []
    states = []
    tls = []
    times = []
    target_10 = []
    target_30 = []
    for seed in (11, 22, 33):
        for time_sec in (100.0, 400.0):
            group = f"test:seed{seed}:{int(time_sec)}:tls0"
            for action in range(3):
                features.append([float(action)])
                groups.append(group)
                references.append(action == 0)
                states.append(f"phase{action}")
                tls.append("tls0")
                times.append(time_sec)
                target_10.append((0.0, -1.0, 0.5)[action])
                target_30.append((0.0, 0.5, -1.0)[action])
    targets = {
        "prefix_mean_cost_10s": np.asarray(target_10, dtype=float),
        "prefix_mean_cost_30s": np.asarray(target_30, dtype=float),
    }
    return MechanismDataset(
        feature_names=("action",),
        features=np.asarray(features, dtype=float),
        context_names=("demand",),
        context=np.zeros((len(features), 1), dtype=float),
        priors={name: np.zeros(len(features), dtype=float) for name in targets},
        targets=targets,
        domains=np.asarray(["test"] * len(features)),
        metadata={
            "action_group_ids": groups,
            "is_reference": references,
            "candidate_states": states,
            "row_tls": tls,
            "row_times": times,
        },
    )


def _spec(key: str, target_name: str) -> dict[str, object]:
    return {
        "key": key,
        "family": BASE_LATENT_MODEL_FAMILY,
        "target_name": target_name,
        "config": {
            "scope": TLS_TIME_PHASE_PAIR_SCOPE,
            "shrinkage": 0.0,
            "time_bin_sec": 300,
            "uncertainty_quantile": 0.9,
        },
    }


def test_multihorizon_fold_is_seed_disjoint_and_target_specific(tmp_path: Path) -> None:
    fold = _fit_fold(
        heldout_seed=33,
        contrast=_contrast(),
        expanded_specs=(
            _spec("base::h10s", "prefix_mean_cost_10s"),
            _spec("base::h30s", "prefix_mean_cost_30s"),
        ),
        scenario="test",
        expected_action_count=3,
    )
    records = [
        row for result in fold["model_results"] for row in result["records"]
    ]

    assert fold["training_validation_disjoint"] is True
    assert fold["heldout_absent_from_training"] is True
    assert {row["simulator_seed"] for row in records} == {33}
    assert {row["target_name"] for row in records} == {
        "prefix_mean_cost_10s",
        "prefix_mean_cost_30s",
    }
    assert len(records) == 4

    path = tmp_path / "oof.npz"
    _atomic_oof_npz(path, records)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["group_id"].shape == (4,)
        assert set(archive["target_name"].tolist()) == {
            "prefix_mean_cost_10s",
            "prefix_mean_cost_30s",
        }
