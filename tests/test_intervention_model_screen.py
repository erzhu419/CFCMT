from __future__ import annotations

from pathlib import Path

import numpy as np

import cf_h2o.eval.traffic_signal_intervention_model_screen as v130
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster.launch_tsc_intervention_model_screen import (
    SIGNATURE,
    build_spec,
)


def _toy_dataset() -> tuple[MechanismDataset, np.ndarray, np.ndarray]:
    features = []
    targets = []
    groups = []
    references = []
    row_seeds = []
    for seed in range(22):
        for action in range(3):
            features.append([float(seed), float(action)])
            targets.append(0.0 if action == 0 else 0.1 * action)
            groups.append(f"toy:seed{seed}:group0")
            references.append(action == 0)
            row_seeds.append(seed)
    dataset = MechanismDataset(
        feature_names=("total_q", "green_q"),
        features=np.asarray(features, dtype=float),
        context_names=("context",),
        context=np.zeros((len(features), 1), dtype=float),
        priors={},
        targets={"interval_cost": np.asarray(targets, dtype=float)},
        domains=np.asarray(["jinan"] * len(features)),
        metadata={
            "action_group_ids": np.asarray(groups),
            "is_reference": np.asarray(references),
        },
    )
    return dataset, np.asarray(targets), np.asarray(row_seeds)


def test_outer_fold_fits_both_screened_models_once(monkeypatch) -> None:
    dataset, actual, row_seeds = _toy_dataset()
    fit_calls = []

    class FakeModel:
        def __init__(self, arm: str) -> None:
            self.arm = arm

        def fit(self, train: MechanismDataset) -> dict[str, object]:
            fit_calls.append((self.arm, train.size))
            return {"arm": self.arm, "training_rows": train.size}

        def predict(self, evaluation: MechanismDataset) -> dict:
            return {
                "control_cost": {"mean": np.zeros(evaluation.size, dtype=float)}
            }

    monkeypatch.setattr(v130, "_model", lambda arm: FakeModel(arm))
    fold = v130._outer_fold(
        0,
        dataset=dataset,
        actual=actual,
        all_seeds=tuple(range(22)),
        row_seeds=row_seeds,
        eligible=np.ones(dataset.size, dtype=bool),
    )
    assert fit_calls == [(arm, 33) for arm in v130.MODEL_ARMS]
    assert set(fold["arms"]) == set(v130.MODEL_ARMS)


def test_v130_model_configs_target_precision_without_source_inputs() -> None:
    classifier = v130._model("causal_improvement_classifier")
    advantage = v130._model("group_normalized_advantage")
    assert classifier.causal is True
    assert advantage.causal is True
    assert advantage.config.candidate_only is True
    assert advantage.config.balance_candidate_signs is True
    assert (
        advantage.config.target_normalization_protocol
        == v130.GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
    )


def test_outer_fold_real_models_accept_action_group_contract() -> None:
    dataset, actual, row_seeds = _toy_dataset()
    fold = v130._outer_fold(
        0,
        dataset=dataset,
        actual=actual,
        all_seeds=tuple(range(22)),
        row_seeds=row_seeds,
        eligible=np.ones(dataset.size, dtype=bool),
    )
    assert set(fold["arms"]) == set(v130.MODEL_ARMS)
    assert all("fit" in fold["arms"][arm] for arm in v130.MODEL_ARMS)


def test_v130_launcher_has_no_source_prediction_artifact() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        selector_cache_root=Path("/selector"),
        selector_cache_audit_remote=Path("/evidence/audit.json"),
        selector_cache_audit_sha256="a" * 64,
        remote_output_root=Path("/result"),
        nodes=("node001", "node002", "node003", "node005", "node006"),
        cache_workers=20,
        fold_workers=20,
    )
    assert "traffic_signal_intervention_model_screen" in spec["cmd"]
    assert "prediction-artifact" not in spec["cmd"]
    assert spec["signature"] == SIGNATURE
    assert spec["cpu"] == 20
    assert "node004" not in spec["allowed_nodes"]
