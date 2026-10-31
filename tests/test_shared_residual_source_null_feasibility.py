from __future__ import annotations

from pathlib import Path

import numpy as np

import cf_h2o.eval.traffic_signal_shared_residual_source_null_feasibility as v129
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster.launch_tsc_shared_residual_source_null_feasibility import (
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
    return (
        dataset,
        np.asarray(targets, dtype=float),
        np.asarray(row_seeds, dtype=int),
    )


def test_shared_prior_predictions_preserve_source_increment() -> None:
    residual = np.asarray([0.0, 1.0, 2.0])
    mask = np.asarray([True, False, True, True])
    priors = {
        "target_only": np.asarray([0.0, 8.0, 0.5, 1.0]),
        "source_aligned": np.asarray([0.2, 9.0, 0.1, -0.5]),
    }
    predictions = v129._shared_prior_predictions(residual, priors, mask)
    np.testing.assert_allclose(
        predictions["source_aligned"] - predictions["target_only"],
        [0.2, -0.4, -1.5],
    )


def test_outer_fold_fits_one_residual_for_all_prior_arms(monkeypatch) -> None:
    dataset, actual, row_seeds = _toy_dataset()
    fit_calls = []

    class FakeResidual:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, train: MechanismDataset) -> dict[str, int]:
            fit_calls.append(train.size)
            return {"training_rows": train.size}

        def predict(self, evaluation: MechanismDataset) -> dict:
            return {
                "control_cost": {"mean": np.zeros(evaluation.size, dtype=float)}
            }

    monkeypatch.setattr(v129, "CausalReferenceResidualRegressor", FakeResidual)
    priors = {
        "target_only": np.zeros(dataset.size, dtype=float),
        "source_aligned": np.tile([0.0, -0.2, 0.2], 22),
        "source_placebo": np.tile([0.0, 0.2, -0.2], 22),
    }
    fold = v129._outer_fold(
        0,
        target_residual_dataset=dataset,
        actual=actual,
        priors=priors,
        all_seeds=tuple(range(22)),
        row_seeds=row_seeds,
        eligible=np.ones(dataset.size, dtype=bool),
    )
    assert fit_calls == [33]
    assert fold["shared_target_residual_fit"] == {"training_rows": 33}
    assert all(
        fold["arms"][arm]["residual_fit"] == "shared_target_only_residual"
        for arm in v129.BASE_ARMS
    )


def test_v129_launcher_replaces_only_the_frozen_method_module() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        prediction_artifact_remote=Path("/evidence/predictions.pkl"),
        prediction_artifact_sha256="a" * 64,
        selector_cache_root=Path("/selector"),
        selector_cache_audit_remote=Path("/evidence/audit.json"),
        selector_cache_audit_sha256="b" * 64,
        remote_output_root=Path("/result"),
        nodes=("node001", "node002", "node003", "node005", "node006"),
        cache_workers=20,
        fold_workers=20,
    )
    assert "shared_residual_source_null_feasibility" in spec["cmd"]
    assert "anchored_source_null_residual_feasibility" not in spec["cmd"]
    assert spec["signature"] == SIGNATURE
    assert spec["cpu"] == 20
    assert "node004" not in spec["allowed_nodes"]


def test_v129_main_publishes_scientific_rejection_as_completed_task(
    monkeypatch,
    tmp_path: Path,
) -> None:
    result = {
        "protocol": v129.RESULT_PROTOCOL,
        "development_gate": {"passed": False, "decision": "reject"},
        "arm_summaries": {},
        "paired_source_contribution": {},
        "runtime_seconds": 1.0,
    }
    written = []
    monkeypatch.setattr(
        v129,
        "run_shared_residual_source_null_feasibility",
        lambda **_: result,
    )
    monkeypatch.setattr(
        v129,
        "atomic_write_json",
        lambda path, payload: written.append((path, payload)),
    )
    output = tmp_path / "result.json"
    returncode = v129.main(
        [
            "--prediction-artifact",
            "prediction.pkl",
            "--prediction-artifact-sha256",
            "a" * 64,
            "--selector-cache-root",
            "cache",
            "--selector-manifest",
            "manifest.json",
            "--selector-cache-audit",
            "audit.json",
            "--selector-cache-audit-sha256",
            "b" * 64,
            "--selector-scenario",
            "scenario",
            "--selector-seeds",
            "1",
            "--selector-collection-shards",
            "1",
            "--conversion-root",
            "conversion",
            "--out",
            str(output),
        ]
    )
    assert returncode == 0
    assert written == [(output, result)]
