from __future__ import annotations

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_target_calibrated_source_crossfit import (
    STACKED_GATE_BASE_FEATURES,
    TARGET_MODEL_FEATURES,
    assemble_crossfit_predictions,
)
from cf_h2o.eval.traffic_signal_stacked_b100_source_gate import _stacked_base
from cf_h2o.traffic_signal.target_calibrated_source_gate import GATE_BASE_FEATURES


def _row(arm: str, fold: int, source_group: str | None, rows: list[int]) -> dict:
    values = np.asarray(rows, dtype=float) + fold
    return {
        "arm": arm,
        "fold": fold,
        "source_group": source_group,
        "heldout_rows": np.asarray(rows, dtype=int),
        "score": values,
        "uncertainty": values + 0.1,
        "trust": np.ones(len(rows), dtype=float),
    }


def test_crossfit_assembly_requires_exact_once_coverage() -> None:
    rows = []
    for fold, heldout in enumerate(([0, 1, 2], [3, 4, 5])):
        rows.append(_row("target_only", fold, None, list(heldout)))
        rows.append(_row("source_aligned", fold, "a", list(heldout)))
        rows.append(_row("source_aligned", fold, "b", list(heldout)))
    target, source = assemble_crossfit_predictions(
        rows, row_count=6, source_groups=("a", "b")
    )
    assert len(target) == 3
    assert set(source) == {"a", "b"}
    assert all(np.isfinite(values).all() for values in target)
    assert all(
        np.isfinite(values).all()
        for predictions in source.values()
        for values in predictions
    )

    with pytest.raises(ValueError, match="overlap"):
        assemble_crossfit_predictions(
            [*rows, _row("target_only", 2, None, [0])],
            row_count=6,
            source_groups=("a", "b"),
        )


def test_stacked_gate_adds_only_target_model_diagnostics_to_local_base() -> None:
    assert STACKED_GATE_BASE_FEATURES == (*GATE_BASE_FEATURES, *TARGET_MODEL_FEATURES)
    assert TARGET_MODEL_FEATURES == (
        "target_model_score",
        "target_model_uncertainty",
    )
    local = np.zeros((4, len(GATE_BASE_FEATURES)), dtype=float)
    stacked = _stacked_base(
        local,
        (
            np.arange(4, dtype=float),
            np.full(4, 0.25),
            np.ones(4),
        ),
    )
    assert stacked.shape == (4, len(STACKED_GATE_BASE_FEATURES))
    np.testing.assert_array_equal(stacked[:, -2], np.arange(4, dtype=float))
    np.testing.assert_array_equal(stacked[:, -1], np.full(4, 0.25))
