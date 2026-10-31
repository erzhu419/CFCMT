import numpy as np
import pandas as pd

from cf_h2o.eval.sumo_apc_avl_local_snapshot_validation import (
    STATIC_LOCAL_FEATURE_NAMES,
    _make_transitions,
    _selector_choice,
    _static_local_features,
)


def _toy_avl() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_sec": [0.0, 0.0, 60.0, 60.0],
            "city_key": ["toy"] * 4,
            "city": ["Toy"] * 4,
            "vehicle_id": ["v1", "v2", "v1", "v2"],
            "line_key": ["L1"] * 4,
            "line_uid": ["u1"] * 4,
            "segment_idx": [1.0, 3.0, 2.0, 4.0],
            "x": [0.0, 200.0, 100.0, 300.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "speed_mps": [5.0, 7.0, 6.0, 8.0],
            "occupancy": [10.0, 30.0, 12.0, 28.0],
            "last_boardings": [1.0, 2.0, 0.0, 0.0],
            "last_alightings": [0.0, 1.0, 0.0, 0.0],
            "cumulative_boardings": [10.0, 30.0, 11.0, 31.0],
            "cumulative_alightings": [0.0, 5.0, 1.0, 6.0],
        }
    )


def test_make_transitions_uses_next_same_vehicle_snapshot():
    transitions = _make_transitions(_toy_avl(), max_gap_seconds=90.0)

    assert len(transitions) == 2
    assert list(transitions["vehicle_id"]) == ["v1", "v2"]
    assert list(transitions["target_occupancy"]) == [12.0, 28.0]
    assert list(transitions["transition_gap_seconds"]) == [60.0, 60.0]


def test_static_local_features_use_same_line_neighbors():
    transitions = _make_transitions(_toy_avl(), max_gap_seconds=90.0)
    features = _static_local_features(transitions, vehicle_capacity=40.0, segment_scale=10.0)

    assert features.shape == (2, len(STATIC_LOCAL_FEATURE_NAMES))
    assert np.isfinite(features).all()
    first = dict(zip(STATIC_LOCAL_FEATURE_NAMES, features[0]))
    assert np.isclose(first["same_line_occupancy_mean"], 30.0 / 40.0)
    assert np.isclose(first["nearest_occupancy"], 30.0 / 40.0)
    assert np.isclose(first["front_segment_gap_norm"], 2.0 / 10.0)
    assert np.isclose(first["front_occupancy"], 30.0 / 40.0)


def test_selector_keeps_no_local_when_static_is_not_stable(monkeypatch):
    calls = iter(
        [
            {"mse": 1.0},
            {"mse": 0.9},
            {"mse": 1.0},
            {"mse": 1.1},
        ]
    )

    def fake_fit_eval(*_args, **_kwargs):
        return next(calls)

    monkeypatch.setattr("cf_h2o.eval.sumo_apc_avl_local_snapshot_validation._fit_eval", fake_fit_eval)
    out = _selector_choice(["a", "b"], {"a": object(), "b": object()}, ridge=1.0, min_win_rate=0.5, min_improvement=0.0)

    assert out["static_win_rate"] == 0.5
    assert out["selected"] == "no_local"
