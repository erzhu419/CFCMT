import numpy as np

from cf_h2o.eval.gtfsrt_local_snapshot_validation import _ridge_fit, _ridge_predict, _selector_choice


def test_ridge_fit_predicts_linear_signal():
    x = np.arange(20, dtype=np.float64).reshape(-1, 1)
    y = 2.0 + 0.5 * x[:, 0]
    model = _ridge_fit(x, y, ridge=1e-6)
    pred = _ridge_predict(model, x)
    assert np.mean((pred - y) ** 2) < 1e-8


def test_selector_requires_mean_gain_and_win_rate(monkeypatch):
    calls = iter(
        [
            {"mse": 10.0},
            {"mse": 9.0},
            {"mse": 10.0},
            {"mse": 11.0},
        ]
    )

    def fake_fit_eval(*_args, **_kwargs):
        return next(calls)

    monkeypatch.setattr("cf_h2o.eval.gtfsrt_local_snapshot_validation._fit_eval", fake_fit_eval)
    out = _selector_choice(["a", "b"], {"a": object(), "b": object()}, ridge=1.0, min_win_rate=0.5, min_improvement=0.0)

    assert out["mean_static_vs_no_local"] == 1.0
    assert out["static_win_rate"] == 0.5
    assert out["selected"] == "no_local"
