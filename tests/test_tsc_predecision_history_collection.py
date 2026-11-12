from types import SimpleNamespace

import numpy as np
import pytest

from scripts.cluster import collect_tsc_predecision_history as collector


def test_history_mean_and_sixty_second_linear_trend():
    times = 300 + collector.OFFSETS
    slopes = np.arange(8 * len(collector.FIELDS)).reshape(8, -1) / 10
    values = 7 + times[:, None, None] * slopes
    result = collector.history_summary(values, times, 300)
    np.testing.assert_allclose(result[:, 0::2], 7 + times.mean() * slopes)
    np.testing.assert_allclose(result[:, 1::2], 60 * slopes)
    with pytest.raises(ValueError, match="six preceding observations"):
        collector.history_summary(values, times + 10, 300)  # includes decision time


def _prepared_probe(monkeypatch):
    candidates = [f"phase{i}" for i in range(8)]
    group = dict(group_id="g", checkpoint_sec=300, tls_id="tls", seed=1,
                 scenario="jinan", candidate_states=candidates, reference_index=2)
    features = np.arange(16).reshape(8, 2).astype(float)
    context = np.ones((8, 1))
    metadata = dict(action_group_ids=["g"] * 8, row_tls=["tls"] * 8,
                    row_times=[300] * 8, candidate_states=candidates)
    data = SimpleNamespace(feature_names=("a", "b"), features=features,
                           context=context, metadata=metadata, size=8)
    bank = SimpleNamespace(feature_names=("b", "unused", "a"),
                           features=np.column_stack((features[:, 1], np.zeros(8), features[:, 0])),
                           context=context.copy(), metadata=metadata)
    physical = dict(time_sec=300, vehicles={}, executors={}, pending_ids=())
    monkeypatch.setattr(collector, "physical_snapshot", lambda *_: physical)

    def candidate_features(state, candidate, executor, **kwargs):
        result = np.zeros(len(collector.v3.FEATURE_NAMES_V3))
        for field in collector.FIELDS:
            action = int(candidate.state.removeprefix("phase")) * 100 if field in collector.ACTION_FIELDS else 0
            result[collector.v3.FEATURE_NAMES_V3.index(field)] = state.sim_time + action
        return result

    monkeypatch.setattr(collector.v3, "candidate_features_v3", candidate_features)
    probe = collector.HistoryProbe(None, [group], bank, {"g": {"baseline_snapshot": physical}})
    for time in range(240, 301, 10):
        # Candidate enumeration intentionally differs from the decision's row order.
        info = SimpleNamespace(candidates=[SimpleNamespace(state=x) for x in reversed(candidates)])
        state = SimpleNamespace(sim_time=time, info=info)
        probe.prepare_interval(states={"tls": state}, executors={"tls": None})
    return probe, data


def test_history_aligns_candidate_states_and_excludes_decision_time(monkeypatch):
    probe, data = _prepared_probe(monkeypatch)
    decision = probe.select(data, reference_index=2)
    row = probe.recovered[0]
    assert row["history_times"] == [240, 250, 260, 270, 280, 290]
    result = np.asarray(row["features"])
    index = collector.FEATURE_NAMES.index("hist_mean_green_q")
    np.testing.assert_array_equal(result[:, index], 265 + np.arange(8) * 100)
    np.testing.assert_array_equal(result[:, index + 1], np.full(8, 60))
    assert all(row["parity"].values())
    assert decision.selected_index == 2 and decision.forced_reference


def test_current_dataset_mismatch_cannot_reuse_frozen_labels(monkeypatch):
    probe, data = _prepared_probe(monkeypatch)
    data.features = data.features.copy()
    data.features[0, 0] += 1
    with pytest.raises(ValueError, match='"features": false'):
        probe.select(data, reference_index=2)
    assert probe.recovered == []
