"""Frozen forecasts must retain their detector horizon under a longer service view."""
import numpy as np

from experiments.tsc_corridor_service_alignment_20261109.aligned_predictor import AlignedPredictor
from experiments.tsc_corridor_count_fusion_20261109.evidence_model import EvidencePredictor
from experiments.tsc_corridor_residual_transfer_20261109.residual_model import HistogramHead


def flight(eta=8.0, *, t=20, straight=True, vid='car'):
    return dict(edge='road', vid=vid, entry_s=0, age_s=t, on_straight_lane=straight,
                analytic_eta_s=eta, remaining_m=eta*10, speed_limit_mps=10.0,
                queued_at_entry=False, tail_s=2.7)


def row(flights=None, *, t=20, blocked=False):
    return dict(t=t, tls='signal', flights=[flight(t=t)] if flights is None else flights,
                spillback=blocked, unmatched=False)


def counts(absent=0, **events):
    values = [0]*11
    values[0] = absent
    for name, n in events.items():
        values[int(name[1:])] = n
    return values


def test_original_ten_service_columns_and_detector_endpoint_are_identical():
    a = (True, 8, 'missing', False)
    b = (False, 4, 'missing', False)
    target = HistogramHead({a: counts(5, s4=8, s9=12), b: counts(20, s6=3)})
    source = HistogramHead({a: counts(15, s6=13, s10=3), b: counts(7, s8=10)})
    placebo = HistogramHead({a: counts(10, s3=10, s10=11), b: counts(10, s4=7)})
    previous = EvidencePredictor(target, source, placebo)
    aligned = AlignedPredictor(target, source, placebo)
    current = row([flight(), flight(4.0, straight=False, vid='other_lane')])
    old = previous.forecast(current)['road']
    new = aligned.forecast(current)['road']
    assert new.shape == (5, 14)
    np.testing.assert_array_equal(np.concatenate((new[:, :10], new[:, -1:]), axis=1), old)
    assert aligned.history == previous.history


def test_physical_eta_beyond_detector_horizon_does_not_leak_into_service_view():
    empty = HistogramHead()
    predictor = AlignedPredictor(empty, empty, empty)
    # 10.1+2.7 fits within the service view but lies beyond the frozen detector forecast.
    prediction = predictor.forecast(row([flight(10.1)]))['road']
    np.testing.assert_array_equal(prediction, np.zeros((5, 14)))


def test_late_supported_arrival_receives_only_thirteenth_service_second():
    empty = HistogramHead()
    predictor = AlignedPredictor(empty, empty, empty)
    prediction = predictor.forecast(row([flight(9.8)]))['road']
    assert np.all(prediction[:, :12] == 0)  # line ETA=12.5s
    assert np.all(prediction[:, 12] == 1)
    assert np.all(prediction[:, 13] == 1)  # unchanged detector t+10 endpoint


def test_blocked_rows_update_past_positions_before_the_next_observation():
    empty = HistogramHead()
    predictor = AlignedPredictor(empty, empty, empty)
    assert predictor.forecast(row([flight(16, t=10)], t=10, blocked=True)) == {}
    prediction = predictor.forecast(row([flight(8, t=20)]))['road']
    # Past speed8m/s gives detector ETA10; analytic ETA8 is still a separate control.
    assert prediction[3, 10] == 0
    assert prediction[3, 12] == 1
    assert prediction[4, 10] == 1
