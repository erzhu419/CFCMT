"""Expected service preserves probability information without changing point forecasts."""
import numpy as np

from experiments.tsc_corridor_expected_service_20261109.expected_predictor import ExpectedServicePredictor
from experiments.tsc_corridor_service_alignment_20261109.aligned_predictor import AlignedPredictor
from experiments.tsc_corridor_service_alignment_20261109.evaluate_alignment import choose_aligned
from experiments.tsc_corridor_residual_transfer_20261109.residual_model import HistogramHead


def flight(*, remaining=12.0, vid='car', t=20):
    return dict(edge='road', vid=vid, entry_s=0, age_s=t, on_straight_lane=True,
                analytic_eta_s=remaining/10, remaining_m=remaining,
                speed_limit_mps=10.0, queued_at_entry=False, tail_s=2.7)


def row(flights=None, *, t=20, blocked=False):
    return dict(t=t, tls='signal', states=['rG', 'Gr'], pp=0, gp=0, base=[0, 0],
                protected={'road': [0, ((1 << 13)-1) ^ ((1 << 6)-1)]},
                flights=[flight(t=t)] if flights is None else flights,
                spillback=blocked, unmatched=False)


def heads():
    context = (True, 11, '<=0.5', False)
    # A previously slow car near the detector may arrive within two seconds.
    target = HistogramHead({context: [500, 0, 500]+[0]*8})
    source = HistogramHead({context: [470, 0, 530]+[0]*8})
    placebo = HistogramHead({context: [1000]+[0]*10})
    return target, source, placebo


def set_past(predictor, flights):
    predictor.history = {(f['edge'], f['vid'], f['entry_s']):
                         (10, f['remaining_m']+10, True) for f in flights}


def test_probabilities_around_half_are_preserved_for_legal_service_selection():
    predictor = ExpectedServicePredictor(*heads())
    current = row()
    set_past(predictor, current['flights'])
    values = predictor.forecast(current)['road']
    assert values.shape == (5, 14)
    np.testing.assert_allclose(values[0, 4:13], 500/1020)
    np.testing.assert_allclose(values[1, 4:13], 1030/2020)
    assert values[0, -1] == 0 and values[1, -1] == 1
    # The .49 event has expected service value, although its old hard indicator is zero.
    assert choose_aligned(current, {'road': values})['T'] == 1
    assert choose_aligned(current, predictor.hard_forecasts)['T'] == 0
    np.testing.assert_array_equal(values[3:], predictor.hard_forecasts['road'][3:])


def test_two_vehicle_endpoints_are_decoded_before_edge_aggregation():
    flights = [flight(), flight(remaining=20, vid='second')]
    predictor = ExpectedServicePredictor(*heads())
    set_past(predictor, flights)
    values = predictor.forecast(row(flights))['road']
    np.testing.assert_allclose(values[0, 4:13], 2*500/1020)
    np.testing.assert_allclose(values[1, 4:13], 2*1030/2020)
    assert values[0, -1] == 0 and values[1, -1] == 2
    assert predictor.hard_forecasts['road'][0, 12] == 0
    assert predictor.hard_forecasts['road'][1, 12] == 2


def test_parallel_hard_forecast_exactly_reproduces_old_predictor_and_past_state():
    expected = ExpectedServicePredictor(*heads())
    old = AlignedPredictor(*heads())
    sequence = [row([flight(remaining=22, t=10)], t=10, blocked=True),
                row([flight()]), row([flight(remaining=8, t=30)], t=30)]
    for current in sequence:
        old_values = old.forecast(current)
        new_values = expected.forecast(current)
        assert expected.hard_forecasts.keys() == old_values.keys()
        for edge in old_values:
            np.testing.assert_array_equal(expected.hard_forecasts[edge], old_values[edge])
            np.testing.assert_array_equal(new_values[edge][:, -1], old_values[edge][:, -1])
            np.testing.assert_array_equal(new_values[edge][3:], old_values[edge][3:])
        assert expected.history == old.history
    # A later excluded row must clear previous forecast diagnostics but retain position history.
    assert expected.forecast(row(t=40, blocked=True)) == {}
    assert expected.hard_forecasts == {}


def test_ten_second_detector_cap_and_fractional_tail_are_unchanged():
    empty = HistogramHead()
    predictor = ExpectedServicePredictor(empty, empty, empty)
    outside = predictor.forecast(row([flight(remaining=101)]))['road']
    np.testing.assert_array_equal(outside, np.zeros((5, 14)))
    predictor = ExpectedServicePredictor(empty, empty, empty)
    late = predictor.forecast(row([flight(remaining=98)]))['road']
    assert np.all(late[:, :12] == 0)
    assert np.all(late[:, 12:] == 1)  # detector ETA9.8 + tail2.7 reaches line at12.5s
