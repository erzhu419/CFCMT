"""A corrected late prediction must be counted after forecasts are frozen."""
import gzip
import json

from experiments.tsc_corridor_residual_mechanism_20261109 import replay as r
from experiments.tsc_corridor_residual_transfer_20261109.residual_model import HistogramHead


def test_late_prediction_gain_is_observed_without_changing_frozen_requests(tmp_path):
    flight = dict(edge='road', vid='car', entry_s=0, age_s=20, on_straight_lane=True,
                  analytic_eta_s=9.0, remaining_m=90.0, speed_limit_mps=10.0,
                  queued_at_entry=False, tail_s=2.7)
    row = dict(t=20, tls='signal', states=['Gr', 'rG'], pp=0, gp=0, base=[0, 0],
               protected={'road': [1023, 0]}, flights=[flight], spillback=False, unmatched=False)
    outcome = dict(edge='road', vid='car', entry_s=0, end_s=32, status='straight')
    for name, record in [('decisions', row), ('labels', outcome)]:
        with gzip.open(tmp_path / (name+'.jsonl.gz'), 'wt') as stream:
            stream.write(json.dumps(record)+'\n')
    context = (True, 9, 'missing', False)
    target = HistogramHead({context: [40]+[0]*9+[60]})
    source = HistogramHead({context: [400]+[0]*10})
    predictor = r.DetailedPredictor(target, source, source)

    class CommonSupport:
        def forecast(self, row):
            return {'road': True}

    maes = dict(T=1.0, S=0.0, P=0.0, history=1.0, analytic=1.0)
    differences = {f'{a}_vs_{b}': 0 for a, b in r.PAIRS}
    previous = dict(scopes={scope: dict(counts={'scored_edge_epochs': 1}, mae=maes,
                                      action_differences=differences) for scope in r.SCOPES})
    decisions, losses = r.DecisionPathObserver(), r.LossContextObserver()
    result = r.replay_hour(tmp_path, 'jinan', 1, 0, predictor, CommonSupport(),
                           previous, decisions, losses)
    for scope in r.SCOPES:
        assert result['reproduction'][scope]['exact_reproduction']
        counts = decisions.result()[scope]['unlocked']['counts']
        assert counts['individual_endpoint_improved_flights'] == 1
        assert counts['endpoint_improved_but_no_service_change_flights'] == 1
        assert counts['request_changed'] == 0
        assert losses.result()['by_city']['jinan'][scope]['overall']['counts']['weighted_count_loss_delta'] == -1/6
