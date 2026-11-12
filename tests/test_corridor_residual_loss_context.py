"""Count-loss conservation, actual cancellation and head/prior decomposition."""
import numpy as np
import pytest

from experiments.tsc_corridor_residual_mechanism_20261109 import loss_context as m


def item(t, s, y, *, eta_bin=10, queue=False, speed='(0.5,0.9]', target_n=1000, source_n=1000):
    raw = np.zeros((5, 11))
    raw[0, 10] = .75 if t else .25
    raw[1, 10] = .6 if s else .4
    raw[3, 10] = 1
    decoded = np.zeros((5, 11))
    decoded[0, 10], decoded[1, 10] = t, s
    return dict(flight=dict(edge='road'), context=(True, eta_bin, speed, queue), eta=eta_bin,
                raw_cdfs=raw, decoded=decoded, actual10=y, target_n=target_n, source_n=source_n)


def add(observer, items, *, seed=1, weight=.1, block=0):
    forecast = {'road': sum((x['decoded'] for x in items), np.zeros((5, 11)))}
    observer.add(dict(t=20, tls='signal'), items, forecast, {'common': True},
                 city='jinan', seed=seed, block=block, edge_weights={'common': weight})


def panel(observer):
    return observer.result()['by_city']['jinan']['common']


def test_unique_context_edge_mae_partition_and_feature_partitions_are_each_additive():
    observer = m.LossContextObserver()
    x = item(1, 0, 0)
    add(observer, [x], weight=1/12)
    result = panel(observer)
    assert result['overall']['counts']['weighted_count_loss_delta'] == -1/12
    label = m.context_label(x['context'])
    cell = result['tables']['context'][label]
    assert cell['counts']['attributed_count_loss_delta'] == -1
    assert cell['counts']['weighted_attributed_count_loss_delta'] == -1/12
    assert cell['examples'][0]['actual_count'] == 0
    for check in result['attribution_checks'].values():
        assert check['difference'] == 0


def test_flight_errors_can_increase_without_edge_count_error_changing():
    observer = m.LossContextObserver()
    # One lost true positive and one new false positive cancel in the edge total.
    add(observer, [item(1, 0, 1, eta_bin=9), item(0, 1, 0, eta_bin=10)])
    result = panel(observer)
    counts = result['overall']['counts']
    assert counts['flight_absolute_error_delta'] == 2
    assert counts['count_loss_delta'] == counts['weighted_count_loss_delta'] == 0
    assert counts['group_net_cancellation_edges'] == 1
    assert result['attribution']['context']['__joint__']['counts']['attributed_count_loss_delta'] == 0


def test_multiple_active_contexts_stay_joint_instead_of_arbitrary_order_credit():
    observer = m.LossContextObserver()
    add(observer, [item(0, 1, 0, eta_bin=9), item(0, 1, 0, eta_bin=10)])
    result = panel(observer)
    assert result['overall']['counts']['count_loss_delta'] == 2
    assert result['attribution']['context']['__joint__']['counts']['attributed_count_loss_delta'] == 2
    assert result['attribution']['current_lane']['straight']['counts']['attributed_count_loss_delta'] == 2
    for key, cell in result['tables']['context'].items():
        if key != '__joint__':
            assert cell['counts'].get('attributed_count_loss_delta', 0) == 0
    for name in ('context', 'block_context', *m.FEATURES):
        assert sum(cell['counts'].get('weighted_attributed_count_loss_delta', 0)
                   for cell in result['tables'][name].values()) == pytest.approx(.2)


def test_declared_seed_weights_are_preserved_rather_than_pooled():
    observer = m.LossContextObserver()
    add(observer, [item(0, 1, 0)], seed=1, weight=1/6)
    add(observer, [item(1, 0, 0)], seed=2, weight=1/60)
    result = panel(observer)
    assert result['overall']['counts']['count_loss_delta'] == 0
    assert result['overall']['counts']['weighted_count_loss_delta'] == pytest.approx(.15)
    assert sum(c['counts']['weighted_attributed_count_loss_delta'] for c in result['attribution']['context'].values()) == pytest.approx(.15)


def test_source_head_difference_can_be_entirely_prior_strength():
    observer = m.LossContextObserver()
    x = item(1, 1, 1, target_n=20, source_n=100)
    ft = (20 + 20*.8)/40
    fh = (20 + 100*.8)/120
    x['raw_cdfs'][0, 10] = ft
    x['raw_cdfs'][1, 10] = (ft+fh)/2
    add(observer, [x])
    cell = panel(observer)['tables']['context'][m.context_label(x['context'])]
    assert cell['mean_source_minus_target_cdf10'] == pytest.approx(fh-ft)
    assert cell['mean_distribution_difference_component'] == pytest.approx(0, abs=1e-15)
    assert cell['mean_prior_strength_difference_component'] == pytest.approx(fh-ft)
    assert cell['counts']['cdf_decomposition_residual_sum'] == pytest.approx(0, abs=1e-15)
    assert cell['target_n_range'] == [20, 20]
    assert cell['source_n_range'] == [100, 100]


def test_missing_source_is_not_treated_as_an_observed_source_distribution():
    observer = m.LossContextObserver()
    x = item(1, 1, 1, source_n=0)
    x['raw_cdfs'][1] = x['raw_cdfs'][0]
    add(observer, [x])
    cell = panel(observer)['tables']['context'][m.context_label(x['context'])]
    assert cell['mean_source_cdf10'] is None
    assert cell['source_n_range'] is None
    assert cell['counts']['source_missing_flights'] == 1
