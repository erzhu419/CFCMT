"""All-frame shared motion coefficients and normalized conditional LC readout."""
import numpy as np
import pytest

from experiments.tsc_packet_discrete_braking_20261110.conditional_features import prepare_receiving_context
from experiments.tsc_packet_shared_motion_20261110.api import (
    canonical_features, prepare_plan, shared_decisions, evaluate_shared,
)
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import packet
from test_tsc_packet_lateral_order_review import two_lateral_actors


def test_shared_beta_scores_background_and_preserves_one_sided_mean_and_lc_normalization():
    kernel, unused, unused_actions = two_lateral_actors()
    background = packet('background', 'a2', 70, 3)
    kernel.state['packets']['background'] = background
    kernel.state['intents']['background'] = dict(road_id='a', choice='END')
    kernel.noise.register('background', kernel.state['time_s'])
    kernel.seen.add('background')
    kernel.state['lane_sensors'] = kernel.sensor_operator._reconstruct(kernel.state['packets'])
    kernel._connections = {key: kernel._connection(row) for key, row in kernel.state['packets'].items()}
    prepare_receiving_context(kernel)
    phi = canonical_features(kernel)
    assert set(phi) == set(kernel.state['packets']) == {'focal', 'front', 'background'}
    for key, row in kernel.state['packets'].items():
        assert phi[key].shape == (38,) and phi[key][0] == 1.
        expected = kernel.model.features(kernel._context(row, kernel.state['packets'], *kernel._connections[key]))
        np.testing.assert_array_equal(phi[key], expected)
    plan = prepare_plan(kernel, 'focal', ['focal', 'front'])
    assert 'background' not in plan['local'] and 'background' in plan['phi']
    zero = np.zeros(38)
    direction = zero.copy(); direction[0] = 1.
    epsilon = 1e-4
    readouts = {}
    for name, beta in (('zero', zero), ('negative', -epsilon*direction), ('positive', epsilon*direction)):
        decisions = shared_decisions(kernel, phi, beta)
        assert set(decisions) == set(kernel.state['packets'])
        assert decisions['background']['motion_residual_mps'] == pytest.approx(beta[0])
        result = evaluate_shared(kernel, plan, beta)
        assert result['decision_count'] == 3
        assert result['motion']['focal']['mean_matches_proposal']
        assert result['motion']['focal']['body_slack_m'] > 0
        assert result['motion']['focal']['spatial_slack_m'] > 0
        assert result['class_mass'].sum() == pytest.approx(1., abs=1e-12)
        np.testing.assert_allclose(result['grad_mass_left'].sum(axis=0), 0., atol=1e-12)
        np.testing.assert_allclose(result['grad_mass_right'].sum(axis=0), 0., atol=1e-12)
        readouts[name] = result
    assert readouts['negative']['negative_residual_background_count'] == 1
    baseline = readouts['zero']['motion']['focal']['mean_mps']
    assert (readouts['negative']['motion']['focal']['mean_mps']-baseline)/epsilon == pytest.approx(-1., abs=1e-9)
    assert (readouts['positive']['motion']['focal']['mean_mps']-baseline)/epsilon == pytest.approx(0., abs=1e-12)
